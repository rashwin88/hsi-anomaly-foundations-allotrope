"""S3 side of the upload path: multipart uploads with per-part SHA-256.

Bytes never pass through the api. The client hashes each fixed-size part,
the api signs that hash into the part's presigned URL, and S3 rejects any
part whose body doesn't match. On completion S3 stores a composite checksum
(`b64(sha256(part1_raw || part2_raw || …))-N`) which `complete_upload`
recomputes from the declared part hashes and checks.

S3 cannot produce a whole-file SHA-256 for a multipart object, so the
whole-file digest in `dataset_files.sha256` is client-declared; integrity
of the stored bytes rests on the per-part checks.
"""

import base64
import gzip
import hashlib
import io
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import boto3
from botocore.config import Config

from .config import settings

PART_SIZE = 64 * 1024 * 1024          # fixed; the client splits on this boundary
URL_TTL_SECONDS = 24 * 3600           # re-request (resume) to get fresh URLs


@cache
def client():
    """Regional SigV4 client, so presigned URLs need no redirect."""
    # `or None`: an empty INVENTORY_AWS_PROFILE (containers) means env-var creds.
    session = boto3.Session(profile_name=settings.aws_profile or None, region_name=settings.region)
    return session.client(
        "s3",
        endpoint_url=f"https://s3.{settings.region}.amazonaws.com",
        # Server-side imports run many UploadPartCopy calls at once; the
        # default pool of 10 connections would serialise them.
        config=Config(signature_version="s3v4", max_pool_connections=64),
    )


def part_count(size_bytes: int) -> int:
    """Number of PART_SIZE parts; an empty file is still one (empty) part."""
    return max(1, -(-size_bytes // PART_SIZE))


def start_upload(key: str) -> str:
    """Open a multipart upload with SHA-256 part checksums; returns its id."""
    r = client().create_multipart_upload(
        Bucket=settings.bucket, Key=key, ChecksumAlgorithm="SHA256"
    )
    return r["UploadId"]


def part_urls(key: str, upload_id: str, part_sha256_b64: list[str]) -> list[str]:
    """One presigned PUT per part, each bound to that part's SHA-256."""
    return [
        client().generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": settings.bucket, "Key": key, "UploadId": upload_id,
                "PartNumber": n, "ChecksumSHA256": digest,
            },
            ExpiresIn=URL_TTL_SECONDS,
        )
        for n, digest in enumerate(part_sha256_b64, start=1)
    ]


def uploaded_parts(key: str, upload_id: str) -> dict[int, dict]:
    """Parts S3 already holds, by part number (for resume and completion)."""
    out: dict[int, dict] = {}
    for page in client().get_paginator("list_parts").paginate(
        Bucket=settings.bucket, Key=key, UploadId=upload_id
    ):
        for p in page.get("Parts", []):
            out[p["PartNumber"]] = p
    return out


def abort_upload(key: str, upload_id: str) -> None:
    client().abort_multipart_upload(Bucket=settings.bucket, Key=key, UploadId=upload_id)


def composite_sha256(part_sha256_b64: list[str]) -> str:
    """The checksum S3 stores for a multipart object: hash of the raw part hashes."""
    raw = b"".join(base64.b64decode(d) for d in part_sha256_b64)
    return f"{base64.b64encode(hashlib.sha256(raw).digest()).decode()}-{len(part_sha256_b64)}"


class UploadIncomplete(Exception):
    """Raised by complete_upload when S3 is missing parts; `.missing` lists them."""

    def __init__(self, missing: list[int]):
        super().__init__(f"missing parts {missing}")
        self.missing = missing


def complete_upload(key: str, upload_id: str, part_sha256_b64: list[str], size_bytes: int) -> str:
    """Close the upload, verify size and composite checksum; return the version id.

    Raises UploadIncomplete if any part is absent, ValueError if what S3 holds
    disagrees with what was declared (the object is then deleted, so a bad
    file never sits in the bucket looking finished).
    """
    held = uploaded_parts(key, upload_id)
    missing = [n for n in range(1, len(part_sha256_b64) + 1) if n not in held]
    if missing:
        raise UploadIncomplete(missing)
    client().complete_multipart_upload(
        Bucket=settings.bucket, Key=key, UploadId=upload_id,
        MultipartUpload={"Parts": [
            {"PartNumber": n, "ETag": held[n]["ETag"], "ChecksumSHA256": held[n]["ChecksumSHA256"]}
            for n in range(1, len(part_sha256_b64) + 1)
        ]},
    )
    head = client().head_object(Bucket=settings.bucket, Key=key, ChecksumMode="ENABLED")
    expected = composite_sha256(part_sha256_b64)
    if head["ContentLength"] != size_bytes or head.get("ChecksumSHA256") != expected:
        client().delete_object(Bucket=settings.bucket, Key=key, VersionId=head["VersionId"])
        raise ValueError(
            f"{key}: S3 holds {head['ContentLength']} B / {head.get('ChecksumSHA256')}, "
            f"declared {size_bytes} B / {expected}"
        )
    return head["VersionId"]


MANIFEST_VERSION = 1


def put_manifest(prefix: str, body: dict) -> None:
    """Write `<prefix>/manifest.json`. The bucket is versioned, so every
    rewrite (finalise, tag changes) keeps the previous one recoverable."""
    client().put_object(
        Bucket=settings.bucket, Key=f"{prefix}/manifest.json",
        Body=json.dumps({"manifest_version": MANIFEST_VERSION, **body}, indent=2).encode(),
        ContentType="application/json", ChecksumAlgorithm="SHA256",
    )


def put_files_manifest(prefix: str, rows: Iterable[dict]) -> None:
    """Write `<prefix>/files.jsonl.gz`: one JSON line per file. Streamed into
    a gzip buffer, so 500k rows cost tens of MB, not one giant JSON string."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write(json.dumps(r, separators=(",", ":")).encode() + b"\n")
    client().put_object(
        Bucket=settings.bucket, Key=f"{prefix}/files.jsonl.gz", Body=buf.getvalue(),
        ContentType="application/x-ndjson", ContentEncoding="gzip", ChecksumAlgorithm="SHA256",
    )


def download_url(key: str, version_id: str | None) -> str:
    """Presigned GET, pinned to `version_id` so a later overwrite can't swap
    the bytes behind a verified checksum."""
    params = {"Bucket": settings.bucket, "Key": key}
    if version_id:
        params["VersionId"] = version_id
    return client().generate_presigned_url("get_object", Params=params, ExpiresIn=URL_TTL_SECONDS)


def hex_to_b64(sha256_hex: str) -> str:
    return base64.b64encode(bytes.fromhex(sha256_hex)).decode()


def verify_single_put(key: str, size_bytes: int, sha256_hex: str) -> tuple[str | None, str | None]:
    """(version_id, None) if `key` holds exactly these bytes, else (None, reason).

    For a single-PUT object written with ChecksumSHA256, S3's stored checksum
    is the true whole-file SHA-256 (no `-N` part suffix), so this HEAD proves
    the bytes without reading them.
    """
    try:
        h = client().head_object(Bucket=settings.bucket, Key=key, ChecksumMode="ENABLED")
    except client().exceptions.ClientError as e:
        return None, f"not stored ({e.response['Error'].get('Code', 'error')})"
    if h["ContentLength"] != size_bytes:
        return None, f"size {h['ContentLength']} != declared {size_bytes}"
    if h.get("ChecksumSHA256") != hex_to_b64(sha256_hex):
        return None, "stored SHA-256 missing or different (must be a single PUT with ChecksumSHA256)"
    return h.get("VersionId"), None


def delete_all_versions(prefix: str) -> tuple[int, int]:
    """Permanently delete every version and delete marker under `prefix`.
    Returns (objects, bytes) removed. Irreversible — callers gate it."""
    objects = size = 0
    for page in client().get_paginator("list_object_versions").paginate(Bucket=settings.bucket, Prefix=prefix):
        batch = [{"Key": v["Key"], "VersionId": v["VersionId"]}
                 for v in page.get("Versions", []) + page.get("DeleteMarkers", [])]
        objects += len(page.get("Versions", []))
        size += sum(v["Size"] for v in page.get("Versions", []))
        for i in range(0, len(batch), 1000):
            client().delete_objects(Bucket=settings.bucket, Delete={"Objects": batch[i:i + 1000], "Quiet": True})
    return objects, size


COPY_WORKERS = 16                     # UploadPartCopy calls in flight per imported file


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key/with/slashes` → (bucket, key)."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not key:
        raise ValueError(f"not an s3://bucket/key URI: {uri}")
    return bucket, key


def source_size(bucket: str, key: str) -> int:
    return client().head_object(Bucket=bucket, Key=key)["ContentLength"]


def copy_parts(
    key: str, upload_id: str, src_bucket: str, src_key: str, size_bytes: int,
    part_sha256_b64: list[str], skip: set[int],
) -> list[int]:
    """Server-side copy of every part not in `skip`, in parallel.

    The upload was opened with ChecksumAlgorithm=SHA256, so S3 computes each
    copied part's SHA-256 itself; any part whose result differs from the
    declared hash is returned (verified on real S3, 2026-10-13). No bytes
    pass through this process.
    """
    def one(n: int) -> int | None:
        lo = (n - 1) * PART_SIZE
        hi = min(lo + PART_SIZE, size_bytes) - 1
        r = client().upload_part_copy(
            Bucket=settings.bucket, Key=key, UploadId=upload_id, PartNumber=n,
            CopySource={"Bucket": src_bucket, "Key": src_key},
            # An empty source has no byte range to name; copy it whole.
            **({"CopySourceRange": f"bytes={lo}-{hi}"} if size_bytes else {}),
        )
        return None if r["CopyPartResult"].get("ChecksumSHA256") == part_sha256_b64[n - 1] else n

    todo = [n for n in range(1, len(part_sha256_b64) + 1) if n not in skip]
    with ThreadPoolExecutor(max_workers=COPY_WORKERS) as pool:
        return sorted(n for n in pool.map(one, todo) if n is not None)
