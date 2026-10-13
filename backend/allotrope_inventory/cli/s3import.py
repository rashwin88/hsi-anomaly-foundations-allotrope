"""`import-s3`: bring objects already in S3 into the inventory, server-side.

1. List the objects under an s3:// prefix (each `<key>.md5` sidecar is
   paired with its object rather than treated as payload-only).
2. Read every object once — ranged GETs, several parts in flight — and
   compute the whole-file SHA-256, per-part SHA-256 and MD5 in one ordered
   pass. Objects hash concurrently; hashlib releases the GIL on large
   buffers, so threads use real cores. Where a provider `.md5` exists, the
   MD5 must match it or the object is refused.
3. Ask the api to copy each object in with UploadPartCopy (no bytes through
   this machine), several objects at once, then finalise.

Hashes are cached per `bucket/key@etag`, so a re-run after an interruption
skips straight to the copy. Needs boto3 and read access to the source bucket
(on the EC2 box: the instance role).
"""

import base64
import hashlib
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .transfer import PART_SIZE

CACHE = Path(os.environ.get("INVENTORY_HASH_CACHE", Path.home() / ".cache" / "allotrope-inventory" / "s3-hashes.json"))
_cache_lock = threading.Lock()


def _s3():
    import boto3                      # import-s3 only; the rest of the CLI doesn't need it
    from botocore.config import Config

    return boto3.client("s3", region_name="ap-south-1", config=Config(max_pool_connections=128))


def list_objects(uri: str) -> tuple[str, str, list[dict], dict[str, str]]:
    """(bucket, prefix, payload objects, {object key: provider md5})."""
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    cli = _s3()
    objs = [
        o for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for o in page.get("Contents", []) if not o["Key"].endswith("/")
    ]
    keys = {o["Key"] for o in objs}
    md5 = {}
    for o in objs:
        if o["Key"].endswith(".md5") and o["Key"][:-4] in keys:
            line = cli.get_object(Bucket=bucket, Key=o["Key"])["Body"].read().decode().strip()
            md5[o["Key"][:-4]] = line.split()[0].lower()
    return bucket, prefix, objs, md5


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(key: str, value: dict) -> None:
    with _cache_lock:
        data = _load_cache()
        data[key] = value
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(CACHE)


def hash_object(bucket: str, obj: dict, in_flight: int = 6, on_bytes=lambda n: None) -> dict:
    """{size, sha256, part_sha256, md5} for one object, cached by ETag."""
    ck = f"{bucket}/{obj['Key']}@{obj['ETag']}"
    hit = _load_cache().get(ck)
    if hit:
        on_bytes(obj["Size"])
        return hit

    cli, size = _s3(), obj["Size"]
    n_parts = max(1, -(-size // PART_SIZE))
    whole, md5, parts = hashlib.sha256(), hashlib.md5(), [None] * n_parts
    # Parts are fetched out of order by a small pool but fed to the whole-file
    # digests strictly in order; `ready` holds parts that arrived early.
    ready: dict[int, bytes] = {}
    done = queue.Queue()

    def fetch(i: int) -> None:
        # Always put *something*: an exception left in the pool would leave
        # the ordered consumer below waiting forever for part i.
        try:
            lo = i * PART_SIZE
            rng = f"bytes={lo}-{min(lo + PART_SIZE, size) - 1}"
            body = cli.get_object(Bucket=bucket, Key=obj["Key"], Range=rng)["Body"].read() if size else b""
            done.put((i, body))
        except Exception as e:
            done.put((i, e))

    with ThreadPoolExecutor(max_workers=in_flight) as pool:
        nxt_submit = 0
        for _ in range(min(in_flight, n_parts)):
            pool.submit(fetch, nxt_submit); nxt_submit += 1
        for want in range(n_parts):
            while want not in ready:
                i, body = done.get()
                ready[i] = body
            body = ready.pop(want)
            if isinstance(body, Exception):
                raise RuntimeError(f"{obj['Key']} part {want + 1}: {body}") from body
            whole.update(body); md5.update(body)
            parts[want] = base64.b64encode(hashlib.sha256(body).digest()).decode()
            on_bytes(len(body))
            if nxt_submit < n_parts:
                pool.submit(fetch, nxt_submit); nxt_submit += 1

    result = {"size": size, "sha256": whole.hexdigest(), "part_sha256": parts, "md5": md5.hexdigest()}
    _save_cache(ck, result)
    return result


def hash_all(bucket: str, objs: list[dict], workers: int, on_bytes) -> dict[str, dict]:
    """Hash every object, `workers` objects at a time. Raises on the first failure."""
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(hash_object, bucket, o, 6, on_bytes): o["Key"] for o in objs}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out
