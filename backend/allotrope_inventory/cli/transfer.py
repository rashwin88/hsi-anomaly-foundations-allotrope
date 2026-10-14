"""Moving bytes: hash locally, PUT parts to presigned URLs, GET and verify.

PART_SIZE must equal the api's (allotrope_inventory/s3.py); `ingest` checks
it against the `part_size` every upload plan carries.
"""

import base64
import hashlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PART_SIZE = 64 * 1024 * 1024
READ_SIZE = 8 * 1024 * 1024           # divides PART_SIZE, so reads never straddle a part
RETRIES = 4


def hash_file(path: Path) -> tuple[int, str, list[str]]:
    """One read: (size, whole-file sha256 hex, base64 sha256 per PART_SIZE part)."""
    whole, part, parts, in_part, size = hashlib.sha256(), hashlib.sha256(), [], 0, 0
    with path.open("rb") as f:
        while chunk := f.read(READ_SIZE):
            whole.update(chunk)
            part.update(chunk)
            in_part += len(chunk)
            size += len(chunk)
            if in_part == PART_SIZE:
                parts.append(base64.b64encode(part.digest()).decode())
                part, in_part = hashlib.sha256(), 0
    if in_part or not parts:          # trailing partial part, or an empty file
        parts.append(base64.b64encode(part.digest()).decode())
    return size, whole.hexdigest(), parts


def _put_part(http: httpx.Client, path: Path, part: dict) -> int:
    with path.open("rb") as f:
        f.seek((part["part_number"] - 1) * PART_SIZE)
        body = f.read(PART_SIZE)
    for attempt in range(RETRIES):
        try:
            r = http.put(part["url"], content=body, headers=part["headers"])
            if r.status_code == 200:
                return len(body)
            # 4xx from S3 (bad digest, expired URL) won't fix itself on retry.
            if r.status_code < 500:
                raise RuntimeError(f"{path.name} part {part['part_number']}: {r.status_code} {r.text[:200]}")
        except httpx.TransportError:
            if attempt == RETRIES - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{path.name} part {part['part_number']}: gave up after {RETRIES} tries")


def upload_parts(
    jobs: list[tuple[Path, dict]], workers: int, on_bytes: Callable[[int], None]
) -> None:
    """PUT every (local file, part) in parallel; raises on the first hard failure."""
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=20.0)) as http, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_put_part, http, path, part) for path, part in jobs]
        for fut in as_completed(futures):
            on_bytes(fut.result())


def download_file(url: str, dest: Path, sha256: str) -> None:
    """Stream to `<dest>.part`, verify, then rename — a bad download never lands."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with httpx.stream("GET", url, timeout=httpx.Timeout(600.0, connect=20.0)) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(READ_SIZE):
                digest.update(chunk)
                f.write(chunk)
    if digest.hexdigest() != sha256:
        tmp.unlink()
        raise RuntimeError(f"{dest}: sha256 mismatch after download")
    tmp.replace(dest)
