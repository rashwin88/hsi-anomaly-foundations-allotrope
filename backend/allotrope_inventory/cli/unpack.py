"""`unpack-archive`: stream an archive dataset into a collection, never touching disk.

    S3 parts, in order ─▶ pigz -dc ─▶ tar stream ─▶ per member: SHA-256,
    single PUT with ChecksumSHA256 (S3 verifies the whole file) ─▶ rows to
    POST /files/bulk in batches (the api re-checks each object by HEAD).

gzip can only be decompressed serially, so that one stream sets the pace.
Everything else is kept off its path: uploads run on a pool (capped at
MAX_IN_FLIGHT members, so a slow S3 can't fill memory), and registration
runs on its own thread, so the HEAD checks behind each batch never stall
decompression. Re-running resumes: the stream is decompressed from the
start again — a .tar.gz cannot be seeked — but members already verified in
the collection are neither uploaded nor registered twice.

Needs boto3, read on the archive and write on the collection prefix (on the
EC2 box: the instance role). Uses `pigz` when it is on PATH, else Python's
gzip reader.
"""

import base64
import fnmatch
import hashlib
import os
import queue
import re
import shutil
import subprocess
import tarfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

BATCH = 2000                          # rows per POST /files/bulk
MAX_IN_FLIGHT = 256                   # members read but not yet uploaded (~1.7 GB of patches)


def _natural(key: str) -> list:
    """part_2 before part_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]


def _s3():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", region_name="ap-south-1", config=Config(max_pool_connections=300))


def run(c, archive: dict, coll: dict, parts_glob: str, put_workers: int, echo=print) -> None:
    cli = _s3()
    a_bucket, _, a_prefix = archive["s3_uri"].removeprefix("s3://").partition("/")
    c_bucket, _, c_prefix = coll["s3_uri"].removeprefix("s3://").partition("/")

    parts = sorted(
        (f["relative_path"] for f in _all_files(c, archive["id"])
         if fnmatch.fnmatch(f["relative_path"].rsplit("/", 1)[-1], parts_glob)),
        key=_natural,
    )
    if not parts:
        raise SystemExit(f"no archive files match {parts_glob}")
    echo(f"stream: {len(parts)} parts in order: {parts[0]} … {parts[-1]}")
    done = set(c.verified_files(coll["id"]))
    echo(f"{len(done):,} members already in the collection")

    # The parts concatenated are one gzip stream. pigz decompresses it when
    # present (the box); otherwise tarfile's own gzip reader does, slower.
    if shutil.which("pigz"):
        pigz = subprocess.Popen(["pigz", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        sink, stream, mode = pigz.stdin, pigz.stdout, "r|"
    else:
        pigz = None
        rfd, wfd = os.pipe()
        sink, stream, mode = os.fdopen(wfd, "wb"), os.fdopen(rfd, "rb"), "r|gz"
    feed_error: list[BaseException] = []

    def feed() -> None:
        try:
            for rel in parts:
                body = cli.get_object(Bucket=a_bucket, Key=f"{a_prefix}payload/{rel}")["Body"]
                for chunk in body.iter_chunks(16 * 1024 * 1024):
                    sink.write(chunk)
        except BaseException as e:    # surfaced by the main loop
            feed_error.append(e)
        finally:
            sink.close()

    threading.Thread(target=feed, daemon=True).start()

    def put(path: str, data: bytes, digest: bytes) -> None:
        cli.put_object(
            Bucket=c_bucket, Key=f"{c_prefix}payload/{path}", Body=data,
            ChecksumSHA256=base64.b64encode(digest).decode(),
        )

    slots = threading.Semaphore(MAX_IN_FLIGHT)
    batches: queue.Queue = queue.Queue(maxsize=2)   # backpressure if registration falls behind
    reg_error: list[BaseException] = []

    def register() -> None:
        # Waits for a batch's uploads, then registers it. On failure keeps
        # draining (so the reader can't block on a full queue) but stops posting.
        while (batch := batches.get()) is not None:
            if reg_error:
                continue
            try:
                for f, _ in batch:
                    f.result()        # re-raise any upload failure
                r = c.post(f"/datasets/{coll['id']}/files/bulk",
                           {"files": [row for _, row in batch]}, timeout=None)
                if r["failed"]:
                    raise RuntimeError(f"bulk registration refused {len(r['failed'])} files, e.g. {r['failed'][:3]}")
            except BaseException as e:
                reg_error.append(e)

    registrar = threading.Thread(target=register, daemon=True)
    registrar.start()

    def upload(path: str, data: bytes, digest: bytes) -> None:
        try:
            put(path, data, digest)
        finally:
            slots.release()

    pending: list[tuple[Future, dict]] = []
    stats = {"seen": 0, "new": 0, "bytes": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=put_workers) as pool, \
            tarfile.open(fileobj=stream, mode=mode) as tar:
        for m in tar:
            if reg_error:
                break
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            stats["seen"] += 1
            stats["bytes"] += len(data)
            if m.name not in done:
                digest = hashlib.sha256(data).digest()
                slots.acquire()
                pending.append((pool.submit(upload, m.name, data, digest),
                                {"relative_path": m.name, "size_bytes": len(data), "sha256": digest.hex()}))
                stats["new"] += 1
                if len(pending) >= BATCH:
                    batches.put(pending)
                    pending = []
            if stats["seen"] % 5000 == 0:
                rate = stats["bytes"] / max(time.time() - t0, 1) / 1048576
                echo(f"  {stats['seen']:,} members ({stats['new']:,} new) · "
                     f"{stats['bytes'] / 1024**4:.2f} TiB · {rate:.0f} MiB/s")
        if pending:
            batches.put(pending)
        batches.put(None)
        registrar.join()

    if reg_error:
        raise SystemExit(f"registration failed: {reg_error[0]}")
    if feed_error:
        raise SystemExit(f"reading the archive failed: {feed_error[0]}")
    if pigz is not None and pigz.wait() != 0:
        raise SystemExit(f"pigz exited {pigz.returncode}: the archive stream is corrupt or truncated")
    echo(f"unpacked {stats['seen']:,} members ({stats['new']:,} new, "
         f"{stats['bytes'] / 1024**4:.2f} TiB) in {(time.time() - t0) / 3600:.1f} h")


def _all_files(c, dataset_id: str) -> list[dict]:
    out, offset = [], 0
    while True:
        page = c.get(f"/datasets/{dataset_id}/files", limit=1000, offset=offset)
        out += page["items"]
        offset += len(page["items"])
        if not page["items"] or offset >= page["total"]:
            return out
