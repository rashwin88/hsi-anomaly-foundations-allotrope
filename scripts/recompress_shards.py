"""
Re-encode SpectralEarth shards with band-delta + zstd (`pixels.npy` -> `pixels.i16dz`) for shipping.

Reads shards written by scripts/shard_spectralearth.py, writes the same layout under a new prefix
plus its own manifest.json and a copy of stats.json. Runs on the prod box: S3 -> EC2 -> S3 in one
region moves no billable bytes, so compression happens before anything leaves AWS (S3 egress is
$0.109/GB; the airgapped H200 plan, research/experiments/indradhanu-se-pretrain/H200_PLAN.md).

Integrity, end to end: each source shard's sha256 is checked against the source manifest before it
is re-encoded, and the output's sha256 goes into the new manifest - so the copy that ships is
verifiably the data that was sharded.

Resumable and parallel, like the sharder: a shard is done when its sidecar exists (written after
the tar); `--part K/N` splits the work; `--finalize` writes the manifest once all are present.

    python scripts/recompress_shards.py s3://.../se-v1/ s3://.../se-v1z/ --train 1000 --test 20 --part 0/4
    python scripts/recompress_shards.py s3://.../se-v1/ s3://.../se-v1z/ --train 1000 --test 20 --finalize
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.spectralearth.sharding import recompress_shard  # noqa: E402
from scripts.shard_spectralearth import Output  # noqa: E402


def jobs(train: int, test: int) -> list[tuple[str, int]]:
    """(split, shard index) for the first `train` train shards and `test` test shards."""
    return [("train", i) for i in range(train)] + [("test", i) for i in range(test)]


def run_part(src: Output, dst: Output, todo: list[tuple[str, int]], k: int, n: int) -> None:
    manifest = json.loads(src.get_bytes("manifest.json"))
    expected = {row["shard"]: row["sha256"] for split in manifest["splits"].values() for row in split["shards"]}
    mine = [j for idx, j in enumerate(todo) if idx % n == k]
    print(f"part {k}/{n}: {len(mine)} of {len(todo)} shards", flush=True)
    for done, (split, i) in enumerate(mine, 1):
        name = f"{split}/shard-{i:06d}"
        if dst.exists(f"{name}.json"):
            continue
        t = time.time()
        raw = src.get_bytes(f"{name}.tar")
        if hashlib.sha256(raw).hexdigest() != expected[f"{name}.tar"]:
            raise SystemExit(f"{name}.tar: source bytes do not match the source manifest")
        fd, tmp = tempfile.mkstemp(suffix=".tar")
        try:
            with os.fdopen(fd, "wb") as out:
                keys = recompress_shard(_BytesReader(raw), out)
            data = Path(tmp).read_bytes()
            dst.put_file(f"{name}.tar", Path(tmp))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        dst.put_bytes(f"{name}.json", json.dumps({
            "shard": f"{name}.tar", "patches": len(keys), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "source_sha256": expected[f"{name}.tar"],
            "source_bytes": len(raw), "keys": keys}).encode())
        dt = time.time() - t
        print(f"{name}: {len(raw) / 2**20:.0f} -> {len(data) / 2**20:.0f} MiB ({len(raw) / len(data):.2f}x), "
              f"{dt:.1f} s [{done}/{len(mine)}]", flush=True)


class _BytesReader:
    """A minimal file object over bytes for tarfile's streaming mode."""

    def __init__(self, data: bytes):
        self._data, self._pos = data, 0

    def read(self, size: int = -1) -> bytes:
        end = len(self._data) if size < 0 else self._pos + size
        out = self._data[self._pos:end]
        self._pos += len(out)
        return out


def finalize(src: Output, dst: Output, todo: list[tuple[str, int]]) -> None:
    rows = {"train": [], "test": []}
    for split, i in todo:
        side = json.loads(dst.get_bytes(f"{split}/shard-{i:06d}.json"))
        rows[split].append({k: side[k] for k in ("shard", "patches", "bytes", "sha256", "source_sha256")})
    source = json.loads(src.get_bytes("manifest.json"))
    manifest = {"encoding": "pixels.i16dz: band-delta int16 + zstd-3 (app/utils/shard_codec.py)",
                "source_manifest": {k: source[k] for k in ("collection", "split", "shuffle_seed", "patches_per_shard")},
                "splits": {s: {"shards": r, "patches": sum(x["patches"] for x in r),
                               "bytes": sum(x["bytes"] for x in r)} for s, r in rows.items() if r}}
    dst.put_bytes("stats.json", src.get_bytes("stats.json"))
    dst.put_bytes("manifest.json", json.dumps(manifest, indent=1).encode())
    for s, m in manifest["splits"].items():
        print(f"{s}: {len(m['shards'])} shards, {m['patches']} patches, {m['bytes'] / 2**30:.1f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("src", help="s3:// prefix (or folder) of the raw shard set")
    ap.add_argument("dst", help="s3:// prefix (or folder) for the re-encoded set")
    ap.add_argument("--train", type=int, required=True, help="first N train shards")
    ap.add_argument("--test", type=int, required=True, help="first N test shards")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    src, dst, todo = Output(args.src), Output(args.dst), jobs(args.train, args.test)
    if args.finalize:
        finalize(src, dst, todo)
    else:
        k, n = (int(x) for x in args.part.split("/"))
        run_part(src, dst, todo, k, n)


if __name__ == "__main__":
    main()
