"""
Pack the SpectralEarth collection into WebDataset shards for training.

Reads the inventory collection through app/utils/spectralearth (catalogue -> verified
fetch -> patch), writes `<out>/<split>/shard-NNNNNN.tar` with a JSON sidecar each, then
`--finalize` adds `<out>/manifest.json` and `<out>/stats.json` (per-band mean and std over
the train split - the wavelength-normalisation curve). Design:
docs/proposals/spectral-attention-bottleneck.md, "Data: SpectralEarth shards".

Resumable and parallel: the plan is fixed by the seed, a shard's sidecar is written only
after its tar is stored, and a shard with a sidecar is skipped. Run N processes with
`--part 0/N` ... `--part N-1/N`, then once with `--finalize`.

    python scripts/shard_spectralearth.py s3://allotrope-raw-inventory-india/enmap/SpectralEarth/<ds>/ \
        s3://allotrope-training-data-prod-india/patches/spectralearth/se-v1/ --part 0/4
    python scripts/shard_spectralearth.py <collection> <out> --finalize

`<out>` may be a local folder. `--limit N` shards only the first N patches of each split's
plan (smoke tests); use the same --limit for --finalize.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.dataset.spectralearth import load_band_table  # noqa: E402
from app.utils.general_utils import s3_config  # noqa: E402
from app.utils.spectralearth.catalog import SpectralEarthCatalog  # noqa: E402
from app.utils.spectralearth.sharding import (  # noqa: E402
    PATCHES_PER_SHARD, BandStats, shard_plan, write_shard)
from app.utils.spectralearth.source import SpectralEarthSource  # noqa: E402

# The se-v1 split, as fixed in docs/lld/spectralearth.md: 5% of sites, hashed with this seed.
SPLIT_FRACTION, SPLIT_SEED = 0.05, "se-v1"


class Output:
    """A local folder or an s3:// prefix, with just what the sharder needs."""

    def __init__(self, uri: str):
        self.s3 = uri.startswith("s3://")
        if self.s3:
            self.bucket, _, prefix = uri.removeprefix("s3://").partition("/")
            self.prefix, self.client = prefix.rstrip("/"), s3_config.client()
        else:
            self.root = Path(uri)

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def exists(self, rel: str) -> bool:
        if not self.s3:
            return (self.root / rel).exists()
        r = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self._key(rel), MaxKeys=1)
        return any(o["Key"] == self._key(rel) for o in r.get("Contents", []))

    def put_file(self, rel: str, path: Path) -> None:
        if self.s3:
            self.client.upload_file(str(path), self.bucket, self._key(rel))
        else:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, self.root / rel)

    def put_bytes(self, rel: str, data: bytes) -> None:
        if self.s3:
            self.client.put_object(Bucket=self.bucket, Key=self._key(rel), Body=data)
        else:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_bytes(data)

    def get_bytes(self, rel: str) -> bytes:
        if self.s3:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(rel))["Body"].read()
        return (self.root / rel).read_bytes()


def plans(collection: str, seed: int, limit: int | None) -> dict[str, list]:
    """Split name -> its shard plan."""
    catalog = SpectralEarthCatalog.from_files_manifest(collection)
    train, test = catalog.split(SPLIT_FRACTION, seed=SPLIT_SEED)
    out = {}
    for name, part in (("train", train), ("test", test)):
        entries = list(part)
        if limit is not None:
            entries = [e for s in shard_plan(entries, seed) for e in s][:limit]
        out[name] = shard_plan(entries, seed, PATCHES_PER_SHARD)
    return out


def run_part(out: Output, plan: dict[str, list], k: int, n: int, workers: int) -> None:
    source = SpectralEarthSource(workers=workers)
    jobs = [(split, i, shard) for split, shards in plan.items() for i, shard in enumerate(shards)]
    mine = [j for idx, j in enumerate(jobs) if idx % n == k]
    print(f"part {k}/{n}: {len(mine)} of {len(jobs)} shards", flush=True)
    for done, (split, i, shard) in enumerate(mine, 1):
        name = f"{split}/shard-{i:06d}"
        if out.exists(f"{name}.json"):
            continue
        t = time.time()
        fd, tmp = tempfile.mkstemp(suffix=".tar")
        try:
            with os.fdopen(fd, "wb") as f:
                keys, stats = write_shard(source.load_many(shard), f)
            data_sha = hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
            size = os.path.getsize(tmp)
            out.put_file(f"{name}.tar", Path(tmp))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        # The sidecar is the "done" marker, so it is written only after the tar is stored.
        out.put_bytes(f"{name}.json", json.dumps({
            "shard": f"{name}.tar", "patches": len(keys), "bytes": size, "sha256": data_sha,
            "keys": keys, "band_count": stats.count.tolist(), "band_sum": stats.total.tolist(),
            "band_sum_sq": stats.total_sq.tolist()}).encode())
        dt = time.time() - t
        print(f"{name}: {len(keys)} patches, {size / 2**20:.0f} MiB, {dt:.1f} s "
              f"({size / 2**20 / dt:.0f} MiB/s) [{done}/{len(mine)}]", flush=True)


def finalize(out: Output, plan: dict[str, list], collection: str, seed: int) -> None:
    """Check every planned shard is present, then write manifest.json and stats.json."""
    manifest = {"collection": collection, "split": {"fraction": SPLIT_FRACTION, "seed": SPLIT_SEED},
                "shuffle_seed": seed, "patches_per_shard": PATCHES_PER_SHARD, "splits": {}}
    train_stats = BandStats()
    for split, shards in plan.items():
        rows = []
        for i, shard in enumerate(shards):
            side = json.loads(out.get_bytes(f"{split}/shard-{i:06d}.json"))
            if side["keys"] != [f"se_{e.site_id}_{e.patch_id}" for e in shard]:
                raise SystemExit(f"{split}/shard-{i:06d}: keys don't match the plan")
            rows.append({k: side[k] for k in ("shard", "patches", "bytes", "sha256")})
            if split == "train":
                train_stats = train_stats.merge(BandStats(
                    np.array(side["band_count"], np.int64), np.array(side["band_sum"]),
                    np.array(side["band_sum_sq"])))
        manifest["splits"][split] = {"shards": rows, "patches": sum(r["patches"] for r in rows),
                                     "bytes": sum(r["bytes"] for r in rows)}
    bands = load_band_table()
    mean, std = train_stats.mean_std()
    stats = {"source": "train split", "band_order": "SpectralEarth band order (not sorted)",
             "wavelengths_nm": bands.wavelengths_nm.tolist(), "mean": mean.tolist(),
             "std": std.tolist(), "count": train_stats.count.tolist(), "units": "reflectance"}
    out.put_bytes("stats.json", json.dumps(stats, indent=1).encode())
    out.put_bytes("manifest.json", json.dumps(manifest, indent=1).encode())
    for split, s in manifest["splits"].items():
        print(f"{split}: {len(s['shards'])} shards, {s['patches']} patches, {s['bytes'] / 2**40:.3f} TiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("collection", help="s3:// prefix (or local folder) holding files.jsonl.gz")
    ap.add_argument("out", help="s3:// prefix or local folder for the shards")
    ap.add_argument("--part", default="0/1", help="K/N: write every N-th shard starting at K")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for the shard plan")
    ap.add_argument("--limit", type=int, default=None, help="patches per split (smoke tests)")
    ap.add_argument("--workers", type=int, default=32, help="parallel patch fetches")
    args = ap.parse_args()
    out, plan = Output(args.out), plans(args.collection, args.seed, args.limit)
    if args.finalize:
        finalize(out, plan, args.collection, args.seed)
    else:
        k, n = (int(x) for x in args.part.split("/"))
        run_part(out, plan, k, n, args.workers)


if __name__ == "__main__":
    main()
