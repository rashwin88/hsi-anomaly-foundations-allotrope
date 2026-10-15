"""
Pack the AVIRIS-NG L2 flight lines in the inventory into training shards at native bands and native
pixel size - the airborne part of the fine-tuning data (docs/lld/aviris-shards.md).

Source: the inventory's aviris_ng scene datasets (`s3://allotrope-raw-inventory-india/aviris_ng/L2/ds_*/`),
one per flight line. Only the `_corr_*_img` reflectance pair is fetched (the water-vapour layer and
preview are not needed), each file verified against the inventory's sha256 while streaming to disk.
Output: the EnMAP/PRISMA shard layout with compressed pixels (app/utils/aviris/tiles.py). Runs on
the prod box: inventory -> EC2 -> training bucket, one region.

Same five steps as scripts/shard_prisma.py, each resumable:
    plan      list the aviris_ng scene datasets                               -> <work>/plan.json
    tiles     per line: fetch + verify the reflectance pair, tile strip by strip into
              <work>/tiles/all/<line>.tar (+ .json: keys, centre, stats), delete the download [--part K/N]
    split     scripts/shard_prisma.py's split: sites (15 km), benchmark sites held out, seed av-v1
    shards    as scripts/shard_enmap.py                                                        [--part K/N]
    finalize  as scripts/shard_enmap.py

    python scripts/shard_aviris.py plan     s3://allotrope-raw-inventory-india/aviris_ng/L2/ /opt/aviris-work
    python scripts/shard_aviris.py tiles    - /opt/aviris-work --part 0/2
    python scripts/shard_aviris.py split    - /opt/aviris-work
    python scripts/shard_aviris.py shards   /opt/aviris-work s3://allotrope-training-data-prod-india/patches/aviris_ng/native-v1/ --part 0/2
    python scripts/shard_aviris.py finalize /opt/aviris-work s3://allotrope-training-data-prod-india/patches/aviris_ng/native-v1/
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.aviris.scene import NODATA, REFLECTANCE_SCALE, open_scene  # noqa: E402
from app.utils.aviris.tiles import tiles  # noqa: E402
from app.utils.prisma.split import BENCH_KM, BENCH_SITES, SITE_KM  # noqa: E402
from app.utils.spectralearth.sharding import BandStats, write_samples  # noqa: E402
from scripts.shard_enmap import _s3, _wavelength_stats, finalize, list_scenes, shards_step  # noqa: E402
from scripts.shard_prisma import TEST_FRACTION, split_step  # noqa: E402
from scripts.shard_spectralearth import Output  # noqa: E402

SPLIT_SEED = "av-v1"
WANTED = re.compile(r"_corr_[a-z0-9]+_img\.(bin|hdr)$")


def fetch_reflectance(scene: dict, dest: Path) -> Path:
    """Stream the scene's `_corr_*_img.{bin,hdr}` to dest/<scene_id>/, hashing as it goes. Raises
    ValueError if a file's sha256 differs from the inventory's, or if the pair is incomplete."""
    s3, folder = _s3(), dest / scene["scene_id"]
    folder.mkdir(parents=True, exist_ok=True)
    rows = gzip.decompress(s3.get_object(Bucket=scene["bucket"], Key=scene["prefix"] + "files.jsonl.gz")["Body"].read())
    got = 0
    for row in map(json.loads, rows.decode().splitlines()):
        name = row["relative_path"].rsplit("/", 1)[-1]
        if not WANTED.search(name):
            continue
        body, h = s3.get_object(Bucket=scene["bucket"], Key=scene["prefix"] + "payload/" + row["relative_path"])["Body"], hashlib.sha256()
        with open(folder / name, "wb") as f:
            for chunk in iter(lambda: body.read(64 << 20), b""):
                h.update(chunk); f.write(chunk)
        if h.hexdigest() != row["sha256"]:
            raise ValueError(f"{scene['scene_id']}/{name}: bytes do not match the inventory's sha256")
        got += 1
    if got != 2:
        raise ValueError(f"{scene['scene_id']}: found {got} of the 2 _corr_ files")
    return folder


def plan(inventory_uri: str, work: Path, limit: int | None = None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    scenes = list_scenes(inventory_uri, sensor="aviris_ng")[:limit]
    (work / "plan.json").write_text(json.dumps({"scenes": scenes}, indent=1))
    print(f"{len(scenes)} AVIRIS-NG flight lines")


def tiles_step(work: Path, k: int, n: int) -> None:
    scenes = json.loads((work / "plan.json").read_text())["scenes"]
    mine = [s for i, s in enumerate(scenes) if i % n == k]
    print(f"part {k}/{n}: {len(mine)} of {len(scenes)} flight lines", flush=True)
    for done, s in enumerate(mine, 1):
        out = work / "tiles" / "all" / f"{s['scene_id']}.tar"
        if out.with_suffix(".json").exists():
            continue
        t = time.time()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=work))
        try:
            scene, stats = open_scene(fetch_reflectance(s, tmp)), BandStats()
            with open(out, "wb") as f:
                keys, _ = write_samples(tiles(scene, stats), f, REFLECTANCE_SCALE, NODATA)
            centre = scene.centre
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out.with_suffix(".json").write_text(json.dumps({
            "scene_id": s["scene_id"], "keys": keys, "centre": centre,
            "stats": _wavelength_stats(stats, scene.wavelengths_nm) if keys else {}}))
        print(f"{s['scene_id']}: {len(keys)} tiles, {out.stat().st_size / 2**20:.0f} MiB, "
              f"{time.time() - t:.0f} s [{done}/{len(mine)}]", flush=True)


SPLIT_RECORD = {"by": f"site (line centres within {SITE_KM:g} km), benchmark sites held out within {BENCH_KM:g} km",
                "fraction": TEST_FRACTION, "seed": SPLIT_SEED, "benchmark_sites": BENCH_SITES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("step", choices=["plan", "tiles", "split", "shards", "finalize"])
    ap.add_argument("a", help="plan: inventory s3:// prefix; tiles/split: -; shards/finalize: work dir")
    ap.add_argument("b", help="plan/tiles/split: work dir; shards/finalize: output s3:// prefix or folder")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="plan: only the first N flight lines (smoke tests)")
    args = ap.parse_args()
    k, n = (int(x) for x in args.part.split("/"))
    if args.step == "plan":
        plan(args.a, Path(args.b), args.limit)
    elif args.step == "tiles":
        tiles_step(Path(args.b), k, n)
    elif args.step == "split":
        split_step(Path(args.b), SPLIT_SEED)
    elif args.step == "shards":
        shards_step(Path(args.a), Output(args.b), k, n)
    else:
        finalize(Path(args.a), Output(args.b), source="inventory AVIRIS-NG L2 flight lines (India campaign 2015-16, 2018)",
                 bands="native (372: bbl = 1; centres within 0.42 nm across product versions); pixels.i16dz",
                 split_record=SPLIT_RECORD)


if __name__ == "__main__":
    main()
