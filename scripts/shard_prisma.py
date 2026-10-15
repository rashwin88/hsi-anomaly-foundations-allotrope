"""
Pack PRISMA L2D scenes from the inventory into training shards at native bands - the PRISMA half of
the fine-tuning data (docs/lld/prisma-shards.md).

Source: the inventory's PRISMA scene datasets (`s3://allotrope-raw-inventory-india/prisma/L2D/ds_*/`),
each file checked against the inventory's sha256. Output: the EnMAP shard layout
(`<out>/{train,test}/shard-NNNNNN.tar` + sidecars, manifest.json, stats.json), samples as written by
app/utils/prisma/tiles.py. Runs on the prod box: inventory -> EC2 -> training bucket, one region.

Five steps, each resumable (a unit is done when its sidecar exists). The order differs from EnMAP's
because the inventory has no PRISMA footprints: a scene's place is known only once it is read, so
the split comes after tiling.
    plan      list the scene datasets                                         -> <work>/plan.json
    tiles     per scene: download + verify, read, cut tiles into <work>/tiles/all/<scene>.tar
              (+ .json: keys, scene centre, per-wavelength stats), delete the scene     [--part K/N]
    split     group scenes into sites, hold out benchmark sites, move each scene's tiles into
              <work>/tiles/{train,test,excluded}/ (app/utils/prisma/split.py)
    shards    as scripts/shard_enmap.py: shuffle each split's tiles, write 150-tile shards [--part K/N]
    finalize  as scripts/shard_enmap.py: manifest.json and stats.json keyed by wavelength

    python scripts/shard_prisma.py plan     s3://allotrope-raw-inventory-india/prisma/L2D/ /opt/prisma-work
    python scripts/shard_prisma.py tiles    s3://allotrope-raw-inventory-india/prisma/L2D/ /opt/prisma-work --part 0/2
    python scripts/shard_prisma.py split    - /opt/prisma-work
    python scripts/shard_prisma.py shards   /opt/prisma-work s3://allotrope-training-data-prod-india/patches/prisma/native-v1/ --part 0/2
    python scripts/shard_prisma.py finalize /opt/prisma-work s3://allotrope-training-data-prod-india/patches/prisma/native-v1/
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.prisma.scene import NODATA, REFLECTANCE_SCALE, read_scene  # noqa: E402
from app.utils.prisma.split import BENCH_KM, BENCH_SITES, SITE_KM, split_scenes  # noqa: E402
from app.utils.prisma.tiles import tiles  # noqa: E402
from app.utils.spectralearth.sharding import write_samples  # noqa: E402
from scripts.shard_enmap import _wavelength_stats, fetch_scene, finalize, list_scenes, shards_step  # noqa: E402
from scripts.shard_spectralearth import Output  # noqa: E402

SPLIT_SEED, TEST_FRACTION = "pr-v1", 0.10


def plan(inventory_uri: str, work: Path, limit: int | None = None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    scenes = list_scenes(inventory_uri, sensor="prisma")[:limit]
    (work / "plan.json").write_text(json.dumps({"scenes": scenes}, indent=1))
    print(f"{len(scenes)} PRISMA scenes")


def tiles_step(work: Path, k: int, n: int) -> None:
    scenes = json.loads((work / "plan.json").read_text())["scenes"]
    mine = [s for i, s in enumerate(scenes) if i % n == k]
    print(f"part {k}/{n}: {len(mine)} of {len(scenes)} scenes", flush=True)
    for done, s in enumerate(mine, 1):
        out = work / "tiles" / "all" / f"{s['scene_id']}.tar"
        if out.with_suffix(".json").exists():
            continue
        t = time.time()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=work))
        try:
            scene = read_scene(next(fetch_scene(s, tmp).glob("*.he5")))
            with open(out, "wb") as f:
                keys, stats = write_samples(tiles(scene), f, REFLECTANCE_SCALE, NODATA)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out.with_suffix(".json").write_text(json.dumps({
            "scene_id": s["scene_id"], "keys": keys, "centre": scene.centre,
            "stats": _wavelength_stats(stats, scene.wavelengths_nm) if keys else {}}))
        print(f"{s['scene_id']}: {len(keys)} tiles, {out.stat().st_size / 2**20:.0f} MiB, "
              f"{time.time() - t:.0f} s [{done}/{len(mine)}]", flush=True)


def split_step(work: Path, seed: str = SPLIT_SEED) -> None:
    """Also used by scripts/shard_aviris.py, with its own seed."""
    p = json.loads((work / "plan.json").read_text())
    sides = {json.loads(f.read_text())["scene_id"]: f for f in (work / "tiles" / "all").glob("*.json")}
    missing = [s["scene_id"] for s in p["scenes"] if s["scene_id"] not in sides]
    if missing:
        raise SystemExit(f"{len(missing)} scenes not tiled yet, e.g. {missing[:3]}")
    centres = {sid: tuple(json.loads(f.read_text())["centre"]) for sid, f in sides.items()}
    assign = split_scenes(centres, TEST_FRACTION, seed)
    for sid, side in sides.items():
        dest = work / "tiles" / assign[sid]
        dest.mkdir(parents=True, exist_ok=True)
        side.with_suffix(".tar").rename(dest / side.with_suffix(".tar").name)
        side.rename(dest / side.name)                                   # sidecar last: marks the move done
    for s in p["scenes"]:
        s["split"], s["centre"] = assign[s["scene_id"]], centres[s["scene_id"]]
    (work / "plan.json").write_text(json.dumps(p, indent=1))
    n = {sp: sum(v == sp for v in assign.values()) for sp in ("train", "test", "excluded")}
    print(f"scenes: {n}")


SPLIT_RECORD = {"by": f"site (scene centres within {SITE_KM:g} km), benchmark sites held out within {BENCH_KM:g} km",
                "fraction": TEST_FRACTION, "seed": SPLIT_SEED, "benchmark_sites": BENCH_SITES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("step", choices=["plan", "tiles", "split", "shards", "finalize"])
    ap.add_argument("a", help="plan/tiles: inventory s3:// prefix; split: -; shards/finalize: work dir")
    ap.add_argument("b", help="plan/tiles/split: work dir; shards/finalize: output s3:// prefix or folder")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="plan: only the first N scenes (smoke tests)")
    args = ap.parse_args()
    k, n = (int(x) for x in args.part.split("/"))
    if args.step == "plan":
        plan(args.a, Path(args.b), args.limit)
    elif args.step == "tiles":
        tiles_step(Path(args.b), k, n)
    elif args.step == "split":
        split_step(Path(args.b))
    elif args.step == "shards":
        shards_step(Path(args.a), Output(args.b), k, n)
    else:
        finalize(Path(args.a), Output(args.b), source="inventory PRISMA L2D scene datasets",
                 bands="native (234: every band PRISMA flags valid; centres fixed across scenes)", split_record=SPLIT_RECORD)


if __name__ == "__main__":
    main()
