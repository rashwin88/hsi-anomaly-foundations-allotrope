"""
Pack EnMAP L2A scenes from the inventory into training shards at native bands, with the quality
labels and scene facts the cloud/water segmentation head needs.

Source: the inventory's EnMAP scene datasets (`s3://allotrope-raw-inventory-india/enmap/L2A/ds_*/`,
kind=scene). Each file is checked against the inventory's sha256 before use. Output: the same shard
layout as SpectralEarth (`<out>/{train,test}/shard-NNNNNN.tar` + JSON sidecar, manifest.json,
stats.json), samples as written by app/utils/enmap/tiles.py. Runs on the prod box: inventory ->
EC2 -> training bucket in one region, no egress.

Four steps, each resumable (a unit is done when its sidecar exists):
    plan      read every scene's METADATA.XML only; split datatakes train/test (app/utils/enmap/split.py)
    tiles     per scene: download + verify, cut tiles into <work>/tiles/<split>/<scene>.tar (+ .json with
              per-wavelength stats), delete the scene                                   [--part K/N]
    shards    shuffle each split's tiles (seeded), write 150-tile shards to <out>           [--part K/N]
    finalize  manifest.json (shards, sha256, test datatakes) and stats.json keyed by wavelength

    python scripts/shard_enmap.py plan     s3://allotrope-raw-inventory-india/enmap/L2A/ /opt/enmap-work
    python scripts/shard_enmap.py tiles    s3://allotrope-raw-inventory-india/enmap/L2A/ /opt/enmap-work --part 0/2
    python scripts/shard_enmap.py shards   /opt/enmap-work s3://allotrope-training-data-prod-india/patches/enmap/native-v1/ --part 0/2
    python scripts/shard_enmap.py finalize /opt/enmap-work s3://allotrope-training-data-prod-india/patches/enmap/native-v1/

Gotcha: stats are keyed by wavelength (rounded to 0.01 nm), not band position - 5 of 210 scenes
number their bands differently, so position 141 is not the same wavelength everywhere.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.enmap.scene import EnmapScene  # noqa: E402
from app.utils.enmap.split import split_datatakes  # noqa: E402
from app.utils.enmap.tiles import NODATA, read_scene, tiles  # noqa: E402
from app.utils.spectralearth.sharding import PATCHES_PER_SHARD, BandStats, write_samples  # noqa: E402
from scripts.shard_spectralearth import Output  # noqa: E402

SPLIT_SEED, SHUFFLE_SEED, TEST_FRACTION = "en-v1", 0, 0.10
REFLECTANCE_SCALE = 0.0001


# --- inventory access --------------------------------------------------------------------------

def _s3():
    from app.utils.general_utils import s3_config
    return s3_config.client()


def list_scenes(inventory_uri: str, sensor: str = "enmap") -> list[dict]:
    """`sensor` scene datasets under the prefix: {id, scene_id, bucket, prefix}, sorted by scene id.
    Also used by scripts/shard_prisma.py."""
    bucket, _, prefix = inventory_uri.removeprefix("s3://").partition("/")
    s3, out = _s3(), []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            m = json.loads(s3.get_object(Bucket=bucket, Key=cp["Prefix"] + "manifest.json")["Body"].read())
            if m.get("kind") == "scene" and m.get("status") == "available" and m.get("sensor_type") == sensor:
                out.append({"id": m["id"], "scene_id": m["sensor_scene_id"], "bucket": bucket, "prefix": cp["Prefix"]})
    return sorted(out, key=lambda s: s["scene_id"])


def fetch_scene(scene: dict, dest: Path, only_suffix: str | None = None) -> Path:
    """Download the scene's payload into dest/<scene_id>/, each file verified against the
    inventory's sha256. Returns the folder. Raises ValueError on a mismatch."""
    s3, folder = _s3(), dest / scene["scene_id"]
    folder.mkdir(parents=True, exist_ok=True)
    rows = gzip.decompress(s3.get_object(Bucket=scene["bucket"], Key=scene["prefix"] + "files.jsonl.gz")["Body"].read())
    for line in rows.decode().splitlines():
        row = json.loads(line)
        name = row["relative_path"].rsplit("/", 1)[-1]
        if only_suffix and not name.lower().endswith(only_suffix.lower()):
            continue
        data = s3.get_object(Bucket=scene["bucket"], Key=scene["prefix"] + "payload/" + row["relative_path"])["Body"].read()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"{scene['scene_id']}/{name}: bytes do not match the inventory's sha256")
        (folder / name).write_bytes(data)
    return folder


# --- steps -------------------------------------------------------------------------------------

def plan(inventory_uri: str, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    scenes = list_scenes(inventory_uri)
    with tempfile.TemporaryDirectory() as tmp:
        for s in scenes:
            s["facts"] = EnmapScene(str(fetch_scene(s, Path(tmp), "METADATA.XML"))).facts
    test = split_datatakes([s["facts"] for s in scenes], TEST_FRACTION, SPLIT_SEED)
    for s in scenes:
        s["split"] = "test" if s["facts"]["datatake"] in test else "train"
    (work / "plan.json").write_text(json.dumps({"scenes": scenes, "test_datatakes": sorted(test)}, indent=1))
    n = {sp: sum(s["split"] == sp for s in scenes) for sp in ("train", "test")}
    print(f"{len(scenes)} scenes, {len(set(s['facts']['datatake'] for s in scenes))} datatakes; "
          f"test: {len(test)} datatakes, {n['test']} scenes; train: {n['train']} scenes")


def _wavelength_stats(stats: BandStats, wavelengths: np.ndarray) -> dict:
    return {f"{w:.2f}": [int(c), float(s), float(q)]
            for w, c, s, q in zip(wavelengths, stats.count, stats.total, stats.total_sq)}


def tiles_step(inventory_uri: str, work: Path, k: int, n: int) -> None:
    scenes = json.loads((work / "plan.json").read_text())["scenes"]
    mine = [s for i, s in enumerate(scenes) if i % n == k]
    print(f"part {k}/{n}: {len(mine)} of {len(scenes)} scenes", flush=True)
    for done, s in enumerate(mine, 1):
        out = work / "tiles" / s["split"] / f"{s['scene_id']}.tar"
        if out.with_suffix(".json").exists():
            continue
        t = time.time()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=work))
        try:
            scene = EnmapScene(str(fetch_scene(s, tmp)))
            with open(out, "wb") as f:
                # write_samples also accumulates per-band stats over the kept tiles' valid voxels
                keys, stats = write_samples(tiles(scene, read_scene(scene)), f, REFLECTANCE_SCALE, NODATA)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out.with_suffix(".json").write_text(json.dumps({
            "scene_id": s["scene_id"], "split": s["split"], "keys": keys,
            "stats": _wavelength_stats(stats, scene.wavelengths_nm) if keys else {}}))
        print(f"{s['scene_id'][:60]}: {len(keys)} tiles, {out.stat().st_size / 2**20:.0f} MiB, "
              f"{time.time() - t:.0f} s [{done}/{len(mine)}]", flush=True)


def shard_plan(work: Path) -> dict[str, list[list[tuple[str, str]]]]:
    """split -> shards, each a list of (scene tile tar, key); fixed by SHUFFLE_SEED."""
    plan_ = {}
    for split in ("train", "test"):
        items = []
        for side in sorted((work / "tiles" / split).glob("*.json")):
            meta = json.loads(side.read_text())
            items += [(str(side.with_suffix(".tar")), key) for key in meta["keys"]]
        random.Random(f"{SHUFFLE_SEED}:{split}").shuffle(items)
        plan_[split] = [items[i:i + PATCHES_PER_SHARD] for i in range(0, len(items), PATCHES_PER_SHARD)]
    return plan_


class _TileReader:
    """Random access to samples in the per-scene tile tars (index built once per tar)."""

    def __init__(self):
        self._tars: dict[str, tuple[tarfile.TarFile, dict]] = {}

    def sample(self, tar_path: str, key: str) -> dict[str, bytes]:
        if tar_path not in self._tars:
            tf = tarfile.open(tar_path)
            by_key = defaultdict(list)
            for m in tf.getmembers():
                by_key[m.name.split(".", 1)[0]].append(m)
            self._tars[tar_path] = (tf, by_key)
        tf, by_key = self._tars[tar_path]
        return {m.name.split(".", 1)[1]: tf.extractfile(m).read() for m in by_key[key]}


def shards_step(work: Path, out: Output, k: int, n: int) -> None:
    jobs = [(split, i, shard) for split, shards in shard_plan(work).items() for i, shard in enumerate(shards)]
    mine = [j for idx, j in enumerate(jobs) if idx % n == k]
    print(f"part {k}/{n}: {len(mine)} of {len(jobs)} shards", flush=True)
    reader = _TileReader()
    for done, (split, i, shard) in enumerate(mine, 1):
        name = f"{split}/shard-{i:06d}"
        if out.exists(f"{name}.json"):
            continue
        fd, tmp = tempfile.mkstemp(suffix=".tar", dir=work)
        try:
            with os.fdopen(fd, "wb") as f:
                keys, _ = write_samples(((key, reader.sample(tar, key)) for tar, key in shard), f)
            data = Path(tmp).read_bytes()
            out.put_file(f"{name}.tar", Path(tmp))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        out.put_bytes(f"{name}.json", json.dumps({"shard": f"{name}.tar", "patches": len(keys), "bytes": len(data),
                                                  "sha256": hashlib.sha256(data).hexdigest(), "keys": keys}).encode())
        print(f"{name}: {len(keys)} tiles, {len(data) / 2**20:.0f} MiB [{done}/{len(mine)}]", flush=True)


def finalize(work: Path, out: Output, source: str = "inventory EnMAP L2A scene datasets",
             bands: str = "native (all 224, each scene's own wavelengths)", split_record: dict | None = None) -> None:
    """manifest.json + stats.json. scripts/shard_prisma.py passes its own source, bands and split."""
    p = json.loads((work / "plan.json").read_text())
    shards = shard_plan(work)
    rows = {split: [json.loads(out.get_bytes(f"{split}/shard-{i:06d}.json")) for i in range(len(s))]
            for split, s in shards.items()}
    for split, r in rows.items():
        for i, (row, planned) in enumerate(zip(r, shards[split])):
            if row["keys"] != [key for _, key in planned]:
                raise SystemExit(f"{split}/shard-{i:06d}: keys do not match the plan")
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    for side in (work / "tiles" / "train").glob("*.json"):
        for w, (c, s_, q) in json.loads(side.read_text())["stats"].items():
            a = acc[w]; a[0] += c; a[1] += s_; a[2] += q
    wl = sorted(acc, key=float)
    mean = [acc[w][1] / acc[w][0] if acc[w][0] else None for w in wl]
    std = [float(np.sqrt(max(acc[w][2] / acc[w][0] - m * m, 0.0))) if m is not None else None for w, m in zip(wl, mean)]
    out.put_bytes("stats.json", json.dumps({"source": f"{source}: train split, kept tiles, valid voxels",
        "keyed_by": "wavelength_nm (rounded 0.01)", "wavelengths_nm": [float(w) for w in wl],
        "mean": mean, "std": std, "count": [acc[w][0] for w in wl], "units": "reflectance"}, indent=1).encode())
    out.put_bytes("manifest.json", json.dumps({
        "source": source, "bands": bands,
        "split": split_record or {"by": "datatake, stratified on cloud", "fraction": TEST_FRACTION, "seed": SPLIT_SEED,
                           "test_datatakes": p["test_datatakes"]},
        "shuffle_seed": SHUFFLE_SEED, "patches_per_shard": PATCHES_PER_SHARD,
        "scenes": [{k: s[k] for k in ("id", "scene_id", "split")} for s in p["scenes"]],
        "splits": {sp: {"shards": [{k: x[k] for k in ("shard", "patches", "bytes", "sha256")} for x in r],
                        "patches": sum(x["patches"] for x in r), "bytes": sum(x["bytes"] for x in r)}
                   for sp, r in rows.items()}}, indent=1).encode())
    for sp, r in rows.items():
        print(f"{sp}: {len(r)} shards, {sum(x['patches'] for x in r)} tiles, {sum(x['bytes'] for x in r) / 2**30:.1f} GiB")
    print(f"stats over {len(wl)} distinct wavelengths")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("step", choices=["plan", "tiles", "shards", "finalize"])
    ap.add_argument("a", help="plan/tiles: inventory s3:// prefix; shards/finalize: work dir")
    ap.add_argument("b", help="plan/tiles: work dir; shards/finalize: output s3:// prefix or folder")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="plan: only the first N scenes (smoke tests)")
    args = ap.parse_args()
    k, n = (int(x) for x in args.part.split("/"))
    if args.step == "plan":
        global list_scenes
        if args.limit:
            full = list_scenes
            list_scenes = lambda uri: full(uri)[: args.limit]  # noqa: E731
        plan(args.a, Path(args.b))
    elif args.step == "tiles":
        tiles_step(args.a, Path(args.b), k, n)
    elif args.step == "shards":
        shards_step(Path(args.a), Output(args.b), k, n)
    else:
        finalize(Path(args.a), Output(args.b))


if __name__ == "__main__":
    main()
