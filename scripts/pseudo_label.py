"""
Teacher labels for sensors without masks (FT04, docs/proposals/fine-tuning-and-segmentation.md, stage 4): run a
checkpoint's segmentation head over every tile of a shard folder and save its class probabilities.

    python scripts/pseudo_label.py <checkpoint> <shard dir> <out dir> [--band-list <json>] [--batch 32]

Each tile is read as the fine-tuning loader reads it (decoded, aligned onto the band list) and written as
<out>/<tile key>.npy, (5, 128, 128) uint8 = round(255 x sigmoid) for water, cloud, thin_cloud, shadow, snow. The
loader turns these into guarded targets (app/utils/segmentation/targets.pseudo_targets); thresholds and guards live
there, so the teacher's raw output is kept. Tiles already written are skipped, so a killed run resumes.
"""

import argparse
import io
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def tiles(shard: Path, target_nm):
    """(key, sample) for every tile of one shard, decoded and band-aligned like the loader."""
    from app.foundation_models.trainers.spectral_stem_mae_trainer import NODATA, align_sample, decode_i16dz_member
    with tarfile.open(shard) as t:
        groups: dict = {}
        for m in t.getmembers():
            key, _, ext = m.name.partition(".")
            if ext in ("pixels.i16dz", "pixels.npy", "wavelengths.npy"):
                raw = t.extractfile(m).read()
                groups.setdefault(key, {})[ext] = decode_i16dz_member(ext, raw) if ext.endswith("i16dz") else np.load(io.BytesIO(raw))
        for key, s in groups.items():
            if target_nm is not None:
                s = align_sample(s, target_nm)
            yield key, s, NODATA


def main() -> None:
    import torch
    from app.evaluation.config import ModelSpec
    from app.evaluation.models import EvalModel
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("checkpoint")
    p.add_argument("shards")
    p.add_argument("out")
    p.add_argument("--band-list")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = None
    if a.band_list:
        from app.utils.band_alignment import load_band_list
        target = load_band_list(a.band_list)
    model = EvalModel(ModelSpec(name="teacher", checkpoint=a.checkpoint), out / ".cache")
    net = model.inferencer.model.inner.eval()
    dev = next(net.parameters()).device
    done = written = 0
    t0 = time.time()
    for shard in sorted(Path(a.shards).glob("*.tar")):
        batch = []
        for key, s, nodata in tiles(shard, target):
            if (out / f"{key}.npy").exists():
                done += 1
                continue
            batch.append((key, s, nodata))
            if len(batch) == a.batch:
                written += write(net, dev, batch, out)
                batch = []
        if batch:
            written += write(net, dev, batch, out)
        print(f"{shard.name}: {written} written, {done} already done, {time.time() - t0:.0f} s", flush=True)


def write(net, dev, batch, out: Path) -> int:
    """Segment one batch; tiles with different band lists go through in separate groups."""
    groups: dict = {}
    for item in batch:
        groups.setdefault(tuple(np.round(np.asarray(item[1]["wavelengths.npy"], np.float64), 2)), []).append(item)
    return sum(_write_group(net, dev, g, out) for g in groups.values())


def _write_group(net, dev, batch, out: Path) -> int:
    import torch
    pix = np.stack([np.asarray(s["pixels.npy"] if "pixels.npy" in s else s["pixels.i16dz"]) for _, s, _ in batch])
    nodata = batch[0][2]
    wl = torch.as_tensor(np.asarray(batch[0][1]["wavelengths.npy"], np.float32), device=dev)
    bv = torch.from_numpy(pix != nodata).to(dev)
    x = torch.from_numpy(np.where(pix == nodata, 0, pix).astype(np.float32) * 1e-4).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        p = torch.sigmoid(net.segment(x, wl, bv).float()).cpu().numpy()
    for (key, _, _), pp in zip(batch, p):
        np.save(out / f"{key}.npy", np.round(pp * 255).astype(np.uint8))
    return len(batch)


if __name__ == "__main__":
    main()
