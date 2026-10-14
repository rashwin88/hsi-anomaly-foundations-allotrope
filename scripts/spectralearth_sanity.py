"""
Sample SpectralEarth patches end to end and report whether the data layer behaves.

Run by hand (on the EC2 box, in the worker image, next to the data) after changes to
app/utils/spectralearth/ or before a training run. It exercises catalogue -> source ->
screens -> torch Dataset/DataLoader on real patches and writes one JSON file of numbers
and RGB previews; nothing it does is cached or written back.

What it answers: are the bands labelled right (RGB previews look like real places, the
mean spectrum has the shape of land surface), are values in reflectance range, how much is
nodata or rejected, how are patches spread over sites, and how fast does a DataLoader
deliver native-mode samples.

Usage:
    python scripts/spectralearth_sanity.py s3://<bucket>/<collection prefix>/ --n 500 --out report.json
"""

import argparse
import base64
import collections
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.spectralearth.catalog import SpectralEarthCatalog  # noqa: E402
from app.utils.spectralearth.screens import NodataScreen, ScreenChain  # noqa: E402
from app.utils.spectralearth.source import SpectralEarthSource  # noqa: E402
from app.utils.spectralearth.torch_dataset import SpectralEarthTorchDataset  # noqa: E402

RGB_BANDS = (43, 28, 10)   # 641 / 556 / 468 nm, the indices the SpectralEarth authors use


def _rgb_png(reflectance: np.ndarray) -> str:
    from PIL import Image

    rgb = np.stack([reflectance[b] for b in RGB_BANDS], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    img = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** (1 / 1.6)
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("collection_uri")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--previews", type=int, default=12)
    ap.add_argument("--out", default="spectralearth_sanity.json")
    args = ap.parse_args()
    report: dict = {}

    t = time.time()
    cat = SpectralEarthCatalog.from_files_manifest(args.collection_uri)
    sites = cat.by_site()
    train, test = cat.split(0.05, seed="se-v1")
    report["catalogue"] = {
        "patches": len(cat), "sites": len(sites), "tib": round(cat.total_bytes / 1024**4, 3),
        "load_seconds": round(time.time() - t, 1),
        "patches_per_site": dict(sorted(collections.Counter(len(v) for v in sites.values()).items())),
        "split_se_v1": {"train": len(train), "test": len(test)},
    }

    sample = cat.sample(args.n, seed=args.seed)
    source = SpectralEarthSource(workers=32)
    t = time.time()
    patches = list(source.load_many(sample))
    load_s = time.time() - t
    chain = ScreenChain([NodataScreen()])
    nodata, keep, usable, rejected = [], [], 0, collections.Counter()
    pixels = []
    for p in patches:
        v = p.validity()
        nodata.append(1 - v.mean())
        r = chain.apply(p)
        keep.append(r.keep_fraction); usable += r.usable(); rejected.update(r.rejected_by)
        refl = p.reflectance()
        pixels.append(refl[:, ::8, ::8].reshape(refl.shape[0], -1))   # 256 pixels per patch
    px = np.concatenate(pixels, axis=1)
    report["sample"] = {
        "patches": len(patches), "load_seconds": round(load_s, 1),
        "patches_per_second": round(len(patches) / load_s, 1),
        "nodata_voxel_fraction": {"mean": float(np.mean(nodata)), "max": float(np.max(nodata)),
                                  "patches_with_any": int(sum(x > 0 for x in nodata))},
        "keep_fraction": {"min": float(np.min(keep)), "median": float(np.median(keep))},
        "usable": usable, "rejected_pixels_by_screen": dict(rejected),
        "reflectance_percentiles": {str(q): float(np.percentile(px, q)) for q in (0.1, 1, 50, 99, 99.9)},
    }
    w = patches[0].wavelengths_nm
    order = np.argsort(w)
    report["spectrum"] = {
        "wavelength_nm": w[order].round(2).tolist(),
        "family": [patches[0].families[i].value for i in order],
        "p10": np.percentile(px, 10, axis=1)[order].round(4).tolist(),
        "median": np.percentile(px, 50, axis=1)[order].round(4).tolist(),
        "p90": np.percentile(px, 90, axis=1)[order].round(4).tolist(),
    }
    report["previews"] = [{"key": p.key, "png": _rgb_png(p.reflectance())} for p in patches[: args.previews]]

    import torch
    from torch.utils.data import DataLoader

    ds = SpectralEarthTorchDataset(sample, SpectralEarthSource(workers=4), mode="native")
    dl = DataLoader(ds, batch_size=32, num_workers=8, shuffle=True)
    t, seen = time.time(), 0
    for batch in dl:
        seen += batch["pixels"].shape[0]
        if seen >= min(256, len(ds)):
            break
    report["dataloader_native"] = {"samples": seen, "samples_per_second": round(seen / (time.time() - t), 1),
                                   "batch_shape": list(batch["pixels"].shape), "torch": torch.__version__}

    Path(args.out).write_text(json.dumps(report), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("spectrum", "previews")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
