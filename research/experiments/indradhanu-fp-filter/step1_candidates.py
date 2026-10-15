"""
Step 1 of the Indradhanu false-positive filter experiment: score the four
annotated PRISMA scenes with Indradhanu and measure what a loose candidate
cut catches.

It mirrors the product path (band_filter_apply → anomaly_scoring) using the
same app/ code and the same PRISMA defaults, so the score maps here are what
the app would produce. It exists outside the product because the four scenes
are not onboarded and the worker has no GPU.

Outputs go to OUT_ROOT, outside the repo: per scene, the L1 / SAM / combined
score maps and a keep mask as .npy, plus one summary.json across scenes. The
165-band cube and the reconstruction are not saved (about 1 GB each).

Gotchas:

  - Scores are computed over the spatially valid pixels only; the GT is
    measured over the same pixels, so anomalies sitting on nodata are counted
    as "not scoreable" rather than silently missed.
  - On Windows run with `python -X utf8`: the model manifest carries the
    Devanagari codename and the resolver reads it with the locale encoding.
  - Candidate cuts are percentiles of the per-scene score, never absolute
    values, for the same reason the product uses percentiles.
"""
from __future__ import annotations

import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

from app.foundation_models.inferencers.inferencer_factory import get_inferencer  # noqa: E402
from app.models.dataset.vendables import BandFilterConfig, DEFAULT_COMMON_WAVELENGTH_GRID  # noqa: E402
from app.models.file_processing.sources import FileSourceConfig  # noqa: E402
from app.models.training.inference_config import InferenceConfig  # noqa: E402
from app.utils.anomaly_detection.scoring import compute_score  # noqa: E402
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder  # noqa: E402
from app.utils.pixel_fill.nearest_valid_fill import nearest_valid_fill  # noqa: E402
from allotrope.foundation_models.resolver import list_catalog  # noqa: E402

SCENES_ROOT = Path(r"C:\Users\FITT\Documents\stage1_datasets")
OUT_ROOT = SCENES_ROOT / "indradhanu_fp_filter" / "step1"
MODELS_DIR = REPO / "allotrope_models"

# Loose cuts to report: the top 0.5 / 1 / 2 / 5 % of scored pixels.
CUT_PERCENTILES = (99.5, 99.0, 98.0, 95.0)


def find_scenes() -> list[tuple[Path, Path]]:
    """(he5, hyperspectral GT tif) per Set-N, in Set order."""
    he5 = sorted(glob.glob(str(SCENES_ROOT / "Set-*" / "**" / "*.he5"), recursive=True))
    gts = sorted(
        g for g in glob.glob(str(SCENES_ROOT / "Set-*" / "**" / "*round*ruth*" / "*.tif"),
                             recursive=True)
        if "thermal" not in g.lower()
    )
    assert len(he5) == len(gts) == 4, (he5, gts)
    return [(Path(h), Path(g)) for h, g in zip(he5, gts)]


def build_vendable_cube(he5: Path) -> tuple[np.ndarray, np.ndarray]:
    """165-band reflectance cube (C, H, W) and spatial validity (H, W), as band_filter_apply builds it."""
    bf = BandFilterConfig(
        edge_bands_to_trim=3, min_valid_pixel_pct=20.0, max_invalid_voxel_fraction=0.4,
        quality_masks_to_apply=[], common_wavelength_grid=DEFAULT_COMMON_WAVELENGTH_GRID,
    )
    vendable = PrismaDatasetBuilder(FileSourceConfig(source_path=str(he5))).vend_dataset(
        band_filter_config=bf
    )
    spatial = (vendable.validity_cube.sum(axis=0) > 0).astype(np.int8)
    cube = nearest_valid_fill(vendable.normalized_hyperspectral_cube, spatial[None])
    return cube.astype(np.float32, copy=False), spatial.astype(bool)


def load_indradhanu():
    """Build the inferencer exactly as anomaly_scoring does, with the product defaults."""
    m = {x.codename.lower(): x for x in list_catalog(MODELS_DIR)}["indradhanu"]
    # The local models dir has no constants/ copy; the stats are the in-repo ones.
    cfg = InferenceConfig(
        foundation_model_name=m.foundation_model_name, model_config=m.model_config,
        checkpoint_path=m.checkpoint_abs_path, patch_size=m.default_patch_size,
        stride=m.default_stride, inference_batch_size=m.default_batch_size,
        pixel_stats_path=str(REPO / "app" / "constants" / "hyperspectral.json"),
    )
    return get_inferencer(cfg)


def recall_at_cut(score: np.ndarray, keep: np.ndarray, gt: np.ndarray, pct: float) -> dict:
    """Pixel- and object-level hit rates plus false-positive count at one percentile cut."""
    thr = float(np.percentile(score[keep], pct))
    flagged = (score >= thr) & keep
    labels, n_obj = ndimage.label(gt & keep)
    hit_objects = len(np.unique(labels[flagged & (labels > 0)]))
    return {
        "percentile": pct,
        "threshold": thr,
        "flagged_px": int(flagged.sum()),
        "true_px_caught": int((flagged & gt).sum()),
        "pixel_recall": float((flagged & gt).sum() / max(1, (gt & keep).sum())),
        "objects_caught": int(hit_objects),
        "objects_total": int(n_obj),
        "object_recall": float(hit_objects / max(1, n_obj)),
        "false_positive_px": int((flagged & ~gt).sum()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    inferencer = load_indradhanu()
    only = set(sys.argv[1:])  # optional Set numbers to run, e.g. "1"
    summary = []
    for n, (he5, gt_path) in enumerate(find_scenes(), start=1):
        if only and str(n) not in only:
            continue
        t0 = time.time()
        cube, keep = build_vendable_cube(he5)
        gt = tifffile.imread(gt_path).astype(bool)
        assert gt.shape == keep.shape, (gt.shape, keep.shape)
        t_build = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            recon = inferencer.predict_full_scene(
                torch.from_numpy(cube), torch.from_numpy(keep[None].astype(np.float32))
            ).cpu().numpy()
        t_infer = time.time() - t0

        out = OUT_ROOT / f"set{n}"
        out.mkdir(exist_ok=True)
        scores = {m: compute_score(cube, recon, keep, method=m) for m in ("L1", "SAM", "combined")}
        for m, s in scores.items():
            np.save(out / f"score_{m}.npy", s)
        np.save(out / "keep.npy", keep)
        np.save(out / "gt.npy", gt)

        rec = {
            "set": n, "scene": he5.stem, "shape": list(keep.shape),
            "valid_px": int(keep.sum()), "gt_px": int(gt.sum()),
            "gt_px_on_valid": int((gt & keep).sum()),
            "build_s": round(t_build, 1), "infer_s": round(t_infer, 1),
            "cuts": {m: [recall_at_cut(s, keep, gt, p) for p in CUT_PERCENTILES]
                     for m, s in scores.items()},
        }
        summary.append(rec)
        print(json.dumps({k: rec[k] for k in ("set", "valid_px", "gt_px_on_valid", "build_s", "infer_s")}),
              flush=True)
        del cube, recon

    (OUT_ROOT / ("summary.json" if not only else f"summary_{'_'.join(sorted(only))}.json")).write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
