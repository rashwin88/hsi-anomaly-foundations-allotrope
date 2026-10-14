"""
The 4 labelled PRISMA anomaly scenes: loading them the way each model needs, and scoring a model's
anomaly map against the ground truth.

Used by scripts/evaluate_prisma_bench.py - the detection half of the SpectralEarth pretraining
decision rule (docs/proposals/spectral-attention-bottleneck.md) and the reference every new
Indradhanu is held to (production v0.2.0, mean combined ROC-AUC 0.674 as measured 2026-10-02).

Why three loading modes: the comparison is only fair if each model reads the scene the way it was
built to.
  product165  exactly what the product's band_filter_apply hands anomaly_scoring for PRISMA: default
              exclusions, 3 edge bands trimmed, 165-band common grid, nearest-valid fill. v0.2.0.
  se202       resampled onto SpectralEarth's 202 wavelengths (sorted ascending), exclusions narrowed
              to SpectralEarth's own water gaps and no edge trimming, so every one of the 202 can be
              filled. The fixed-band stems (A, B) need exactly these; C' can read them too.
  native      PRISMA's own bands after the default band filter, no resampling - C' only.

Gotchas:
  - Scores are compared with the product's own compute_roc, so AUCs match the product's numbers.
  - Ground truth may sit ~1 px off the scene grid; "objects caught" allows a 1-pixel tolerance, AUC
    does not (it is the product's AUC).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Relative to the stage1_datasets root (C:/Users/FITT/Documents/stage1_datasets on the dev laptop).
SCENES = {
    "set1": ("Set-1/Hypersepctral Datasets/PRS_L2D_STD_20201214060713_20201214060717_0001.he5",
             "Set-1/Hypersepctral Datasets/Ground truth/groundtruth_hyper.tif"),
    "set2": ("Set-2/Hypersepctral Dataset/PRS_L2D_STD_20241205050514_20241205050518_0001.he5",
             "Set-2/Hypersepctral Dataset/Ground truth/Hyp_Ground_Truth.tif"),
    "set3": ("Set-3/Hyper/PRS_L2D_STD_20210516050459_20210516050503_0001.he5",
             "Set-3/Hyper/Ground truth/Groundtruth hyper anomalies.tif"),
    "set4": ("Set-4/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5",
             "Set-4/Hyper/Ground truth/ground_truth_hyper.tif"),
}
MODES = ("product165", "se202", "native")
# SpectralEarth dropped EnMAP's water bands at 1295-1507 and 1728-1968 nm; only those gaps are
# excluded in se202 so every SpectralEarth wavelength has PRISMA bands on both sides of it.
SE202_EXCLUSIONS = [(1300.0, 1512.0), (1722.0, 1972.0)]


@dataclass
class BenchScene:
    name: str
    cube: np.ndarray          # (C, H, W) float32 reflectance (nearest-valid filled)
    validity: np.ndarray      # (C, H, W) int8, 1 = measured
    wavelengths_nm: np.ndarray  # (C,)
    spatial_valid: np.ndarray   # (H, W) uint8
    gt: np.ndarray            # (H, W) uint8, 1 = anomaly


def load_scene(root: str | Path, name: str, mode: str) -> BenchScene:
    """One benchmark scene in one loading mode (see the module docstring)."""
    from app.models.dataset.spectralearth import load_band_table
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID, BandFilterConfig
    from app.models.file_processing.sources import FileSourceConfig
    from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
    from app.utils.pixel_fill.nearest_valid_fill import nearest_valid_fill

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    he5, gt_rel = (Path(root) / p for p in SCENES[name])
    product = dict(exclusion_ranges=[(0, 450), (912, 978), (1131, 1152), (1350, 1450), (1800, 1950)],
                   edge_bands_to_trim=3, min_valid_pixel_pct=20.0, max_invalid_voxel_fraction=0.4,
                   quality_masks_to_apply=[])
    if mode == "product165":
        cfg = BandFilterConfig(**product, common_wavelength_grid=DEFAULT_COMMON_WAVELENGTH_GRID)
    elif mode == "se202":
        se = np.sort(load_band_table().wavelengths_nm)
        cfg = BandFilterConfig(**{**product, "exclusion_ranges": SE202_EXCLUSIONS, "edge_bands_to_trim": 0},
                               common_wavelength_grid=se)
    else:
        cfg = BandFilterConfig(**product, common_wavelength_grid=None)
    v = PrismaDatasetBuilder(FileSourceConfig(source_path=str(he5))).vend_dataset(band_filter_config=cfg)
    validity = np.asarray(v.validity_cube, dtype=np.int8)
    spatial = (validity.sum(axis=0) > 0).astype(np.uint8)
    cube = nearest_valid_fill(v.normalized_hyperspectral_cube, spatial[None]).astype(np.float32)
    return BenchScene(name, cube, validity, np.asarray(v.band_cw_order, dtype=np.float64), spatial,
                      load_gt(gt_rel, spatial.shape))


def load_gt(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) uint8 anomaly mask (> 0). Raises ValueError if it does not match the scene grid."""
    import rasterio
    with rasterio.open(path) as src:
        gt = (src.read(1) > 0).astype(np.uint8)
    if gt.shape != shape:
        raise ValueError(f"{path.name}: ground truth {gt.shape} vs scene {shape}")
    return gt


def score_metrics(score: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> dict:
    """ROC-AUC (the product's compute_roc), TPR at 1% FPR, and how many ground-truth objects have at
    least one pixel in the top 1% of scores (1-pixel tolerance), over `keep` pixels."""
    from scipy import ndimage
    from app.utils.anomaly_detection.scoring import compute_roc

    roc = compute_roc(score, gt, keep)
    fpr, tpr = np.asarray(roc["fpr"]), np.asarray(roc["tpr"])
    tpr_1 = float(np.interp(0.01, fpr, tpr)) if len(fpr) > 1 else float("nan")
    k = keep.astype(bool) & np.isfinite(score)
    cut = np.percentile(score[k], 99.0)
    hits = ndimage.binary_dilation((score >= cut) & k, iterations=1)
    labels, n_obj = ndimage.label(gt)
    caught = len(np.unique(labels[hits & (labels > 0)]))
    return {"auc": float(roc["auc"]), "tpr_at_1pct_fpr": tpr_1, "objects": int(n_obj), "objects_caught": int(caught),
            "n_positive": int(roc.get("n_positive", 0))}
