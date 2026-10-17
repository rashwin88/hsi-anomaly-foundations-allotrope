"""
Scorers: (cube, reconstruction, valid, classes) -> (H, W) score map, configured by a ScorerSpec
(docs/lld/evaluation.md).

    residual_mahalanobis  the product default (app/utils/anomaly_detection/residual_mahalanobis.py)
    combined, L1, SAM     the product's compute_score methods
    class_aware           per-class whitening + same-class ring (tested 2026-10-17, not the default)
    exclude               any base scorer with chosen classes scored 0; fit_on_kept fits the base on the
                          remaining pixels only (user rule 2026-10-17: no detection on water or cloud)
    pca_rx                the same whitened-PCA distance on the cube itself - a classical baseline, no model
"""

from __future__ import annotations

import numpy as np

from app.evaluation.config import CLASS_IDS


def score(spec, cube: np.ndarray, recon: np.ndarray | None, valid: np.ndarray, classes: np.ndarray | None) -> np.ndarray:
    from app.utils.anomaly_detection.residual_mahalanobis import (
        class_aware_residual_mahalanobis, residual_mahalanobis)
    from app.utils.anomaly_detection.scoring import compute_score
    valid = valid.astype(bool)
    if spec.kind == "residual_mahalanobis":
        return residual_mahalanobis(cube, recon, valid, d=spec.d, local_window=spec.local_window)
    if spec.kind == "pca_rx":
        return residual_mahalanobis(cube, np.zeros_like(cube), valid, d=spec.d, local_window=spec.local_window)
    if spec.kind in ("combined", "L1", "SAM"):
        return compute_score(cube, recon, valid.astype(np.uint8), method=spec.kind)
    if spec.kind == "class_aware":
        return class_aware_residual_mahalanobis(cube, recon, valid, classes, d=spec.d, local_window=spec.local_window)
    if spec.kind == "exclude":
        out = np.isin(classes, [CLASS_IDS[c] for c in spec.classes])
        s = score(spec.base, cube, recon, valid & ~out if spec.fit_on_kept else valid, classes)
        return np.where(out, 0.0, s).astype(np.float32)
    raise ValueError(f"unknown scorer kind {spec.kind!r}")
