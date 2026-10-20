"""
Scorers: (cube, reconstruction, valid, classes) -> (H, W) score map, configured by a ScorerSpec
(docs/lld/evaluation.md).

    residual_mahalanobis  the product default (app/utils/anomaly_detection/residual_mahalanobis.py)
    combined, L1, SAM     the product's compute_score methods
    class_aware           per-class whitening + same-class ring (tested 2026-10-17, not the default)
    exclude               any base scorer with chosen classes scored 0; fit_on_kept fits the base on the
                          remaining pixels only (user rule 2026-10-17: no detection on water or cloud)
    pca_rx                the same whitened-PCA distance on the cube itself - a classical baseline, no model
    fuse                  rank-normalised parts combined by mean or max rank (gains experiment A2, 2026-10-19)
    mixture               residual_mahalanobis per k-means background type, same-type ring (A3)
    scaled_rm             residual_mahalanobis on residual / b ** scale_power, b = the model's own expected error
                          (uncertainty head, docs/proposals/uncertainty-head.md), floored at its scale_floor_pct
                          percentile over valid pixels (default 5)

spatial_sigma (A1) blurs the residual over valid pixels before whitening: a PSF-blurred target's residual adds up
across its footprint while uncorrelated noise averages down - a spatial matched filter. Experiment record:
research/experiments/scoring-gains/.
"""

from __future__ import annotations

import numpy as np

from app.evaluation.config import CLASS_IDS


def score(spec, cube: np.ndarray, recon: np.ndarray | None, valid: np.ndarray, classes: np.ndarray | None,
          log_b: np.ndarray | None = None) -> np.ndarray:
    from app.utils.anomaly_detection.residual_mahalanobis import (
        class_aware_residual_mahalanobis, residual_mahalanobis)
    from app.utils.anomaly_detection.scoring import compute_score
    valid = valid.astype(bool)
    if spec.kind in ("residual_mahalanobis", "pca_rx") and spec.spatial_sigma:
        return blurred_mahalanobis(cube - recon if spec.kind == "residual_mahalanobis" else cube, valid, spec)
    if spec.kind == "residual_mahalanobis":
        return residual_mahalanobis(cube, recon, valid, d=spec.d, local_window=spec.local_window)
    if spec.kind == "pca_rx":
        return residual_mahalanobis(cube, np.zeros_like(cube), valid, d=spec.d, local_window=spec.local_window)
    if spec.kind == "scaled_rm":
        if log_b is None:
            raise ValueError("scaled_rm needs a model with an uncertainty head")
        b = np.exp(log_b.astype(np.float64))
        b = np.maximum(b, np.percentile(b[valid], spec.scale_floor_pct)) ** spec.scale_power
        return blurred_mahalanobis(((cube - recon) / b[None]).astype(np.float32), valid, spec)
    if spec.kind == "fuse":
        ranks = np.stack([rank_map(score(p, cube, recon, valid, classes, log_b), valid) for p in spec.parts])
        return (ranks.mean(0) if spec.rule == "rank_mean" else ranks.max(0)).astype(np.float32)
    if spec.kind == "mixture":
        residual = cube - recon
        if spec.spatial_sigma:
            residual = blur(residual, valid, spec.spatial_sigma)
        return class_aware_residual_mahalanobis(residual, np.zeros_like(residual), valid, background_types(cube, valid, spec.k),
                                                d=spec.d, local_window=spec.local_window)
    if spec.kind in ("combined", "L1", "SAM"):
        return compute_score(cube, recon, valid.astype(np.uint8), method=spec.kind)
    if spec.kind == "class_aware":
        return class_aware_residual_mahalanobis(cube, recon, valid, classes, d=spec.d, local_window=spec.local_window)
    if spec.kind == "exclude":
        out = np.isin(classes, [CLASS_IDS[c] for c in spec.classes])
        s = score(spec.base, cube, recon, valid & ~out if spec.fit_on_kept else valid, classes, log_b)
        return np.where(out, 0.0, s).astype(np.float32)
    raise ValueError(f"unknown scorer kind {spec.kind!r}")


def blur(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """(C, H, W) Gaussian blur of each band over valid pixels only (normalised convolution, so nodata does not
    bleed in); 0 outside valid."""
    from scipy.ndimage import gaussian_filter
    k = valid.astype(np.float32)
    w = np.maximum(gaussian_filter(k, sigma), 1e-6)
    out = np.empty(x.shape, np.float32)
    for i in range(len(x)):
        out[i] = gaussian_filter(x[i] * k, sigma) / w * k
    return out


def blurred_mahalanobis(residual: np.ndarray, valid: np.ndarray, spec) -> np.ndarray:
    """residual_mahalanobis on the blurred residual: whitening is refitted on it, so the score is in units of the
    blurred background's own spread."""
    from app.utils.anomaly_detection.residual_mahalanobis import local_mean_score, whitened_pca
    X = whitened_pca(blur(residual, valid, spec.spatial_sigma) if spec.spatial_sigma else residual, valid, spec.d)
    if spec.local_window is None:
        return ((X ** 2).sum(0) * valid).astype(np.float32)
    return local_mean_score(X, valid, spec.local_window)


def rank_map(s: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Scores -> fraction of valid pixels at or below (ties averaged), 0 outside valid: puts scorers with
    different scales on one footing."""
    from scipy.stats import rankdata
    out = np.zeros(s.shape, np.float32)
    out[valid] = rankdata(s[valid], method="average") / valid.sum()
    return out


def background_types(cube: np.ndarray, valid: np.ndarray, k: int, pcs: int = 10, iters: int = 25,
                     sample: int = 50_000, seed: int = 0) -> np.ndarray:
    """(H, W) int background type 0..k-1: k-means (Lloyd, fixed seed) on the cube's whitened top-10 principal
    components, fitted on a sample, then every valid pixel assigned to its nearest centre."""
    from app.utils.anomaly_detection.residual_mahalanobis import whitened_pca
    X = whitened_pca(cube, valid, pcs)[:, valid].T                    # (N, pcs)
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), size=min(len(X), sample), replace=False)]
    C = S[rng.choice(len(S), size=k, replace=False)]
    for _ in range(iters):
        lab = ((S[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([S[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    out = np.zeros(valid.shape, np.int64)
    out[valid] = np.concatenate([((x[:, None] - C[None]) ** 2).sum(-1).argmin(1)
                                 for x in np.array_split(X, max(1, len(X) // 200_000))])
    return out
