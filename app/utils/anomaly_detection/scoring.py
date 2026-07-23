"""Pixel-wise anomaly score from a reconstruction + original cube.

Lifted from `benchmarking/hyperspectral/segformer_mae_benchmark.ipynb` § 3
into a shared module so the Allotrope `anomaly_detection` action handler
can call it for every foundation model. Each method returns a (H, W)
score map; higher = more anomalous.

Score methods:

- L1
    Mean absolute residual across bands. Works for any number of bands;
    natural choice for thermal scenes where SAM has no meaning.
        score(i, j) = mean_c |x(c, i, j) − x_hat(c, i, j)|

- MSE
    Mean squared residual across bands. Matches the loss used at training
    for the MSE-loss masked AE variants.

- SAM
    Spectral angle (radians) between observed and reconstructed spectra.
    Only meaningful when there are multiple bands. For single-band cubes
    SAM is undefined; callers should reject this method via the
    capabilities table.

- combined
    Convex combination of L1 and SAM after normalising each to its valid
    pixel max. Default for the hyperspectral SegFormer-MAE (Indradhanu) —
    matches the L1+SAM training loss.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


ScoringMethod = Literal["L1", "MSE", "SAM", "combined"]


def compute_score(
    original: np.ndarray,
    reconstruction: np.ndarray,
    validity: np.ndarray,
    method: ScoringMethod,
    *,
    combined_weight: float = 0.5,
) -> np.ndarray:
    """Compute a (H, W) anomaly score map.

    Args:
        original:       (C, H, W) input cube.
        reconstruction: (C, H, W) reconstructed cube (same shape).
        validity:       (H, W) uint/bool spatial validity mask. Invalid
                        pixels are zeroed in the output.
        method:         L1 | MSE | SAM | combined.
        combined_weight: w in `w * L1_norm + (1-w) * SAM_norm` when
                        method='combined'. Default 0.5.

    Returns:
        (H, W) float32 score map; non-negative, validity-masked.
    """
    if original.shape != reconstruction.shape:
        raise ValueError(
            f"shape mismatch: original {original.shape} vs "
            f"reconstruction {reconstruction.shape}"
        )
    if original.ndim != 3:
        raise ValueError(
            f"expected cube of shape (C, H, W); got {original.shape}"
        )
    spatial = validity.astype(bool)

    if method == "L1":
        score = np.mean(np.abs(original - reconstruction), axis=0)
    elif method == "MSE":
        diff = original - reconstruction
        score = np.mean(diff * diff, axis=0)
    elif method == "SAM":
        score = _sam(original, reconstruction)
    elif method == "combined":
        l1 = np.mean(np.abs(original - reconstruction), axis=0)
        sam = _sam(original, reconstruction)
        l1n = _normalise(l1, spatial)
        samn = _normalise(sam, spatial)
        score = combined_weight * l1n + (1.0 - combined_weight) * samn
    else:
        raise ValueError(f"unknown scoring method {method!r}")

    score = score.astype(np.float32, copy=False)
    score[~spatial] = 0.0
    return score


def _sam(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Spectral angle (radians) between observed and reconstructed
    spectra. Returns (H, W)."""
    eps = 1e-8
    dot = (original * reconstruction).sum(axis=0)
    norm_o = np.sqrt((original * original).sum(axis=0) + eps)
    norm_r = np.sqrt((reconstruction * reconstruction).sum(axis=0) + eps)
    cos_sim = np.clip(dot / (norm_o * norm_r), -1.0, 1.0)
    return np.arccos(cos_sim).astype(np.float32, copy=False)


def _normalise(arr: np.ndarray, validity: np.ndarray) -> np.ndarray:
    """Scale `arr` to [0, 1] over its valid-pixel range. Robust to flat
    inputs (returns zeros)."""
    vals = arr[validity]
    if vals.size == 0:
        return np.zeros_like(arr)
    hi = float(vals.max())
    if hi < 1e-12:
        return np.zeros_like(arr)
    out = (arr / hi).astype(np.float32, copy=False)
    out[~validity] = 0.0
    return out


def compute_roc(
    score: np.ndarray, gt: np.ndarray, validity: np.ndarray,
    *, max_thresholds: int = 256,
) -> dict:
    """Compute FPR / TPR / AUC for a score map against a ground-truth
    binary mask. Restricted to validity-masked pixels.

    Returns: {"fpr": [...], "tpr": [...], "thresholds": [...], "auc": float,
              "n_positive": int, "n_negative": int}

    `max_thresholds` caps the number of points on the curve — at most
    256 sampled threshold values, evenly spaced over the score range
    plus the unique-value breakpoints sklearn would emit. For our
    purposes 256 points is more than the eye can resolve and keeps the
    diagnostics JSON compact.
    """
    mask = validity.astype(bool)
    s = score[mask].astype(np.float32, copy=False)
    g = gt[mask].astype(np.int8, copy=False)
    n_pos = int(g.sum())
    n_neg = int(g.size - n_pos)

    if n_pos == 0 or n_neg == 0:
        return {
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
            "thresholds": [0.0, 0.0],
            "auc": 0.5,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "degenerate": True,
        }

    # Build a sweep of thresholds: percentiles of the score over valid
    # pixels, capped to max_thresholds.
    n_pts = min(max_thresholds, max(20, s.size // 10000 + 64))
    qs = np.linspace(0.0, 100.0, n_pts)
    ths = np.percentile(s, qs)
    ths = np.unique(ths)[::-1]  # descending so positives accumulate from the top

    tpr_list: list[float] = []
    fpr_list: list[float] = []
    for t in ths:
        pred = s >= t
        tp = int(((pred & (g == 1))).sum())
        fp = int(((pred & (g == 0))).sum())
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Anchor the curve at (0, 0) and (1, 1).
    fpr_arr = np.concatenate([[0.0], np.asarray(fpr_list), [1.0]])
    tpr_arr = np.concatenate([[0.0], np.asarray(tpr_list), [1.0]])
    order = np.argsort(fpr_arr, kind="stable")
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]
    # Trapezoidal AUC.
    auc = float(np.trapezoid(tpr_arr, fpr_arr))

    return {
        "fpr": [float(v) for v in fpr_arr],
        "tpr": [float(v) for v in tpr_arr],
        "thresholds": [float(t) for t in np.concatenate([[float("inf")], ths, [-float("inf")]])][: len(fpr_arr)],
        "auc": auc,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }
