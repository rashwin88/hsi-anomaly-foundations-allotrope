"""
Residual Mahalanobis scoring: how unusual a pixel's reconstruction error is, not how large.

Used by compute_score (method "residual_mahalanobis"), the default score for Indradhanu in the
anomaly_scoring action since 2026-10-15. Ported from research/experiments/dual-window-mahalanobis/
dw_core.py (whitened_pca, local_mean_scores) with the settings found there; see that README for the
derivation and docs/lld/residual-mahalanobis.md for the decision.

Why: L1 and SAM measure the size of the residual r = x - x_hat. A scene's normal residuals are large
along a few shared directions - overall brightness, haze, detector striping, dark water - and tiny
along the rest. Projecting r onto its top-d principal components, whitening each by its spread, and
scoring the squared length (Mahalanobis distance) discounts those common directions and rewards
errors in directions the background never takes. Measured on the four labelled PRISMA scenes
(mean ROC-AUC): v0.2.0 0.662 (combined) -> 0.840 (global mean) -> 0.854 (local mean); F001 at epoch
52 0.715 -> 0.849 -> 0.865, better on every scene.

    X     = whitened top-d principal components of r over valid pixels      (d, H, W)
    score = || X(p) - mean of X over the ring between the inner and outer window ||^2
            (local_window); or || X(p) ||^2 with the global mean (local_window=None)

Gotchas:
  - The covariance is the scene's own, from up to 200,000 valid pixels (fixed seed): deterministic,
    and well conditioned at d = 30 where full-band RX is not (hence plain RX was dropped in 2026-05).
  - Windows are full odd widths. A pixel whose ring holds fewer than 2 d valid pixels (scene edges,
    beside nodata) falls back to the global-mean score.
  - Scores are squared distances (chi-squared-like, heavy right tail): render with a sqrt stretch,
    threshold by percentile like every other score.
"""

import numpy as np

PCA_COMPONENTS = 30
LOCAL_WINDOW = (21, 61)          # inner (guard) and outer full widths, the PRISMA setting
MIN_RING_FACTOR = 2              # ~2 d ring samples keeps the mean estimate stable
_SAMPLE = 200_000


def whitened_pca(residual: np.ndarray, keep: np.ndarray, d: int = PCA_COMPONENTS, seed: int = 0) -> np.ndarray:
    """(C, H, W) residual -> (d, H, W) float32 whitened top-d principal components; 0 outside keep."""
    X = residual[:, keep].T.astype(np.float64)                       # (N, C)
    mu = X.mean(0)
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(len(X), _SAMPLE), replace=False)] - mu
    evals, evecs = np.linalg.eigh(np.cov(sub.T))
    order = np.argsort(evals)[::-1][:d]
    P = ((X - mu) @ evecs[:, order]) / np.sqrt(np.maximum(evals[order], 1e-12))
    out = np.zeros((len(order),) + keep.shape, dtype=np.float32)
    out[:, keep] = P.T
    return out


def _box(I: np.ndarray, k: int, r: int, h: int, w: int) -> np.ndarray:
    """Sum over the (2k+1)^2 box centred on each pixel, from a zero-led integral image padded by r."""
    a, b = r - k, r + k + 1
    return I[:, b:b + h, b:b + w] - I[:, a:a + h, b:b + w] - I[:, b:b + h, a:a + w] + I[:, a:a + h, a:a + w]


def local_mean_score(X: np.ndarray, keep: np.ndarray, window: tuple[int, int] = LOCAL_WINDOW) -> np.ndarray:
    """||x - ring mean||^2 in whitened space (local mean, global covariance); (H, W) float32."""
    d, H, W = X.shape
    inner, outer = window
    r = outer // 2
    F = np.concatenate([keep[None], X * keep], 0).astype(np.float64)
    I = np.pad(np.pad(F, ((0, 0), (r, r), (r, r))).cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    A = _box(I, outer // 2, r, H, W) - _box(I, inner // 2, r, H, W)   # ring: count, then sums
    n = A[0]
    ok = keep & (n >= MIN_RING_FACTOR * d)
    s = ((X - A[1:] / np.maximum(n, 1)) ** 2).sum(0)
    return (np.where(ok, s, (X ** 2).sum(0)) * keep).astype(np.float32)


def residual_mahalanobis(original: np.ndarray, reconstruction: np.ndarray, keep: np.ndarray,
                         d: int = PCA_COMPONENTS, local_window: tuple[int, int] | None = LOCAL_WINDOW) -> np.ndarray:
    """(H, W) float32 score map; 0 outside keep. local_window=None scores against the global mean."""
    keep = keep.astype(bool)
    X = whitened_pca(original - reconstruction, keep, d)
    if local_window is None:
        return ((X ** 2).sum(0) * keep).astype(np.float32)
    return local_mean_score(X, keep, local_window)
