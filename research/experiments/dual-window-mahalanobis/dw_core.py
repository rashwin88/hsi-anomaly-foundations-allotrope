"""
Dual-window Mahalanobis: score each pixel against the ring of pixels between
an inner (guard) window and an outer window, instead of the whole scene.

Shared by the PRISMA and public-dataset runs. Everything here works on a
(d, H, W) feature stack — the top-d principal components, whitened — plus a
validity mask, so the same code scores Indradhanu residuals and raw cubes.

How it stays affordable: the ring's mean and covariance only need three box
sums — count, sum of x, sum of x·xᵀ — and a box sum over any window is four
lookups in an integral image. So one integral image per row strip serves
every (inner, outer) pair in the sweep, and a 61×61 window costs the same as
a 15×15 one. app/detectors/_local_background.py gathers the ring pixels
explicitly instead, which is exact but scales with window area.

Gotchas:

  - Window sizes are FULL widths (odd), e.g. inner 11 = a guard box of 11×11
    centred on the pixel. app/detectors/LocalRX takes half-widths instead.
  - Pixels whose ring holds fewer than MIN_RING_FACTOR·d valid pixels (scene
    edges, beside nodata) fall back to the global score ‖x‖², which is global
    RX in whitened PCA space. The fraction that fell back is reported.
  - The ridge is relative: reg · trace(C)/d per pixel, because a local ring's
    variance can be orders of magnitude below the global one. It never drops
    below reg · RIDGE_FLOOR, since filled nodata makes perfectly flat rings.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_RING_FACTOR = 2   # Reed-Mallett-Brennan: ~2d samples keeps the SNR loss within 3 dB
RIDGE_FLOOR = 1e-3    # whitened units: the global variance of every component is 1
EIGHT = np.ones((3, 3), dtype=bool)
FROC_STEPS = 40


def whitened_pca(cube: np.ndarray, keep: np.ndarray, d: int = 30, seed: int = 0) -> np.ndarray:
    """(d, H, W) float32 whitened top-d PCs of a (C, H, W) cube; zero outside keep."""
    X = cube[:, keep].T.astype(np.float64)
    mu = X.mean(0)
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(len(X), 200_000), replace=False)] - mu
    evals, evecs = np.linalg.eigh(np.cov(sub.T))
    order = np.argsort(evals)[::-1][:d]
    P = ((X - mu) @ evecs[:, order]) / np.sqrt(evals[order])
    out = np.zeros((d,) + keep.shape, dtype=np.float32)
    out[:, keep] = P.T
    return out


def _box(I: torch.Tensor, k: int, r: int, h: int, w: int) -> torch.Tensor:
    """Sum over a (2k+1)² box centred on each output pixel, from a zero-led integral image."""
    a, b = r - k, r + k + 1
    return I[:, b:b + h, b:b + w] - I[:, a:a + h, b:b + w] - I[:, b:b + h, a:a + w] + I[:, a:a + h, a:a + w]


def dual_window_scores(X: np.ndarray, keep: np.ndarray, windows: list[tuple[int, int]],
                       reg: float = 1e-3, strip: int = 48) -> tuple[dict, dict]:
    """
    Dual-window Mahalanobis for each (inner, outer) full width in `windows`.

    Returns ({window: (H, W) float32 score}, {window: fraction of valid pixels
    that fell back to the global score}).
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, H, W = X.shape
    r = max(o for _, o in windows) // 2
    Xp = np.pad(X * keep, ((0, 0), (r, r), (r, r)))
    mp = np.pad(keep.astype(np.float32), r)
    iu = torch.triu_indices(d, d, device=dev)
    eye = torch.eye(d, device=dev, dtype=torch.float64)
    glob = torch.from_numpy((X ** 2).sum(0).reshape(-1)).to(dev, torch.float64)
    scores = {w: np.zeros((H, W), np.float32) for w in windows}
    fallback = {w: 0 for w in windows}

    for y0 in range(0, H, strip):
        h = min(strip, H - y0)
        xs = torch.from_numpy(Xp[:, y0:y0 + h + 2 * r]).to(dev, torch.float64)
        ms = torch.from_numpy(mp[None, y0:y0 + h + 2 * r]).to(dev, torch.float64)
        F = torch.cat([ms, xs, (xs[iu[0]] * xs[iu[1]])], 0)
        I = torch.nn.functional.pad(F.cumsum(1).cumsum(2), (1, 0, 1, 0))
        del F
        x = xs[:, r:r + h, r:r + W].reshape(d, -1).T                  # (h·W, d)
        valid = ms[0, r:r + h, r:r + W].reshape(-1) > 0
        for inner, outer in windows:
            A = _box(I, outer // 2, r, h, W) - _box(I, inner // 2, r, h, W)
            A = A.reshape(A.shape[0], -1).T                           # (h·W, 1+d+tri)
            n, S1 = A[:, 0], A[:, 1:1 + d]
            S2 = torch.zeros(len(A), d, d, device=dev, dtype=torch.float64)
            S2[:, iu[0], iu[1]] = A[:, 1 + d:]
            S2 = S2 + S2.transpose(1, 2) - torch.diag_embed(S2.diagonal(dim1=1, dim2=2))
            ok = valid & (n >= MIN_RING_FACTOR * d)
            mu = S1[ok] / n[ok, None]
            C = (S2[ok] - n[ok, None, None] * mu[:, :, None] * mu[:, None, :]) / (n[ok, None, None] - 1)
            # floor: flat rings (e.g. nearest-valid fill) have trace 0, and global variance is 1
            ridge = reg * C.diagonal(dim1=1, dim2=2).mean(1).clamp_min(RIDGE_FLOOR)
            C = C + ridge[:, None, None] * eye
            dx = x[ok] - mu
            s = glob[y0 * W:(y0 + h) * W].clone()
            s[ok] = (dx * torch.linalg.solve(C, dx)).sum(1)
            s[~valid] = 0
            scores[(inner, outer)][y0:y0 + h] = s.reshape(h, W).float().cpu().numpy()
            fallback[(inner, outer)] += int((valid & ~ok).sum())
        del I, xs, ms
    n_valid = max(1, int(keep.sum()))
    return scores, {w: f / n_valid for w, f in fallback.items()}


def segmented_dual_window_scores(X: np.ndarray, keep: np.ndarray, labels: np.ndarray,
                                 windows: list[tuple[int, int]], min_class_px: int = 500) -> dict:
    """
    Dual-window Mahalanobis whose ring only counts pixels of the centre pixel's own class.

    Run once per class with keep narrowed to that class, so a ring never
    straddles a land-cover edge. Classes under min_class_px pixels, and rings
    too small after the narrowing, fall back to the global score as usual.
    """
    out = {w: (X ** 2).sum(0).astype(np.float32) * keep for w in windows}
    for c in np.unique(labels[keep]):
        m = keep & (labels == c)
        if m.sum() < min_class_px:
            continue
        scores, _ = dual_window_scores(X, m, windows)
        for w in windows:
            out[w][m] = scores[w][m]
    return out


def local_mean_scores(X: np.ndarray, keep: np.ndarray, windows: list[tuple[int, int]]) -> dict:
    """‖x − ring mean‖² in whitened space: a local mean with the global covariance."""
    d, H, W = X.shape
    r = max(o for _, o in windows) // 2
    F = np.concatenate([keep[None], X * keep], 0).astype(np.float64)
    I = np.pad(np.pad(F, ((0, 0), (r, r), (r, r))).cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    It = torch.from_numpy(I)
    out = {}
    for inner, outer in windows:
        A = (_box(It, outer // 2, r, H, W) - _box(It, inner // 2, r, H, W)).numpy()
        n = A[0]
        ok = keep & (n >= MIN_RING_FACTOR * d)
        mu = A[1:] / np.maximum(n, 1)
        s = ((X - mu) ** 2).sum(0)
        out[(inner, outer)] = np.where(ok, s, (X ** 2).sum(0)).astype(np.float32) * keep
    return out


def evaluate(score: np.ndarray, keep: np.ndarray, gt: np.ndarray, buffer: int = 2,
             cuts: tuple[float, ...] = (99.5, 99.0, 98.0),
             curve: tuple[float, float] = (0.01, 5)) -> dict:
    """Pixel AUC / average precision, plus objects found and false-alarm spots at percentile cuts.

    Object counting follows indradhanu-fp-filter/step5: an object is found if a flag lands on it
    or within `buffer` px; flags in that neutral zone are not false alarms; other flags are
    grouped into 8-connected spots, one false alarm each.
    """
    g = gt & keep
    out = {"auc": float(roc_auc_score(g[keep], score[keep])),
           "ap": float(average_precision_score(g[keep], score[keep]))}
    labels, n_obj = ndimage.label(g, structure=EIGHT)
    zone = ndimage.binary_dilation(g, structure=np.ones((2 * buffer + 1,) * 2, bool)) if buffer else g
    zl = ndimage.label(zone, structure=EIGHT)[0]
    # each object's zone id; objects whose zones merge share an id, as in step5
    obj_zone = ndimage.maximum(zl, labels, index=np.arange(1, n_obj + 1)).astype(int)
    out["objects_total"] = int(n_obj)
    def at(p: float) -> tuple[int, int]:
        flag = keep & (score >= np.percentile(score[keep], p))
        found = int(np.isin(obj_zone, np.unique(zl[flag & zone])).sum())
        return found, int(ndimage.label(flag & ~zone, structure=EIGHT)[1])

    for p in cuts:
        f, sp = at(p)
        out[f"top{100 - p:g}"] = {"found": f, "spots": sp}
    # (top %, found, spots) over the `curve` range of top %, so detectors can be compared
    # at an equal false-alarm budget rather than an equal flagged share
    out["curve"] = [(round(t, 4), *at(100 - t)) for t in np.geomspace(*curve, FROC_STEPS)]
    return out
