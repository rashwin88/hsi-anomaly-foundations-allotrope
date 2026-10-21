"""
Is a difference between two scorers real? Paired bootstrap over labelled objects (research/experiments/bench-resolution/).

The real PRISMA bench is 4 scenes and 909 objects, and the gains we chase are ~0.005-0.01 pAUC. A difference counts
only if it survives resampling the objects: draw each scene's objects with replacement (the same draw for both
scorers - paired), recompute both measures, repeat; the 2.5th-97.5th percentiles of the difference are its 95%
interval. Background pixels are not resampled (there are millions; their sampling error is small next to the
objects').

pAUC from objects: each target pixel p contributes c(p) = max(0, max_fpr - FPR(p)) / max_fpr, FPR(p) being the
share of background pixels scoring at or above it; a scene's pAUC is the sum of c over its target pixels divided by
their count, so an object's (sum of c, pixel count) is all a bootstrap draw needs. Equal to metrics.pauc up to ties.
"""

from __future__ import annotations

import numpy as np


def object_pauc_parts(score: np.ndarray, valid: np.ndarray, objects: np.ndarray, max_fpr: float = 0.01) -> dict:
    """object id -> (sum of c over its pixels, pixel count) inside valid."""
    bg = np.sort(score[valid & (objects == 0)])
    t = valid & (objects > 0)
    fpr = 1.0 - np.searchsorted(bg, score[t], side="left") / max(len(bg), 1)     # share of background >= score
    c = np.maximum(0.0, max_fpr - fpr) / max_fpr
    ids = objects[t]
    sums = np.bincount(ids, weights=c)
    counts = np.bincount(ids)
    return {int(i): (float(sums[i]), int(counts[i])) for i in np.unique(ids)}


def paired_bootstrap(a: dict, b: dict, hits_a: dict | None = None, hits_b: dict | None = None, n: int = 2000,
                     seed: int = 0) -> dict:
    """a, b: scene -> object_pauc_parts for scorers A and B (same objects). hits_*: scene -> {object: 0/1} caught
    at some cut-off (optional). Returns point values, B - A, and the 95% interval of B - A for mean-over-scenes pAUC
    and (if given) the share of objects caught."""
    rng = np.random.default_rng(seed)
    scenes = sorted(a)
    ids = {s: np.array(sorted(a[s])) for s in scenes}
    arr = {s: (np.array([a[s][i] for i in ids[s]]), np.array([b[s][i] for i in ids[s]])) for s in scenes}
    hit = None
    if hits_a is not None:
        hit = {s: (np.array([hits_a[s][i] for i in ids[s]]), np.array([hits_b[s][i] for i in ids[s]])) for s in scenes}

    def measure(draws):
        pa, pb, ha, hb, m = [], [], 0.0, 0.0, 0
        for s, k in zip(scenes, draws):
            A, B = arr[s][0][k], arr[s][1][k]
            pa.append(A[:, 0].sum() / A[:, 1].sum())
            pb.append(B[:, 0].sum() / B[:, 1].sum())
            if hit is not None:
                ha, hb, m = ha + hit[s][0][k].sum(), hb + hit[s][1][k].sum(), m + len(k)
        return np.mean(pa), np.mean(pb), (ha / m if m else np.nan), (hb / m if m else np.nan)

    point = measure([np.arange(len(ids[s])) for s in scenes])
    boots = np.array([measure([rng.integers(len(ids[s]), size=len(ids[s])) for s in scenes]) for _ in range(n)])
    out = {"pauc_a": point[0], "pauc_b": point[1], "pauc_diff": point[1] - point[0],
           "pauc_ci": tuple(np.percentile(boots[:, 1] - boots[:, 0], [2.5, 97.5])),
           "pauc_p_b_better": float(np.mean(boots[:, 1] > boots[:, 0]))}
    if hit is not None:
        out.update({"caught_a": point[2], "caught_b": point[3], "caught_diff": point[3] - point[2],
                    "caught_ci": tuple(np.percentile(boots[:, 3] - boots[:, 2], [2.5, 97.5]))})
    return out
