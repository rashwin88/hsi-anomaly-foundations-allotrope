"""
Detection measures for one score map against labelled objects (docs/lld/evaluation.md).

Anomalies are 0.1-1% of a scene's pixels, so ROC-AUC is decided by how the background bulk is ranked; the measures
that matter look at the top of the ranking, where an analyst looks:

  roc_auc         the product's compute_roc (sampled thresholds) - kept as the headline AUC for continuity
  pauc            area under the EXACT ROC over FPR 0..max_fpr, divided by max_fpr (1 perfect, max_fpr/2 random)
  tpr_at          best detection rate reached with at most `fpr` false alarms (exact)
  objects_caught  objects with a pixel within 1 px of the top `top_pct`% of scores; its chance level is analytic
                  per object: 1 - (1 - q)^m, m = the object's pixel count after the 1-px dilation, q = top_pct / 100
  three_d_roc     AUC(D,tau), AUC(F,tau), AUC_BS, SNPR (min-max normalised; degenerate for heavy-tailed scores)

Every pixel measure is over valid pixels, objects as positives and non-object pixels as negatives; a group's
measures (group_by) take that group's objects as positives against the same negatives.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def exact_roc(s: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) at every distinct score, descending, from (0, 0). s scores, g bool truth, both 1-D."""
    order = np.argsort(-s, kind="stable")
    s_sorted, g_sorted = s[order], g[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]      # end of each run of tied scores
    return np.r_[0.0, np.cumsum(~g_sorted)[last] / max((~g).sum(), 1)], np.r_[0.0, np.cumsum(g_sorted)[last] / max(g.sum(), 1)]


def auc_exact(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def pauc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float = 0.01) -> float:
    """Area under the ROC over FPR 0..max_fpr, / max_fpr. Segment by segment, so vertical runs (several points at
    one FPR, e.g. targets ranked above all background) count fully."""
    x0, x1, y0, y1 = fpr[:-1], fpr[1:], tpr[:-1], tpr[1:]
    y1c = np.where(x1 > max_fpr, y0 + (y1 - y0) * (max_fpr - x0) / np.where(x1 > x0, x1 - x0, 1.0), y1)
    return float(np.sum(((np.minimum(x1, max_fpr) - x0) * (y0 + y1c) / 2)[x0 < max_fpr]) / max_fpr)


def tpr_at(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    return float(tpr[fpr <= max_fpr].max())


def three_d_roc(s: np.ndarray, g: np.ndarray, auc: float) -> dict:
    lo, hi = float(s.min()), float(s.max())
    norm = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s, dtype=float)
    d, f = float(norm[g].mean()), float(norm[~g].mean())
    return {"auc_d_tau": d, "auc_f_tau": f, "auc_bs": auc - f, "snpr": d / f if f else float("inf")}


def object_hits(score: np.ndarray, valid: np.ndarray, objects: np.ndarray, top_pct: float) -> set:
    """Ids of objects with a pixel within 1 px (4-neighbour) of the top top_pct% of valid scores."""
    k = valid & np.isfinite(score)
    hits = ndimage.binary_dilation((score >= np.percentile(score[k], 100 - top_pct)) & k)
    return set(np.unique(objects[hits & (objects > 0)]).tolist())


def dilated_sizes(objects: np.ndarray) -> dict:
    """Object id -> pixel count after one 4-neighbour dilation (the tolerance objects_caught allows), windowed."""
    out = {}
    for i, sl in enumerate(ndimage.find_objects(objects), 1):
        if sl is None:
            continue
        win = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
        out[i] = int(ndimage.binary_dilation(objects[win] == i).sum())
    return out


def chance_hit(m: int, top_pct: float) -> float:
    return 1.0 - (1.0 - top_pct / 100.0) ** m


def scene_metrics(score, valid, objects, attrs: dict, metrics: list, group_by: list) -> tuple[list, dict]:
    """(per-object rows, per-scene measures) for one score map. objects (H, W) int ids, attrs id -> {key: value}.
    metrics: MetricSpec-like objects (kind, max_fpr, fpr, top_pct). Group measures are keyed "<metric>|<key>=<value>"."""
    from app.utils.anomaly_detection.scoring import compute_roc
    valid = valid.astype(bool)
    k = valid & np.isfinite(score)
    pos = (objects > 0) & k
    neg = (objects == 0) & k
    sizes = dilated_sizes(objects)
    ids = sorted(sizes)
    rows = [{"object": i, **attrs.get(i, {}), "size_dilated": sizes[i]} for i in ids]
    out: dict = {"objects": len(ids)}

    def pixel_measures(sel_pos, prefix=""):
        sel = sel_pos | neg
        s, g = score[sel], sel_pos[sel]
        if g.sum() == 0 or (~g).sum() == 0:
            return
        fpr, tpr = exact_roc(s, g)
        for m in metrics:
            if m.kind == "roc_auc" and not prefix:
                out["roc_auc"] = float(compute_roc(score, (objects > 0).astype(np.uint8), valid.astype(np.uint8))["auc"])
            elif m.kind == "roc_auc":
                out[f"roc_auc_exact{prefix}"] = auc_exact(fpr, tpr)
            elif m.kind == "pauc":
                out[f"pauc_{m.max_fpr:g}{prefix}"] = pauc(fpr, tpr, m.max_fpr)
            elif m.kind == "tpr_at":
                out[f"tpr_at_{m.fpr:g}{prefix}"] = tpr_at(fpr, tpr, m.fpr)
            elif m.kind == "three_d_roc" and not prefix:
                out.update(three_d_roc(s, g, auc_exact(fpr, tpr)))

    pixel_measures(pos)
    for m in metrics:
        if m.kind == "objects_caught":
            for q in m.top_pct:
                hit = object_hits(score, valid, objects, q)
                for r in rows:
                    r[f"hit_{q:g}"] = r["object"] in hit
                    r[f"chance_{q:g}"] = chance_hit(r["size_dilated"], q)
    for key in group_by:
        for value in sorted({str(a.get(key)) for a in attrs.values() if key in a}):
            members = [i for i in ids if str(attrs.get(i, {}).get(key)) == value]
            pixel_measures(np.isin(objects, members) & k, f"|{key}={value}")
    return rows, out
