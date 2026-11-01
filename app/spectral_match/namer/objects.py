"""
From per-pixel namer results to named objects, and the object-level maps the product shows.

Called by the spectral_library_match worker (backend/allotrope/action_types/_spectral_namer_run.py) after
infer.name_pixels has named every pixel of the committed anomaly mask; scripts/aigc_object_table.py groups the same
way for the report.

Why objects: an object (an 8-connected component of the mask) is what an analyst points at, and a per-pixel call
speckles - edge pixels are mostly background, so their P(man-made) is lower than the core's. One call per object,
painted on all its pixels, gives a map that reads as "this thing is man-made", and it is what decides which pixels
go on to SAM and family matching. Per-pixel numbers are kept beside it (namer_pixels.parquet).

Per object: P(man-made) is the mean over its pixels, called man-made above MANMADE_CALL_P; the class is the one with
the highest mean P(class | man-made) over its pixels, "confident" when that mean is at least confident_p (else
tentative). Natural objects carry no class (cls "", p_cls NaN). Every object, natural ones included, also carries
`prob_<class>`, its mean P(class | man-made) for each class: the match screen moves the man-made cut-off (0.5 / 0.35
/ 0.25) client-side, and an object that becomes man-made there takes its top `prob_<class>` as its class.

Codes in the maps: level 1 is int8 -1 outside the mask, 0 natural, 1 man-made. The class map is int8 -1 outside the
mask, 0 natural object, 1..C a confident class (index + 1), TENTATIVE_OFFSET + 1..C the same class, tentative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Level-1 call on the object's mean P(man-made): the model's own decision boundary (sigmoid of the level-1 logit).
MANMADE_CALL_P = 0.5
OUTSIDE = -1
TENTATIVE_OFFSET = 10


@dataclass
class NamedObjects:
    labels: np.ndarray          # (H, W) int32, object id 1..n, 0 outside the mask
    pixel_obj: np.ndarray       # (P,) int32, object id of each named pixel
    table: dict                 # column -> (n,) array, one row per object, ordered by obj_id
    classes: tuple

    @property
    def manmade(self) -> np.ndarray:
        """(n,) bool, the object's level-1 call."""
        return self.table["call"] == "manmade"

    def manmade_pixels(self) -> np.ndarray:
        """(H, W) bool: every pixel of a man-made object."""
        keep = np.r_[False, self.manmade]               # index 0 = outside the mask
        return keep[self.labels]


def label_objects(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components of a 0/1 mask, as in scripts/aigc_object_table.py."""
    from scipy.ndimage import label

    labels, n = label(np.asarray(mask, bool), structure=np.ones((3, 3), int))
    return labels.astype(np.int32), int(n)


def name_objects(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray, p_manmade: np.ndarray, probs: np.ndarray,
                 fill: np.ndarray, classes: tuple, confident_p: float) -> NamedObjects:
    """Objects of `mask` named from per-pixel results at (rows, cols); probs (P, C) is P(class | man-made).

    Every pixel of the mask must be among (rows, cols) - the namer is run on the whole mask - or its object's means
    are taken over the named pixels only."""
    from scipy.ndimage import find_objects

    labels, n = label_objects(mask)
    obj = labels[rows, cols]
    if (obj == 0).any():
        raise ValueError("named pixels outside the mask")
    count = np.bincount(obj, minlength=n + 1)[1:].astype(np.float64)
    safe = np.maximum(count, 1)
    mean = lambda v: np.bincount(obj, weights=np.asarray(v, np.float64), minlength=n + 1)[1:] / safe
    p_mm = mean(p_manmade)
    cls_means = np.stack([mean(probs[:, c]) for c in range(len(classes))], axis=1) if n else np.zeros((0, len(classes)))
    best = cls_means.argmax(axis=1) if n else np.zeros(0, int)
    p_cls = cls_means[np.arange(n), best] if n else np.zeros(0)
    mm = p_mm > MANMADE_CALL_P
    r0, r1, c0, c1 = (np.zeros(n, np.int32) for _ in range(4))
    for i, sl in enumerate(find_objects(labels)):
        r0[i], r1[i], c0[i], c1[i] = sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1
    table = {
        "obj_id": np.arange(1, n + 1, dtype=np.int32),
        "n_px": count.astype(np.int32),
        "row": mean(rows).astype(np.float32), "col": mean(cols).astype(np.float32),
        "row_min": r0, "row_max": r1, "col_min": c0, "col_max": c1,
        "p_manmade": p_mm.astype(np.float32),
        "call": np.where(mm, "manmade", "natural").astype(object),
        "cls": np.array([classes[b] if m else "" for b, m in zip(best, mm)], dtype=object),
        "p_cls": np.where(mm, p_cls, np.nan).astype(np.float32),
        "confident": mm & (p_cls >= confident_p),
        "fill": mean(fill).astype(np.float32),
        **{f"prob_{c}": cls_means[:, i].astype(np.float32) for i, c in enumerate(classes)},
    }
    return NamedObjects(labels, obj.astype(np.int32), table, tuple(classes))


def level1_map(objects: NamedObjects) -> np.ndarray:
    """(H, W) int8: -1 outside the mask, 0 natural object, 1 man-made object."""
    lut = np.r_[OUTSIDE, objects.manmade.astype(np.int8)].astype(np.int8)
    return lut[objects.labels]


def class_map(objects: NamedObjects) -> np.ndarray:
    """(H, W) int8: -1 outside, 0 natural, class index + 1, plus TENTATIVE_OFFSET when tentative."""
    t = objects.table
    idx = {c: i + 1 for i, c in enumerate(objects.classes)}
    code = np.array([idx.get(c, 0) for c in t["cls"]], dtype=np.int16)
    code = np.where(t["confident"] | (code == 0), code, code + TENTATIVE_OFFSET)
    return np.r_[OUTSIDE, code].astype(np.int8)[objects.labels]
