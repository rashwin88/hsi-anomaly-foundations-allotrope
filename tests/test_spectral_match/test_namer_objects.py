"""Named objects from per-pixel namer results (app/spectral_match/namer/objects.py)."""

import numpy as np
import pytest

from app.spectral_match.namer.objects import (OUTSIDE, TENTATIVE_OFFSET, class_map, level1_map,
                                              name_objects)

CLASSES = ("oil_fuel", "plastic", "metal")


def _scene():
    """Three objects on a 12 x 12 mask: a 2 x 3 block, a diagonal pair (8-connected), one pixel."""
    mask = np.zeros((12, 12), bool)
    mask[1:3, 1:4] = True                  # object 1, 6 px
    mask[6, 6] = mask[7, 7] = True         # object 2, 2 px - diagonal neighbours are one object
    mask[10, 1] = True                     # object 3, 1 px
    rows, cols = np.nonzero(mask)
    P = len(rows)
    p_mm = np.zeros(P)
    probs = np.zeros((P, len(CLASSES)))
    fill = np.full(P, 0.3)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if r < 3:                          # man-made, plastic leads on average but not per pixel
            p_mm[i] = 0.9 if c < 3 else 0.3
            probs[i] = (0.1, 0.8, 0.1) if c != 1 else (0.6, 0.3, 0.1)
        elif r < 8:                        # man-made, metal at 0.5: tentative at 0.55
            p_mm[i], probs[i] = 0.7, (0.25, 0.25, 0.5)
        else:                              # natural
            p_mm[i], probs[i] = 0.2, (0.0, 0.0, 1.0)
    return mask, rows, cols, p_mm, probs, fill


def test_objects_take_mean_p_manmade_and_the_highest_mean_class():
    mask, rows, cols, p_mm, probs, fill = _scene()
    objs = name_objects(mask, rows, cols, p_mm, probs, fill, CLASSES, confident_p=0.55)
    t = objs.table
    assert t["obj_id"].tolist() == [1, 2, 3]
    assert t["n_px"].tolist() == [6, 2, 1]
    assert t["p_manmade"][0] == pytest.approx((0.9 * 4 + 0.3 * 2) / 6)
    assert t["call"].tolist() == ["manmade", "manmade", "natural"]
    # object 1: mean plastic (0.3 + 0.8 * 2) * 2 / 6 beats mean oil (0.6 + 0.1 * 2) * 2 / 6
    assert t["cls"].tolist() == ["plastic", "metal", ""]
    assert t["p_cls"][0] == pytest.approx((0.3 + 0.8 + 0.8) / 3)
    assert t["confident"].tolist() == [True, False, False]          # metal 0.5 < 0.55: tentative
    assert np.isnan(t["p_cls"][2])
    assert (t["row"][0], t["col"][0]) == pytest.approx((1.5, 2.0))
    assert (t["row_min"][1], t["row_max"][1], t["col_min"][1], t["col_max"][1]) == (6, 7, 6, 7)
    assert t["fill"] == pytest.approx([0.3, 0.3, 0.3])
    assert objs.pixel_obj.tolist() == [1] * 6 + [2, 2, 3]


def test_every_object_keeps_its_mean_probability_of_every_class():
    """The match screen re-calls objects at a lower man-made cut-off; a natural object needs its class means."""
    mask, rows, cols, p_mm, probs, fill = _scene()
    t = name_objects(mask, rows, cols, p_mm, probs, fill, CLASSES, confident_p=0.55).table
    assert t["prob_plastic"][0] == pytest.approx((0.3 + 0.8 + 0.8) / 3)
    assert t["prob_metal"][2] == pytest.approx(1.0)                 # natural object 3 still carries its means
    assert t["prob_oil_fuel"].dtype == np.float32


def test_confident_threshold_is_inclusive():
    mask, rows, cols, p_mm, probs, fill = _scene()
    objs = name_objects(mask, rows, cols, p_mm, probs, fill, CLASSES, confident_p=0.5)
    assert objs.table["confident"].tolist() == [True, True, False]


def test_maps_paint_the_object_call_on_every_pixel():
    mask, rows, cols, p_mm, probs, fill = _scene()
    objs = name_objects(mask, rows, cols, p_mm, probs, fill, CLASSES, confident_p=0.55)
    l1 = level1_map(objs)
    assert l1.dtype == np.int8
    assert (l1[~mask] == OUTSIDE).all()
    assert (l1[1:3, 1:4] == 1).all()                 # low-P edge pixels inherit the object's call
    assert l1[10, 1] == 0
    cm = class_map(objs)
    assert (cm[1:3, 1:4] == CLASSES.index("plastic") + 1).all()
    assert cm[6, 6] == CLASSES.index("metal") + 1 + TENTATIVE_OFFSET
    assert cm[10, 1] == 0 and (cm[~mask] == OUTSIDE).all()
    assert (objs.manmade_pixels() == (mask & (l1 == 1))).all()


def test_an_empty_mask_gives_no_objects():
    mask = np.zeros((5, 5), bool)
    e = np.zeros(0)
    objs = name_objects(mask, e.astype(int), e.astype(int), e, np.zeros((0, 3)), e, CLASSES, 0.55)
    assert len(objs.table["obj_id"]) == 0
    assert (level1_map(objs) == OUTSIDE).all() and not objs.manmade_pixels().any()
