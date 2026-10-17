"""app/utils/evaluation/implant.py: tiers, contrast z by hand, blob shapes, implant arithmetic."""
import numpy as np
import pytest

from app.evaluation.implant import (
    contrast_z, grow_blob, implant_material, implant_signature, ring_mask, solve_strength, tier_of)


def test_tiers():
    assert [tier_of(z) for z in (0.1, 0.502, 0.9, 0.934, 3.0)] == ["subtle", "medium", "medium", "easy", "easy"]


def test_contrast_z_by_hand():
    # The ring is a diamond (4-neighbour dilation, as on the real objects): Manhattan distance 2 or 3 -> 8 + 12 = 20 px.
    # Band 0 is 0.1 on the 8 distance-2 pixels and 0.3 on the 12 distance-3 ones; band 1 is 0.5 everywhere.
    # r = (8 x 0.1 + 12 x 0.3) / 20 = 0.22; sigma0 = sqrt(0.4 x 0.12^2 + 0.6 x 0.08^2) = sqrt(0.0096); sigma1 = 0.
    # Object (0.42, 0.5): |o - r| = (0.2, 0) -> mean 0.1; mean sigma = sqrt(0.0096) / 2 -> z = 0.2 / sqrt(0.0096).
    H = W = 9
    dist = np.abs(np.arange(H)[:, None] - 4) + np.abs(np.arange(W)[None, :] - 4)
    cube = np.zeros((2, H, W)); cube[1] = 0.5; cube[0] = np.where(dist == 2, 0.1, 0.3)
    obj = dist == 0
    cube[:, 4, 4] = (0.42, 0.5)
    ring = ring_mask(obj, np.ones((H, W), bool))
    assert ring.sum() == 20 and np.array_equal(ring, (dist >= 2) & (dist <= 3))
    assert contrast_z(cube, np.ones_like(cube, bool), obj, ring) == pytest.approx(0.2 / np.sqrt(0.0096), rel=1e-4)


def test_grow_blob_compact_connected_and_seeded():
    from scipy import ndimage
    rng = np.random.default_rng(0)
    for size in (1, 2, 5, 12):
        off = grow_blob(rng, size)
        assert len(off) == size and (0, 0) in {tuple(o) for o in off}
        img = np.zeros((40, 40), bool); img[off[:, 0] + 20, off[:, 1] + 20] = True
        assert ndimage.label(img)[1] == 1
    assert np.array_equal(grow_blob(np.random.default_rng(3), 8), grow_blob(np.random.default_rng(3), 8))


def test_implant_arithmetic_and_strength():
    cube = np.full((3, 4, 4), 0.2); obj = np.zeros((4, 4), bool); obj[1, 1] = True
    implant_material(cube, obj, np.array([0.6, np.nan, 0.0]), 0.25)          # (1 - .25) * .2 + .25 * t
    assert np.allclose(cube[:, 1, 1], [0.3, 0.2, 0.15]) and cube[0, 0, 0] == 0.2
    implant_signature(cube, obj, np.array([-1.0, 0.1, np.nan]), 0.5)          # clipped at 0
    assert np.allclose(cube[:, 1, 1], [0.0, 0.25, 0.15])


def test_solve_strength_by_hand():
    sigma, r = np.full(3, 0.05), np.full(3, 0.2)
    step = np.array([0.2, -0.2, np.nan])                                 # the NaN band is ignored
    # o0 = r: z(s) = mean |s * step| / 0.05 = 0.2 s / 0.05 = 4 s -> target 2 at s = 0.5
    s, z = solve_strength(r.copy(), r, sigma, step, 2.0)
    assert s == pytest.approx(0.5, rel=1e-3) and z == pytest.approx(2.0, rel=1e-3)
    # the spot already differs from its ring by 0.05 in every band: z(0) = 1 > 0.5 -> choose another spot
    assert solve_strength(r + 0.05, r, sigma, step, 0.5)[0] is None
    # a material barely different from the pixel cannot reach z 2 with f <= 1: capped at 1, z(1) = 0.01 / 0.05 = 0.2
    s, z = solve_strength(r.copy(), r, sigma, np.array([0.01, 0.01, 0.01]), 2.0, s_max=1.0)
    assert s == 1.0 and z == pytest.approx(0.2, rel=1e-3)
