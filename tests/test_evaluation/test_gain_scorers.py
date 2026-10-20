"""The scoring-gains scorers (research/experiments/scoring-gains/): spatial blur, rank fusion, background mixture."""
import numpy as np
import pytest

from app.evaluation.config import ScorerSpec
from app.evaluation.scorers import background_types, blur, rank_map, score

H = W = 96


def scene(seed=0, C=12):
    rng = np.random.default_rng(seed)
    cube = rng.normal(0.3, 0.02, (C, H, W)).astype(np.float32)
    return cube, np.zeros_like(cube), np.ones((H, W), bool)


def blob(cube, amp, sigma=1.4, at=(48, 48)):
    """A PSF-blurred target: a fixed spectral direction, amplitude falling off as a Gaussian of the distance."""
    yy, xx = np.mgrid[:H, :W]
    w = np.exp(-((yy - at[0]) ** 2 + (xx - at[1]) ** 2) / (2 * sigma ** 2))
    direction = np.linspace(-1, 1, len(cube))[:, None, None]
    return cube + amp * direction * w, w > 0.3


def test_blur_keeps_nodata_out_and_constants_constant():
    x = np.full((2, H, W), 5.0, np.float32)
    valid = np.ones((H, W), bool)
    valid[:, :10] = False
    x[:, ~valid] = 1e6                       # nodata garbage must not bleed in
    out = blur(x, valid, 2.0)
    assert np.allclose(out[:, valid], 5.0, atol=1e-4) and (out[:, ~valid] == 0).all()


def test_spatial_blur_lifts_a_faint_blurred_target():
    cube, recon, valid = scene()
    cube, target = blob(cube, amp=0.03)
    plain = score(ScorerSpec(kind="residual_mahalanobis", d=10), cube, recon, valid, None)
    blurred = score(ScorerSpec(kind="residual_mahalanobis", d=10, spatial_sigma=1.4), cube, recon, valid, None)
    rank = lambda s: (rank_map(s, valid)[target]).mean()
    assert rank(blurred) > rank(plain)


def test_rank_map_by_hand():
    s = np.array([[3.0, 1.0], [2.0, 2.0]])
    valid = np.array([[True, True], [True, False]])
    # valid scores 3, 1, 2 -> ranks 3, 1, 2 out of 3; the invalid pixel is 0
    assert np.allclose(rank_map(s, valid), [[1.0, 1 / 3], [2 / 3, 0.0]])


def test_fuse_rules():
    cube, recon, valid = scene()
    parts = [ScorerSpec(kind="residual_mahalanobis", d=5), ScorerSpec(kind="pca_rx", d=5)]
    mean = score(ScorerSpec(kind="fuse", parts=parts), cube, recon, valid, None)
    hi = score(ScorerSpec(kind="fuse", parts=parts, rule="rank_max"), cube, recon, valid, None)
    assert (hi >= mean - 1e-6).all() and mean.max() <= 1.0
    assert ScorerSpec(kind="fuse", parts=parts).needs_model
    assert not ScorerSpec(kind="fuse", parts=[parts[1], parts[1]]).needs_model
    with pytest.raises(ValueError):
        ScorerSpec(kind="fuse", parts=parts[:1])


def test_background_types_split_two_surfaces():
    cube, recon, valid = scene()
    cube[:, :, W // 2:] += np.linspace(0, 0.5, len(cube))[:, None, None]     # a second surface on the right half
    t = background_types(cube, valid, k=2)
    left, right = t[:, : W // 2], t[:, W // 2:]
    assert len(np.unique(left)) == 1 and len(np.unique(right)) == 1 and left[0, 0] != right[0, 0]


def test_mixture_runs_and_labels():
    cube, recon, valid = scene()
    spec = ScorerSpec(kind="mixture", k=3, d=5)
    s = score(spec, cube, recon, valid, None)
    assert s.shape == (H, W) and np.isfinite(s).all() and spec.label.startswith("mixture(k=3")
