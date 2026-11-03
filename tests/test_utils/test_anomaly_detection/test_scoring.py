"""
Tests for compute_score's opt-in SAM dark fade (sam_dark_fade).
"""

import numpy as np
import pytest

from app.utils.anomaly_detection.scoring import compute_score

_PATTERN = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)


def _cube_pair(bright_err: float = 0.03):
    """(C=4, H=1, W=3) original and reconstruction.

    Pixel 0: dark water (1% mean reflectance), 0.004 absolute error -> large angle (~0.35 rad).
    Pixel 1: bright land (30%), `bright_err` absolute error -> moderate angle.
    Pixel 2: exactly at the 5% threshold, 0.01 absolute error.
    """
    orig = np.stack([
        0.01 * np.array([1.5, 1.0, 0.75, 0.75]),
        np.array([0.30, 0.32, 0.28, 0.30]),
        np.full(4, 0.05),
    ], axis=1)[:, None, :].astype(np.float32)
    recon = orig.copy()
    for px, err in enumerate((0.004, bright_err, 0.01)):
        recon[:, 0, px] += err * _PATTERN
    return orig, recon, np.ones((1, 3), dtype=np.uint8)


@pytest.mark.parametrize("method", ["L1", "MSE", "SAM", "combined"])
def test_fade_none_is_todays_output(method):
    orig, recon, valid = _cube_pair()
    default = compute_score(orig, recon, valid, method)
    explicit = compute_score(orig, recon, valid, method, sam_dark_fade=None)
    assert default.dtype == explicit.dtype == np.float32
    assert default.tobytes() == explicit.tobytes()


# combined also weighs L1, where the bright pixel's larger error already wins; a smaller bright
# error keeps the dark-water trap visible there.
@pytest.mark.parametrize("method, bright_err", [("SAM", 0.03), ("combined", 0.01)])
def test_fade_stops_dark_pixel_outranking_bright(method, bright_err):
    orig, recon, valid = _cube_pair(bright_err)
    plain = compute_score(orig, recon, valid, method)[0]
    faded = compute_score(orig, recon, valid, method, sam_dark_fade=0.05)[0]
    assert plain[0] > plain[1]          # the trap: dark water outranks the bright pixel
    assert faded[0] < faded[1]


def test_rm_components_sets_the_pca_width():
    """v1.0.1: rm_components reaches residual_mahalanobis as d; the default is byte-identical to d=30."""
    from app.utils.anomaly_detection.residual_mahalanobis import residual_mahalanobis

    rng = np.random.default_rng(7)
    orig = rng.random((40, 24, 24)).astype(np.float32)
    recon = (orig + 0.05 * rng.standard_normal(orig.shape)).astype(np.float32)
    valid = np.ones((24, 24), np.uint8)
    default = compute_score(orig, recon, valid, "residual_mahalanobis")
    d30 = compute_score(orig, recon, valid, "residual_mahalanobis", rm_components=30)
    d20 = compute_score(orig, recon, valid, "residual_mahalanobis", rm_components=20)
    assert default.tobytes() == d30.tobytes()
    assert not np.allclose(d20, d30)
    np.testing.assert_array_equal(d20, residual_mahalanobis(orig, recon, valid.astype(bool), d=20))


def test_fade_leaves_pixels_at_or_above_threshold_unchanged():
    orig, recon, valid = _cube_pair()
    plain = compute_score(orig, recon, valid, "SAM")[0]
    faded = compute_score(orig, recon, valid, "SAM", sam_dark_fade=0.05)[0]
    assert orig[:, 0, 2].mean() == pytest.approx(0.05)
    np.testing.assert_allclose(faded[1:], plain[1:], rtol=1e-6)
    assert faded[0] == pytest.approx(plain[0] * 0.2, rel=1e-5)   # 1% / 5% = 0.2
