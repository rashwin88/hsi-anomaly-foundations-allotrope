"""Residual Mahalanobis scoring (app/utils/anomaly_detection/residual_mahalanobis.py)."""

import numpy as np
import pytest

from app.utils.anomaly_detection.residual_mahalanobis import residual_mahalanobis
from app.utils.anomaly_detection.scoring import compute_score

C, H, W = 12, 80, 90


@pytest.fixture
def scene():
    """Background residuals share one strong direction (a brightness-like error, large) plus small
    independent noise. Pixel A has a big error along the shared direction; pixel B a small error
    along a direction the background never takes."""
    rng = np.random.default_rng(0)
    shared = np.ones(C) / np.sqrt(C)
    rare = np.zeros(C); rare[3], rare[7] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    x = np.full((C, H, W), 0.2, np.float32)
    resid = (rng.normal(0, 0.05, (H, W))[None] * shared[:, None, None]
             + rng.normal(0, 0.002, (C, H, W))).astype(np.float32)
    resid[:, 40, 30] += 0.15 * shared          # A: 3 sigma along the common direction
    resid[:, 40, 60] += 0.02 * rare            # B: small, but ~10 sigma in a rare direction
    return x, x - resid


def test_rare_direction_outranks_large_common_error(scene):
    x, r = scene
    keep = np.ones((H, W), bool)
    l1 = compute_score(x, r, keep, method="L1")
    for window in ((21, 61), None):
        s = residual_mahalanobis(x, r, keep, local_window=window)
        assert s[40, 60] > s[40, 30], window                  # B above A
        assert s[40, 60] > np.percentile(s, 99.9), window
    assert l1[40, 30] > l1[40, 60]                            # plain L1 ranks them the other way


def test_compute_score_defaults_to_the_local_window_and_masks_invalid(scene):
    x, r = scene
    keep = np.ones((H, W), bool); keep[:6] = False
    s = compute_score(x, r, keep, method="residual_mahalanobis")
    assert s.dtype == np.float32 and s.shape == (H, W)
    assert np.array_equal(s, residual_mahalanobis(x, r, keep, local_window=(21, 61)))
    assert (s[:6] == 0).all() and (s[keep] >= 0).all()
    g = compute_score(x, r, keep, method="residual_mahalanobis", local_window=None)
    assert not np.array_equal(s, g)


def test_deterministic(scene):
    x, r = scene
    keep = np.ones((H, W), bool)
    assert np.array_equal(residual_mahalanobis(x, r, keep), residual_mahalanobis(x, r, keep))


def test_existing_methods_unchanged_by_the_new_argument(scene):
    x, r = scene
    keep = np.ones((H, W), bool)
    for m in ("L1", "MSE", "SAM", "combined"):
        assert np.array_equal(compute_score(x, r, keep, method=m), compute_score(x, r, keep, method=m, local_window=None))
