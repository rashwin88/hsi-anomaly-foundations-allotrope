"""
Tests for the AnomalyDetector abstract base class.
"""

import pytest
import numpy as np

from app.abstract_classes.anomaly_detector import AnomalyDetector
from app.models.dataset.vendables import VendableThermalDataset


@pytest.fixture
def stub_vendable() -> VendableThermalDataset:
    """Minimal vendable for constructing test detectors."""
    return VendableThermalDataset(
        normalized_thermal_cube=np.zeros((1, 4, 4), dtype=np.float32),
        validity_cube=np.ones((1, 4, 4), dtype=np.float32),
        cloud_mask=None,
        pure_validity_mask=None,
    )


class MeanDeviationDetector(AnomalyDetector):
    """Scores each pixel by L2 distance from the spectral mean."""

    def detect(self, cube, validity_mask=None):
        mean_spectrum = cube.mean(axis=0, keepdims=True)
        return np.sqrt(((cube - mean_spectrum) ** 2).sum(axis=0))


class FittableDetector(AnomalyDetector):
    """Computes background stats in fit(), uses them in detect()."""

    def __init__(self, vendable):
        super().__init__(vendable)
        self.global_mean = None
        self.fitted = False

    def fit(self, **kwargs):
        cube = kwargs["cube"]
        self.global_mean = cube.mean(axis=(1, 2), keepdims=True)
        self.fitted = True

    def detect(self, cube, validity_mask=None):
        if self.global_mean is None:
            raise RuntimeError("Must call fit() before detect()")
        deviation = cube - self.global_mean
        return np.sqrt((deviation ** 2).sum(axis=0))


# ------------------------------------------------------------------
# ABC contract
# ------------------------------------------------------------------


def test_cannot_instantiate_abc_directly(stub_vendable):
    with pytest.raises(TypeError):
        AnomalyDetector(stub_vendable)


def test_concrete_detector_instantiates(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    assert isinstance(detector, AnomalyDetector)


def test_detector_holds_vendable_reference(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    assert detector._vendable is stub_vendable


# ------------------------------------------------------------------
# detect
# ------------------------------------------------------------------


def test_detect_returns_correct_shape(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    cube = np.random.rand(10, 64, 64).astype(np.float32)
    scores = detector.detect(cube)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (64, 64)


def test_detect_with_single_band(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    cube = np.random.rand(1, 32, 32).astype(np.float32)
    scores = detector.detect(cube)

    assert scores.shape == (32, 32)
    np.testing.assert_allclose(scores, 0.0, atol=1e-6)


def test_detect_with_validity_mask_passed_through(stub_vendable):
    class MaskAwareDetector(AnomalyDetector):
        def detect(self, cube, validity_mask=None):
            if validity_mask is not None:
                cube = cube * validity_mask
            return cube.mean(axis=0)

    detector = MaskAwareDetector(stub_vendable)
    cube = np.ones((3, 4, 4), dtype=np.float32)
    mask = np.zeros((3, 4, 4), dtype=np.float32)
    mask[:, :2, :2] = 1.0

    scores = detector.detect(cube, validity_mask=mask)
    assert scores[0, 0] == 1.0
    assert scores[3, 3] == 0.0


# ------------------------------------------------------------------
# detect_batch (default loop implementation)
# ------------------------------------------------------------------


def test_detect_batch_iterates_over_batch_dim(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    batch = np.random.rand(5, 10, 32, 32).astype(np.float32)
    scores = detector.detect_batch(batch)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (5, 32, 32)


def test_detect_batch_matches_individual_calls(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    batch = np.random.rand(3, 8, 16, 16).astype(np.float32)

    batch_scores = detector.detect_batch(batch)
    individual_scores = np.stack(
        [detector.detect(batch[i]) for i in range(3)]
    )

    np.testing.assert_array_equal(batch_scores, individual_scores)


def test_detect_batch_with_validity_masks(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    batch = np.random.rand(4, 6, 20, 20).astype(np.float32)
    masks = np.ones((4, 6, 20, 20), dtype=np.float32)

    scores = detector.detect_batch(batch, validity_masks=masks)
    assert scores.shape == (4, 20, 20)


# ------------------------------------------------------------------
# fit (default no-op)
# ------------------------------------------------------------------


def test_default_fit_is_noop(stub_vendable):
    detector = MeanDeviationDetector(stub_vendable)
    # Should not raise
    detector.fit()
    detector.fit(some_kwarg="value")


def test_fittable_detector_requires_fit_before_detect(stub_vendable):
    detector = FittableDetector(stub_vendable)
    cube = np.random.rand(5, 32, 32).astype(np.float32)

    with pytest.raises(RuntimeError):
        detector.detect(cube)


def test_fittable_detector_works_after_fit(stub_vendable):
    detector = FittableDetector(stub_vendable)
    cube = np.random.rand(5, 32, 32).astype(np.float32)

    detector.fit(cube=cube)
    assert detector.fitted is True

    scores = detector.detect(cube)
    assert scores.shape == (32, 32)


# ------------------------------------------------------------------
# Masked array support
# ------------------------------------------------------------------


def test_detect_accepts_masked_array(stub_vendable):
    class MaskedArrayDetector(AnomalyDetector):
        def detect(self, cube, validity_mask=None):
            if isinstance(cube, np.ma.MaskedArray):
                return np.ma.filled(cube.mean(axis=0), 0.0)
            return cube.mean(axis=0)

    detector = MaskedArrayDetector(stub_vendable)
    data = np.random.rand(3, 10, 10).astype(np.float32)
    mask = np.zeros_like(data, dtype=bool)
    mask[0, :5, :5] = True
    cube = np.ma.MaskedArray(data, mask=mask)

    scores = detector.detect(cube)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10, 10)
