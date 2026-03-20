"""
Tests for the InferenceHarness.

Uses synthetic VendableDatasets and a trivial stub detector to verify
full-scene inference, patch-level inference, overlap averaging, and
fit delegation.
"""

import pytest
import numpy as np

from app.abstract_classes.anomaly_detector import AnomalyDetector
from app.models.anomaly_detection.harness_config import (
    InferenceHarnessConfig,
    PatchConfig,
)
from app.models.anomaly_detection.detection_result import AnomalyDetectionResult
from app.models.dataset.vendables import (
    VendableHyperspectralDataset,
    VendableThermalDataset,
)
from app.utils.anomaly_detection.inference_harness import InferenceHarness


# ------------------------------------------------------------------
# Stub detector — returns the per-pixel spectral mean as the "score"
# ------------------------------------------------------------------


class MeanScoreDetector(AnomalyDetector):
    """Returns cube.mean(axis=0) as the anomaly score map."""

    def __init__(self, vendable):
        super().__init__(vendable)
        self.fit_called = False

    def fit(self, **kwargs):
        self.fit_called = True

    def detect(self, cube, validity_mask=None):
        return cube.mean(axis=0)


# ------------------------------------------------------------------
# Fixtures — synthetic vendables
# ------------------------------------------------------------------


@pytest.fixture
def hyperspectral_vendable() -> VendableHyperspectralDataset:
    """10-band, 100x100 hyperspectral vendable with all-valid mask."""
    C, H, W = 10, 100, 100
    cube = np.random.rand(C, H, W).astype(np.float32)
    validity = np.ones((C, H, W), dtype=np.float32)
    return VendableHyperspectralDataset(
        normalized_hyperspectral_cube=cube,
        validity_cube=validity,
        spectral_family_order=["VNIR"] * C,
        band_cw_order=[400.0 + i * 10 for i in range(C)],
        band_fwhm_order=[5.0] * C,
        band_validity_by_position=[1] * C,
    )


@pytest.fixture
def thermal_vendable() -> VendableThermalDataset:
    """Single-band 100x100 thermal vendable."""
    C, H, W = 1, 100, 100
    cube = np.random.rand(C, H, W).astype(np.float32) * 40 + 260
    validity = np.ones((C, H, W), dtype=np.float32)
    return VendableThermalDataset(
        normalized_thermal_cube=cube,
        validity_cube=validity,
        cloud_mask=np.ones((H, W), dtype=np.float32),
        pure_validity_mask=np.ones((H, W), dtype=np.float32),
    )


# ------------------------------------------------------------------
# Full-scene inference
# ------------------------------------------------------------------


def test_full_scene_returns_correct_shape(hyperspectral_vendable):
    harness = InferenceHarness(config=InferenceHarnessConfig())
    detector = MeanScoreDetector(hyperspectral_vendable)
    result = harness.run(hyperspectral_vendable, detector)

    assert isinstance(result, AnomalyDetectionResult)
    assert result.score_map.shape == (100, 100)
    assert result.patches_processed == 1
    assert result.source_cube_shape == (10, 100, 100)


def test_full_scene_on_thermal(thermal_vendable):
    harness = InferenceHarness(config=InferenceHarnessConfig())
    detector = MeanScoreDetector(thermal_vendable)
    result = harness.run(thermal_vendable, detector)

    assert result.score_map.shape == (100, 100)
    assert result.patches_processed == 1
    assert result.source_cube_shape == (1, 100, 100)


def test_full_scene_scores_match_detector_output(hyperspectral_vendable):
    harness = InferenceHarness(config=InferenceHarnessConfig())
    detector = MeanScoreDetector(hyperspectral_vendable)
    result = harness.run(hyperspectral_vendable, detector)

    expected = hyperspectral_vendable.normalized_hyperspectral_cube.mean(axis=0)
    np.testing.assert_allclose(result.score_map, expected, atol=1e-6)


# ------------------------------------------------------------------
# Fit delegation
# ------------------------------------------------------------------


def test_fit_called_when_enabled(hyperspectral_vendable):
    config = InferenceHarnessConfig(fit_on_full_scene=True)
    harness = InferenceHarness(config=config)
    detector = MeanScoreDetector(hyperspectral_vendable)

    harness.run(hyperspectral_vendable, detector)
    assert detector.fit_called is True


def test_fit_skipped_when_disabled(hyperspectral_vendable):
    config = InferenceHarnessConfig(fit_on_full_scene=False)
    harness = InferenceHarness(config=config)
    detector = MeanScoreDetector(hyperspectral_vendable)

    harness.run(hyperspectral_vendable, detector)
    assert detector.fit_called is False


# ------------------------------------------------------------------
# Patched inference — exact tiling (no overlap)
# ------------------------------------------------------------------


def test_patched_exact_tiling(hyperspectral_vendable):
    pc = PatchConfig(patch_height=50, patch_width=50, stride=50)
    config = InferenceHarnessConfig(patch_config=pc)
    harness = InferenceHarness(config=config)
    detector = MeanScoreDetector(hyperspectral_vendable)

    result = harness.run(hyperspectral_vendable, detector)
    assert result.score_map.shape == (100, 100)
    assert result.patches_processed == 4


def test_patched_exact_tiling_scores_match_full_scene(hyperspectral_vendable):
    pc = PatchConfig(patch_height=50, patch_width=50, stride=50)
    harness_patched = InferenceHarness(
        config=InferenceHarnessConfig(patch_config=pc, fit_on_full_scene=False)
    )
    harness_full = InferenceHarness(
        config=InferenceHarnessConfig(fit_on_full_scene=False)
    )
    detector = MeanScoreDetector(hyperspectral_vendable)

    result_patched = harness_patched.run(hyperspectral_vendable, detector)
    result_full = harness_full.run(hyperspectral_vendable, detector)

    np.testing.assert_allclose(
        result_patched.score_map, result_full.score_map, atol=1e-6
    )


# ------------------------------------------------------------------
# Patched inference — overlapping patches
# ------------------------------------------------------------------


def test_patched_overlapping_produces_correct_shape(hyperspectral_vendable):
    pc = PatchConfig(patch_height=50, patch_width=50, stride=25)
    config = InferenceHarnessConfig(patch_config=pc)
    harness = InferenceHarness(config=config)
    detector = MeanScoreDetector(hyperspectral_vendable)

    result = harness.run(hyperspectral_vendable, detector)
    assert result.score_map.shape == (100, 100)
    assert result.patches_processed > 4


def test_overlapping_patches_average_correctly():
    C, H, W = 5, 80, 80
    constant_val = 7.0
    cube = np.full((C, H, W), constant_val, dtype=np.float32)
    validity = np.ones((C, H, W), dtype=np.float32)

    vendable = VendableHyperspectralDataset(
        normalized_hyperspectral_cube=cube,
        validity_cube=validity,
        spectral_family_order=["VNIR"] * C,
        band_cw_order=[400.0 + i * 10 for i in range(C)],
        band_fwhm_order=[5.0] * C,
        band_validity_by_position=[1] * C,
    )

    pc = PatchConfig(patch_height=40, patch_width=40, stride=20)
    config = InferenceHarnessConfig(patch_config=pc, fit_on_full_scene=False)
    harness = InferenceHarness(config=config)
    detector = MeanScoreDetector(vendable)

    result = harness.run(vendable, detector)
    np.testing.assert_allclose(result.score_map, constant_val, atol=1e-6)


# ------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------


def test_batch_size_does_not_affect_result(hyperspectral_vendable):
    pc = PatchConfig(patch_height=50, patch_width=50, stride=25)
    detector = MeanScoreDetector(hyperspectral_vendable)

    result_bs1 = InferenceHarness(
        config=InferenceHarnessConfig(
            patch_config=pc, batch_size=1, fit_on_full_scene=False
        )
    ).run(hyperspectral_vendable, detector)

    result_bs3 = InferenceHarness(
        config=InferenceHarnessConfig(
            patch_config=pc, batch_size=3, fit_on_full_scene=False
        )
    ).run(hyperspectral_vendable, detector)

    np.testing.assert_allclose(
        result_bs1.score_map, result_bs3.score_map, atol=1e-6
    )
    assert result_bs1.patches_processed == result_bs3.patches_processed


# ------------------------------------------------------------------
# Unsupported vendable type
# ------------------------------------------------------------------


def test_rejects_unsupported_vendable_type(hyperspectral_vendable):
    harness = InferenceHarness(config=InferenceHarnessConfig())
    detector = MeanScoreDetector(hyperspectral_vendable)

    with pytest.raises(TypeError, match="Unsupported vendable type"):
        harness.run("not a vendable", detector)


# ------------------------------------------------------------------
# Non-divisible scene dimensions
# ------------------------------------------------------------------


def test_patched_non_divisible_dimensions():
    C, H, W = 3, 73, 91
    cube = np.random.rand(C, H, W).astype(np.float32)
    validity = np.ones((C, H, W), dtype=np.float32)

    vendable = VendableHyperspectralDataset(
        normalized_hyperspectral_cube=cube,
        validity_cube=validity,
        spectral_family_order=["VNIR"] * C,
        band_cw_order=[400.0 + i * 10 for i in range(C)],
        band_fwhm_order=[5.0] * C,
        band_validity_by_position=[1] * C,
    )

    pc = PatchConfig(patch_height=32, patch_width=32, stride=32)
    config = InferenceHarnessConfig(patch_config=pc, fit_on_full_scene=False)
    harness = InferenceHarness(config=config)
    detector = MeanScoreDetector(vendable)

    result = harness.run(vendable, detector)
    assert result.score_map.shape == (H, W)
    assert result.patches_processed > 0
