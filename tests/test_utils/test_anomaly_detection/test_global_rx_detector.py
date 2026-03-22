"""
Tests for the GlobalRXDetector (rx-had-v1).
"""

import pytest
import numpy as np

from app.detectors.global_rx_detector import GlobalRXDetector
from app.models.dataset.vendables import VendableHyperspectralDataset
from app.models.anomaly_detection.rx_result import GlobalRXResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_vendable(
    cube: np.ndarray,
    validity: np.ndarray,
    band_validity: list[int],
) -> VendableHyperspectralDataset:
    C = cube.shape[0]
    pixels_per_band = validity.shape[1] * validity.shape[2]
    band_level_validity_score = [
        float(validity[b].sum() / pixels_per_band * 100.0) for b in range(C)
    ]
    return VendableHyperspectralDataset(
        normalized_hyperspectral_cube=cube,
        validity_cube=validity,
        spectral_family_order=["VNIR"] * C,
        band_cw_order=[500.0 + i * 5 for i in range(C)],
        band_fwhm_order=[5.0] * C,
        band_validity_by_position=band_validity,
        band_level_validity_score=band_level_validity_score,
    )


def _make_synthetic_scene(
    n_bands=50, height=100, width=100, n_anomalies=5, seed=42,
):
    """
    Multivariate normal background with injected anomaly pixels.
    Returns vendable, anomaly row coords, anomaly col coords.
    """
    rng = np.random.default_rng(seed)

    # Background: zero-mean, small covariance
    background = rng.normal(loc=0.5, scale=0.02, size=(n_bands, height, width))

    # Pick anomaly pixel locations (avoid edges for simplicity)
    anom_rows = rng.choice(range(10, height - 10), size=n_anomalies, replace=False)
    anom_cols = rng.choice(range(10, width - 10), size=n_anomalies, replace=False)

    # Inject anomalies: shifted mean across all bands
    for r, c in zip(anom_rows, anom_cols):
        background[:, r, c] += 0.5

    cube = background.astype(np.float32)
    validity = np.ones_like(cube, dtype=np.float32)
    band_validity = [1] * n_bands

    vendable = _make_vendable(cube, validity, band_validity)
    return vendable, anom_rows, anom_cols


# ------------------------------------------------------------------
# fit() — band filtering
# ------------------------------------------------------------------


class TestFitBandFiltering:

    def test_stage1_drops_invalid_bands(self):
        C, H, W = 10, 20, 20
        cube = np.random.rand(C, H, W).astype(np.float32)
        validity = np.ones((C, H, W), dtype=np.float32)
        band_validity = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # 7 valid

        vendable = _make_vendable(cube, validity, band_validity)
        detector = GlobalRXDetector(vendable)
        detector.fit()

        assert len(detector._good_indices) == 7
        assert 2 not in detector._good_indices
        assert 4 not in detector._good_indices
        assert 8 not in detector._good_indices

    def test_stage2_drops_noisy_bands(self):
        C, H, W = 5, 100, 100
        cube = np.random.rand(C, H, W).astype(np.float32)
        validity = np.ones((C, H, W), dtype=np.float32)

        # Band 2: 20% of pixels invalid → should be dropped at 5% threshold
        validity[2, :20, :] = 0.0

        band_validity = [1] * C
        vendable = _make_vendable(cube, validity, band_validity)
        detector = GlobalRXDetector(vendable)
        detector.fit(band_failure_threshold=0.05)

        assert 2 not in detector._good_indices
        assert len(detector._good_indices) == 4

    def test_stage2_keeps_bands_under_threshold(self):
        C, H, W = 5, 100, 100
        cube = np.random.rand(C, H, W).astype(np.float32)
        validity = np.ones((C, H, W), dtype=np.float32)

        # Band 2: 3% of pixels invalid → should survive at 5% threshold
        validity[2, :3, :] = 0.0

        band_validity = [1] * C
        vendable = _make_vendable(cube, validity, band_validity)
        detector = GlobalRXDetector(vendable)
        detector.fit(band_failure_threshold=0.05)

        assert 2 in detector._good_indices

    def test_custom_threshold(self):
        C, H, W = 5, 100, 100
        cube = np.random.rand(C, H, W).astype(np.float32)
        validity = np.ones((C, H, W), dtype=np.float32)
        validity[0, :15, :] = 0.0  # 15% failure

        vendable = _make_vendable(cube, validity, [1] * C)
        detector = GlobalRXDetector(vendable)

        # At 10% threshold → band 0 dropped
        detector.fit(band_failure_threshold=0.10)
        assert 0 not in detector._good_indices

        # At 20% threshold → band 0 survives
        detector.fit(band_failure_threshold=0.20)
        assert 0 in detector._good_indices


# ------------------------------------------------------------------
# fit() — spatial mask
# ------------------------------------------------------------------


class TestFitSpatialMask:

    def test_all_valid_gives_full_mask(self):
        C, H, W = 5, 20, 20
        cube = np.random.rand(C, H, W).astype(np.float32)
        validity = np.ones((C, H, W), dtype=np.float32)

        vendable = _make_vendable(cube, validity, [1] * C)
        detector = GlobalRXDetector(vendable)
        detector.fit()

        assert detector._spatial_mask.shape == (H, W)
        assert detector._spatial_mask.all()

    def test_pixel_invalid_in_one_band_gets_masked(self):
        C, H, W = 5, 20, 20
        cube = np.random.rand(C, H, W).astype(np.float32)
        validity = np.ones((C, H, W), dtype=np.float32)
        validity[3, 10, 10] = 0.0  # one band invalid at this pixel

        vendable = _make_vendable(cube, validity, [1] * C)
        detector = GlobalRXDetector(vendable)
        detector.fit()

        assert detector._spatial_mask[10, 10] == False
        assert detector._spatial_mask.sum() == H * W - 1


# ------------------------------------------------------------------
# detect() — guards
# ------------------------------------------------------------------


class TestDetectGuards:

    def test_detect_before_fit_raises(self):
        C, H, W = 5, 20, 20
        cube = np.random.rand(C, H, W).astype(np.float32)
        validity = np.ones((C, H, W), dtype=np.float32)

        vendable = _make_vendable(cube, validity, [1] * C)
        detector = GlobalRXDetector(vendable)

        with pytest.raises(RuntimeError, match="fit"):
            detector.detect(cube, validity)

    def test_non_finite_values_raise(self):
        C, H, W = 5, 20, 20
        cube = np.random.rand(C, H, W).astype(np.float32)
        cube[0, 5, 5] = np.nan
        validity = np.ones((C, H, W), dtype=np.float32)

        vendable = _make_vendable(cube, validity, [1] * C)
        detector = GlobalRXDetector(vendable)
        detector.fit()

        with pytest.raises(ValueError, match="non-finite"):
            detector.detect(cube, validity)


# ------------------------------------------------------------------
# detect() — anomaly ranking (smoke test)
# ------------------------------------------------------------------


class TestDetectAnomalyRanking:

    def test_injected_anomalies_rank_highest(self):
        vendable, anom_rows, anom_cols = _make_synthetic_scene(
            n_bands=50, height=100, width=100, n_anomalies=5, seed=42,
        )

        detector = GlobalRXDetector(vendable)
        detector.fit()
        score_map = detector.detect(
            vendable.normalized_hyperspectral_cube,
            vendable.validity_cube,
        )

        assert score_map.shape == (100, 100)

        # Get the top 5 scoring pixels
        valid_scores = score_map[~np.isnan(score_map)]
        threshold = np.sort(valid_scores)[-5]

        for r, c in zip(anom_rows, anom_cols):
            assert score_map[r, c] >= threshold, (
                f"Anomaly pixel ({r},{c}) score {score_map[r, c]:.4f} "
                f"below top-5 threshold {threshold:.4f}"
            )

    def test_result_property_available_after_detect(self):
        vendable, _, _ = _make_synthetic_scene()

        detector = GlobalRXDetector(vendable)
        detector.fit()
        detector.detect(
            vendable.normalized_hyperspectral_cube,
            vendable.validity_cube,
        )

        result = detector.result
        assert isinstance(result, GlobalRXResult)
        assert result.n_good_bands == 50
        assert result.n_valid_pixels == 100 * 100
        assert len(result.good_band_indices) == 50
        assert len(result.good_band_wavelengths) == 50
