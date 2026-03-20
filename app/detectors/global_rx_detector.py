"""
Global RX anomaly detector for hyperspectral data (rx-had-v1).

Two-stage band filtering in fit():
    1. Drop bands flagged invalid by the vendable's band_validity_by_position.
    2. Among survivors, drop any band whose per-pixel failure rate in the
       validity cube exceeds a threshold (default 5%).

Spatial mask: coverage-fraction mask.  A pixel is valid if at least
``min_band_coverage`` (default 0.95) of the surviving bands are valid
there.  Missing bands at valid pixels are filled with the band mean
in detect(), contributing zero anomaly signal.
"""

import logging
import time
from typing import List, Dict

import numpy as np
import spectral

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.models.anomaly_detection.rx_result import GlobalRXResult
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter

logger = logging.getLogger(__name__)

# Defines the percentage of pixels in a given band that must be valid for the band itself to be valid.
DEFAULT_BAND_FAILURE_THRESHOLD = 0.05



class GlobalRXDetector(AnomalyDetector):
    """
    Global RX detector for hyperspectral cubes.

    Call fit() with band_failure_threshold before detect().
    """

    def __init__(self, vendable: VendableDataset):
        super().__init__(vendable)
        self._good_indices: List[int] | None = None
        self._good_wavelengths: List[float] | None = None
        self._spatial_mask: np.ndarray | None = None
        self._logs = {}

    @property
    def logs(self) -> Dict:
        return self._logs


    def fit(self, **kwargs) -> None:
        """
        Two-stage band filtering + coverage-fraction spatial mask.

        kwargs:
            band_failure_threshold (float): max per-pixel failure rate for
                a band to survive stage 2. Default 0.05.
            exclusion_ranges (list of (float, float)): wavelength exclusion
                ranges in nm. Default uses standard PRISMA ranges.
            min_band_coverage (float): fraction of surviving bands that must
                be valid at a pixel for it to enter the spatial mask.
                Default 0.95.  Use 1.0 for strict all-bands behaviour.
        """
        # Initialize logs for further discovery
        self._logs = {}
        self._logs["band_wise_failure_ratio"] = {}

        threshold = kwargs.get(
            "band_failure_threshold", DEFAULT_BAND_FAILURE_THRESHOLD
        )
        min_band_coverage = kwargs.get("min_band_coverage", 0.95)
        # We can exclude specific bands if we need to
        exclusion_ranges = kwargs.get("exclusion_ranges", None)
        # Collect the validity cube
        validity = self._vendable.validity_cube

        # Stage 1: drop bands by validity flags + wavelength exclusion
        band_filter = SpectralBandFilter(
            band_wavelengths=self._vendable.band_cw_order,
            band_validity_flags=self._vendable.band_validity_by_position,
            exclusion_ranges=exclusion_ranges,
        )
        stage1 = band_filter.get_good_band_indices()

        # Stage 2: drop bands with high per-pixel failure rate
        # Use only pixels within the valid swath (where at least some band
        # passes) so band failure rate measures data quality, not geometric trim.
        any_valid = validity.any(axis=0)  # (H, W) — True if any band valid
        # Number of pixels with atleast one band valid
        n_spatial = int(any_valid.sum())
        good = []
        # Loop through stage 1 indices
        for idx in stage1:
            # Pick all the pixels from the validity cube where across bands atleast one pixel is valid.
            valid_in_swath = validity[idx][any_valid].sum()
            failure_rate = 1.0 - (valid_in_swath / n_spatial)
            self._logs["band_wise_failure_ratio"][idx] = failure_rate
            if failure_rate <= threshold:
                good.append(idx)
            else:
                logger.info(
                    "Stage 2: dropping band %d (%.1f%% pixel failure)",
                    idx, failure_rate * 100,
                )
        self._good_indices = good
        self._good_wavelengths = [
            self._vendable.band_cw_order[i] for i in good
        ]

        logger.info(
            "Stage 2: %d / %d bands survive (threshold=%.1f%%)",
            len(good), len(stage1), threshold * 100,
        )

        # Spatial mask: coverage-fraction mask
        n_good = len(good)
        if n_good == 0:
            self._spatial_mask = np.zeros(validity.shape[1:], dtype=bool)
        else:
            valid_count = validity[good].sum(axis=0)
            coverage_fraction = valid_count.astype(np.float32) / n_good
            self._spatial_mask = coverage_fraction >= min_band_coverage

            # Diagnostics: compare to strict all-bands mask
            all_mask = valid_count == n_good
            n_all = int(all_mask.sum())
            n_coverage = int(self._spatial_mask.sum())
            n_recovered = n_coverage - n_all
            logger.info(
                "Spatial mask (min_band_coverage=%.2f): %d valid pixels "
                "(%d with all bands, %d recovered by relaxing)",
                min_band_coverage, n_coverage, n_all, n_recovered,
            )

        n_valid = int(self._spatial_mask.sum())
        ratio = n_valid / n_good if n_good > 0 else 0

        logger.info(
            "Spatial mask: %d valid pixels / %d in swath | "
            "Pixel-to-band ratio: %.1f",
            n_valid, n_spatial, ratio,
        )

        if ratio < 10:
            logger.warning(
                "Only %.1f pixels per band — covariance may be "
                "ill-conditioned. Got %d pixels, %d bands.",
                ratio, n_valid, n_good,
            )

    def detect(self, cube, validity_mask=None):
        """
        Run Global RX on the cube using bands and mask computed in fit().

        Missing bands at spatial-mask pixels are filled with the band mean
        so they contribute zero anomaly signal.  The caller's cube is not
        modified — filling is done on an internal copy.

        Returns (H, W) score map with NaN for invalid pixels.
        """
        if self._good_indices is None or self._spatial_mask is None:
            raise RuntimeError("Call fit() before detect().")

        validity = (
            validity_mask if validity_mask is not None
            else self._vendable.validity_cube
        )

        _, H, W = cube.shape
        good = self._good_indices
        mask = self._spatial_mask
        n_valid = int(mask.sum())

        t_total = time.time()

        # Work on a copy so the caller's cube is untouched
        working = cube[good].copy()  # (B_good, H, W)
        logger.info(
            "GRX: %d bands x (%d, %d) | %d valid pixels",
            len(good), H, W, n_valid,
        )

        # Band-mean fill: for each good band, fill pixels that are in the
        # spatial mask but missing that specific band
        t_fill = time.time()
        total_filled = 0
        max_filled_per_pixel = 0
        if n_valid > 0:
            fill_count = np.zeros((H, W), dtype=np.int32)
            for b_idx, b in enumerate(good):
                band_valid = validity[b].astype(bool)
                needs_fill = mask & (~band_valid)
                n_fill = int(needs_fill.sum())
                if n_fill > 0:
                    # Mean from pixels where this band IS valid within the mask
                    donor = band_valid & mask
                    fill_value = float(working[b_idx][donor].mean())
                    working[b_idx][needs_fill] = fill_value
                    fill_count[needs_fill] += 1
                    total_filled += n_fill

            max_filled_per_pixel = int(fill_count[mask].max()) if n_valid > 0 else 0
            n_pixels_with_fill = int((fill_count[mask] > 0).sum())
            logger.info(
                "GRX band-mean fill: %d band-pixels filled across %d pixels "
                "(max %d bands filled at one pixel) in %.2fs",
                total_filled, n_pixels_with_fill, max_filled_per_pixel,
                time.time() - t_fill,
            )

        # Validate: all valid pixel values must be finite
        valid_values = working[:, mask]  # (B_good, N_valid)
        if not np.all(np.isfinite(valid_values)):
            non_finite = int((~np.isfinite(valid_values)).sum())
            raise ValueError(
                f"{non_finite} non-finite values in valid pixels after fill. "
                "Check the reflectance transformer upstream."
            )

        # Transpose to (N_valid, B_good) — each row is one pixel's spectrum
        pixels = valid_values.T
        logger.info("GRX: computing RX scores for %d pixels x %d bands…", pixels.shape[0], pixels.shape[1])

        # spy.rx expects (rows, cols, bands) → (N_valid, 1, B_good)
        t_rx = time.time()
        rx_scores_flat = spectral.rx(
            np.asarray(pixels[:, np.newaxis, :])
        ).ravel()
        logger.info("GRX: spectral.rx done in %.2fs", time.time() - t_rx)

        # Map scores back to (H, W), NaN where invalid
        score_map = np.full((H, W), np.nan, dtype=np.float64)
        score_map[mask] = rx_scores_flat

        valid_scores = rx_scores_flat[np.isfinite(rx_scores_flat)]
        if len(valid_scores) > 0:
            logger.info(
                "GRX: score range [%.4f, %.4f] median=%.4f p99=%.4f",
                float(valid_scores.min()), float(valid_scores.max()),
                float(np.median(valid_scores)),
                float(np.percentile(valid_scores, 99)),
            )
        logger.info("GRX: total %.2fs", time.time() - t_total)

        self._last_result = GlobalRXResult(
            rx_score_map=score_map,
            spatial_mask=mask,
            good_band_indices=good,
            good_band_wavelengths=self._good_wavelengths,
            n_valid_pixels=n_valid,
            n_good_bands=len(good),
        )

        return score_map

    @property
    def result(self) -> GlobalRXResult:
        """Full result with metadata. Available after detect()."""
        if not hasattr(self, "_last_result"):
            raise RuntimeError("Call detect() before accessing result.")
        return self._last_result
