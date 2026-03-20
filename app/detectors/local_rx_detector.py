"""
Local RX (LRX) anomaly detector for hyperspectral data (lrx-had-v1).

Algorithm
---------
For each valid pixel (r, c):
  1. Extract background pixels from the annulus between outer_window and
     inner_window (guard region). Only spatially valid pixels are used.
  2. Compute the local mean μ and covariance Σ from those background pixels.
  3. Score = (x - μ)ᵀ (Σ + λI)⁻¹ (x - μ)   [Mahalanobis distance]

If fewer than min_bg_pixels background pixels are available (e.g., near
swath edges), the pixel is left as NaN.

Band filtering
--------------
Identical two-stage filtering as GlobalRXDetector:
  Stage 1: validity flags + wavelength exclusion via SpectralBandFilter.
  Stage 2: drop bands whose per-pixel failure rate exceeds threshold.

Spatial mask uses a coverage-fraction threshold (default 0.95): a pixel
is valid if at least 95% of surviving bands are valid there.  Missing
bands are filled with the band mean in detect().

Performance
-----------
A `stride` parameter subsamples the spatial grid (default 1 = full
resolution). With stride > 1 the score map is bilinearly interpolated
back to full resolution. For PRISMA scenes (≈1200×1250, 177 bands) a
stride of 2 reduces computation ≈4×.
"""

import logging
from typing import List

import numpy as np
from scipy.ndimage import zoom

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.models.anomaly_detection.lrx_result import LocalRXResult
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter

logger = logging.getLogger(__name__)

DEFAULT_BAND_FAILURE_THRESHOLD = 0.05
DEFAULT_OUTER_WINDOW = 25   
DEFAULT_INNER_WINDOW = 5 
DEFAULT_REGULARIZATION = 1e-4


class LocalRXDetector(AnomalyDetector):
    """
    Local RX detector for hyperspectral cubes.

    Call fit() to prepare band selection, then detect() to score.
    """

    def __init__(self, vendable: VendableDataset):
        super().__init__(vendable)
        self._good_indices: List[int] | None = None
        self._good_wavelengths: List[float] | None = None
        self._spatial_mask: np.ndarray | None = None

    def fit(self, **kwargs) -> None:
        """
        Two-stage band filtering + coverage-fraction spatial mask.

        kwargs:
            band_failure_threshold (float): default 0.05
            exclusion_ranges (list[(float, float)]): wavelength exclusion
                ranges in nm; defaults to standard PRISMA ranges.
            min_band_coverage (float): fraction of surviving bands that must
                be valid at a pixel for it to enter the spatial mask.
                Default 0.95.  Use 1.0 for strict all-bands behaviour.
        """
        threshold        = kwargs.get("band_failure_threshold", DEFAULT_BAND_FAILURE_THRESHOLD)
        min_band_coverage = kwargs.get("min_band_coverage", 0.95)
        exclusion_ranges = kwargs.get("exclusion_ranges", None)
        validity         = self._vendable.validity_cube

        # Stage 1: flags + wavelength
        band_filter = SpectralBandFilter(
            band_wavelengths=self._vendable.band_cw_order,
            band_validity_flags=self._vendable.band_validity_by_position,
            exclusion_ranges=exclusion_ranges,
        )
        stage1 = band_filter.get_good_band_indices()

        # Stage 2: per-pixel failure rate
        any_valid = validity.any(axis=0)
        n_spatial = int(any_valid.sum())
        good = []
        for idx in stage1:
            failure_rate = 1.0 - validity[idx][any_valid].sum() / n_spatial
            if failure_rate <= threshold:
                good.append(idx)
            else:
                logger.info(
                    "Stage 2: dropping band %d (%.1f%% pixel failure)",
                    idx, failure_rate * 100,
                )

        self._good_indices   = good
        self._good_wavelengths = [self._vendable.band_cw_order[i] for i in good]
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

            all_mask = valid_count == n_good
            n_all = int(all_mask.sum())
            n_coverage = int(self._spatial_mask.sum())
            logger.info(
                "Spatial mask (min_band_coverage=%.2f): %d valid pixels "
                "(%d with all bands, %d recovered by relaxing)",
                min_band_coverage, n_coverage, n_all, n_coverage - n_all,
            )

        n_valid = int(self._spatial_mask.sum())
        logger.info(
            "Spatial mask: %d valid pixels / %d in swath | ratio: %.1f",
            n_valid, n_spatial, n_valid / n_good if n_good else 0,
        )

    # ------------------------------------------------------------------
    # detect
    # ------------------------------------------------------------------

    def detect(self, cube: np.ndarray, validity_mask: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Run Local RX.

        Missing bands at spatial-mask pixels are filled with the band mean
        so they contribute zero anomaly signal.  The caller's cube is not
        modified.

        Args:
            cube:           (B, H, W) reflectance cube (destriped recommended).
            validity_mask:  (B, H, W) binary mask. If None, uses the mask from
                            the vendable used at construction.
            outer_window:   Background window half-size in pixels. Default 25.
            inner_window:   Guard window half-size in pixels. Default 5.
            regularization: Ridge term added to Σ diagonal. Default 1e-4.
            min_bg_pixels:  Minimum background pixels required to score a pixel.
                            Default: n_good_bands + 1.
            stride:         Spatial subsampling factor. 1 = full resolution.
                            Default 1.

        Returns:
            (H, W) score map; NaN for pixels without a valid score.
        """
        if self._good_indices is None or self._spatial_mask is None:
            raise RuntimeError("Call fit() before detect().")

        validity = (
            validity_mask if validity_mask is not None
            else self._vendable.validity_cube
        )

        outer   = int(kwargs.get("outer_window",   DEFAULT_OUTER_WINDOW))
        inner   = int(kwargs.get("inner_window",   DEFAULT_INNER_WINDOW))
        reg     = float(kwargs.get("regularization", DEFAULT_REGULARIZATION))
        stride  = int(kwargs.get("stride", 1))

        good = self._good_indices
        mask = self._spatial_mask
        B    = len(good)
        _, H, W = cube.shape

        min_bg = int(kwargs.get("min_bg_pixels", B + 1))

        # Sub-cube: (B_good, H, W) float64 — copy so we can fill in place
        sub = cube[good].astype(np.float64)

        # Band-mean fill: fill missing bands at spatial-mask pixels
        total_filled = 0
        for b_idx, b in enumerate(good):
            band_valid = validity[b].astype(bool)
            needs_fill = mask & (~band_valid)
            n_fill = int(needs_fill.sum())
            if n_fill > 0:
                donor = band_valid & mask
                fill_value = float(sub[b_idx][donor].mean())
                sub[b_idx][needs_fill] = fill_value
                total_filled += n_fill

        if total_filled > 0:
            logger.info("LRX band-mean fill: %d band-pixels filled", total_filled)

        # Spatial validity per band merged into one 2-D mask
        spatial_valid = mask  # (H, W) bool

        score_map = np.full((H, W), np.nan, dtype=np.float64)


        # Row index set for progress logging
        rows_to_process = range(0, H, stride)
        log_every = max(1, len(rows_to_process) // 10)

        logger.info(
            "LRX: outer=%d inner=%d stride=%d min_bg=%d | "
            "scoring %d rows × %d cols",
            outer, inner, stride,
            min_bg,
            len(rows_to_process),
            len(range(0, W, stride)),
        )

        for step_r, r in enumerate(rows_to_process):
            if step_r % log_every == 0:
                logger.info("LRX progress: row %d / %d", r, H)

            # Background row range (clamped to image bounds)
            r0 = max(0, r - outer)
            r1 = min(H, r + outer + 1)

            for c in range(0, W, stride):
                if not spatial_valid[r, c]:
                    continue

                # Background column range
                c0 = max(0, c - outer)
                c1 = min(W, c + outer + 1)

                # Extract window validity and spectra
                win_valid = spatial_valid[r0:r1, c0:c1]        # (wr, wc)
                win_sub   = sub[:, r0:r1, c0:c1]               # (B, wr, wc)

                # Build background mask: valid AND outside guard window
                bg_mask = win_valid.copy()
                # Compute the guard patch position within this window
                gr0 = max(0, r - inner) - r0
                gr1 = min(H, r + inner + 1) - r0
                gc0 = max(0, c - inner) - c0
                gc1 = min(W, c + inner + 1) - c0
                bg_mask[gr0:gr1, gc0:gc1] = False

                n_bg = int(bg_mask.sum())
                if n_bg < min_bg:
                    continue

                # Background spectra: (n_bg, B)
                X_bg = win_sub[:, bg_mask].T   # (n_bg, B)

                mu  = X_bg.mean(axis=0)                        # (B,)
                dX  = X_bg - mu                                 # (n_bg, B)
                cov = (dX.T @ dX) / (n_bg - 1) + reg * np.eye(B)

                # Test pixel
                x   = sub[:, r, c] - mu                        # (B,)

                # Mahalanobis: x @ cov⁻¹ @ x  via solve (numerically stable)
                score_map[r, c] = float(x @ np.linalg.solve(cov, x))

        # If stride > 1, interpolate NaN-free scored pixels to full resolution
        if stride > 1:
            score_map = self._interpolate_strided(score_map, spatial_valid, stride)

        n_scored = int(np.sum(~np.isnan(score_map) & spatial_valid))
        logger.info("LRX: scored %d / %d valid pixels.", n_scored, int(spatial_valid.sum()))

        self._last_result = LocalRXResult(
            lrx_score_map=score_map,
            spatial_mask=mask,
            computed_mask=(~np.isnan(score_map)) & mask,
            good_band_indices=good,
            good_band_wavelengths=self._good_wavelengths,
            n_valid_pixels=int(mask.sum()),
            n_scored_pixels=n_scored,
            n_good_bands=B,
            outer_window=outer,
            inner_window=inner,
        )
        return score_map

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_strided(
        score_map: np.ndarray,
        spatial_valid: np.ndarray,
        stride: int,
    ) -> np.ndarray:
        """
        Bilinear interpolation of a strided score map back to full resolution.

        score_map has non-NaN values only at rows/cols that are multiples of
        stride. Extract that sub-grid, zoom it by stride, then crop to (H, W).
        Pixels outside the valid swath are set to NaN.
        """
        H, W = score_map.shape

        # Extract the strided sub-grid (values were placed at stride-multiples)
        sub = score_map[::stride, ::stride]          # shape ~ (H/s, W/s)
        sub_valid = ~np.isnan(sub)

        filled = sub.copy()
        filled[~sub_valid] = 0.0

        zoomed_scores  = zoom(filled,             stride, order=1)
        zoomed_weights = zoom(sub_valid.astype(float), stride, order=1)

        full = np.full((H, W), np.nan, dtype=np.float64)
        zH, zW = zoomed_scores.shape
        rH, rW = min(zH, H), min(zW, W)

        has_weight = zoomed_weights[:rH, :rW] > 0
        region = full[:rH, :rW]
        region[has_weight] = (
            zoomed_scores[:rH, :rW][has_weight]
            / zoomed_weights[:rH, :rW][has_weight]
        )
        full[:rH, :rW] = region
        full[~spatial_valid] = np.nan

        return full

    @property
    def result(self) -> LocalRXResult:
        if not hasattr(self, "_last_result"):
            raise RuntimeError("Call detect() before accessing result.")
        return self._last_result
