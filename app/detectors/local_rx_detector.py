"""
Local RX (LRX) anomaly detector for hyperspectral data (lrx-had-v1).

Algorithm
---------
For each valid pixel (r, c):
  1. Extract background pixels from the annulus between outer_window and
     inner_window (guard region). Only spatially valid pixels are used.
  2. Compute the local mean Î¼ and covariance Î£ from those background pixels.
  3. Score = (x - Î¼)áµ€ (Î£ + Î»I)â»Â¹ (x - Î¼)   [Mahalanobis distance]

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
back to full resolution. For PRISMA scenes (â‰ˆ1200Ã—1250, 177 bands) a
stride of 2 reduces computation â‰ˆ4Ã—.

Covariance and Mahalanobis distance computations are batched via
torch.linalg.solve, automatically selecting the best available device
(CUDA > MPS > CPU).
"""

import logging
import time
from typing import List

import numpy as np
import torch
from scipy.ndimage import zoom

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.detectors._local_background import batch_mahalanobis, select_device
from app.models.anomaly_detection.lrx_result import LocalRXResult
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter

logger = logging.getLogger(__name__)

DEFAULT_BAND_FAILURE_THRESHOLD = 0.05
DEFAULT_OUTER_WINDOW = 25
DEFAULT_INNER_WINDOW = 5
DEFAULT_REGULARIZATION = 1e-4
DEFAULT_BATCH_SIZE = 256



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

    def detect(self, cube: np.ndarray, validity_mask: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Run Local RX with batched torch.linalg.solve on the best available device.

        Missing bands at spatial-mask pixels are filled with the band mean
        so they contribute zero anomaly signal.  The caller's cube is not
        modified.

        Args:
            cube:           (B, H, W) reflectance cube (destriped recommended).
            validity_mask:  (B, H, W) binary mask. If None, uses the mask from
                            the vendable used at construction.
            outer_window:   Background window half-size in pixels. Default 25.
            inner_window:   Guard window half-size in pixels. Default 5.
            regularization: Ridge term added to Î£ diagonal. Default 1e-4.
            min_bg_pixels:  Minimum background pixels required to score a pixel.
                            Default: n_good_bands + 1.
            stride:         Spatial subsampling factor. 1 = full resolution.
                            Default 1.
            batch_size:     Number of pixels per GPU batch. Default 256.

        Returns:
            (H, W) score map; NaN for pixels without a valid score.
        """
        t_total = time.time()

        if self._good_indices is None or self._spatial_mask is None:
            raise RuntimeError("Call fit() before detect().")

        validity = (
            validity_mask if validity_mask is not None
            else self._vendable.validity_cube
        )

        outer      = int(kwargs.get("outer_window",   DEFAULT_OUTER_WINDOW))
        inner      = int(kwargs.get("inner_window",   DEFAULT_INNER_WINDOW))
        reg        = float(kwargs.get("regularization", DEFAULT_REGULARIZATION))
        stride     = int(kwargs.get("stride", 1))

        good = self._good_indices
        mask = self._spatial_mask
        B    = len(good)
        _, H, W = cube.shape

        min_bg = int(kwargs.get("min_bg_pixels", B + 1))

        # Device selection + auto batch size
        device = select_device()
        compute_dtype = "float32" if device.type != "cpu" else "float64"

        if "batch_size" in kwargs:
            batch_size = int(kwargs["batch_size"])
        elif device.type == "cuda":
            # Large batches to keep the GPU busy; each element uses
            # ~2 * max_bg * B * 4 bytes (X_bg + dX) + B^2 * 4 (cov).
            max_bg = (2 * outer + 1) ** 2 - (2 * inner + 1) ** 2
            bytes_per_elem = (2 * max_bg * B + B * B) * 4
            vram = torch.cuda.get_device_properties(device).total_memory
            # Use up to 25% of VRAM for the batch
            batch_size = max(256, int(0.25 * vram / bytes_per_elem))
        else:
            batch_size = DEFAULT_BATCH_SIZE

        logger.info(
            "LRX: device=%s | compute_dtype=%s | batch_size=%d",
            device, compute_dtype, batch_size,
        )
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            logger.info("LRX: GPU=%s (%.1f GB)", gpu_name, gpu_mem)

        # Sub-cube: (B_good, H, W) float64 â€” copy so we can fill in place
        t_prep = time.time()
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

        spatial_valid = mask   # (H, W) bool
        score_map = np.full((H, W), np.nan, dtype=np.float64)
        t_prep = time.time() - t_prep
        logger.info("LRX: sub-cube prep done in %.2fs", t_prep)

        # Theoretical max background pixels in the annulus
        max_bg = (2 * outer + 1) ** 2 - (2 * inner + 1) ** 2

        # Pre-allocate batch buffers (reused every flush)
        batch_X_bg = np.zeros((batch_size, max_bg, B), dtype=np.float64)
        batch_n_bg = np.zeros(batch_size, dtype=np.int64)
        batch_x_test = np.zeros((batch_size, B), dtype=np.float64)
        batch_coords: list[tuple[int, int]] = []
        batch_idx = 0

        n_scored = 0
        n_skipped_bg = 0
        n_skipped_invalid = 0
        t_extract = 0.0
        t_solve = 0.0
        n_batches_flushed = 0

        rows_to_process = range(0, H, stride)
        cols_to_process = range(0, W, stride)
        total_pixels = len(rows_to_process) * len(cols_to_process)
        log_every = max(1, len(rows_to_process) // 10)

        logger.info(
            "LRX: outer=%d inner=%d stride=%d reg=%.0e min_bg=%d | "
            "grid=%d rows x %d cols = %d pixels | max_bg=%d",
            outer, inner, stride, reg, min_bg,
            len(rows_to_process), len(cols_to_process),
            total_pixels, max_bg,
        )

        def _flush_batch():
            """Send accumulated batch to device, compute scores, write back."""
            nonlocal batch_idx, n_scored, n_batches_flushed, t_solve
            if batch_idx == 0:
                return
            t0 = time.time()
            scores = batch_mahalanobis(
                batch_X_bg, batch_n_bg, batch_x_test,
                batch_idx, B, reg, device,
            )
            for i, (br, bc) in enumerate(batch_coords):
                score_map[br, bc] = scores[i]
            n_scored += batch_idx
            n_batches_flushed += 1
            t_solve += time.time() - t0
            batch_idx = 0
            batch_coords.clear()

        # ----- main pixel loop (CPU extraction, GPU linalg) -----
        for step_r, r in enumerate(rows_to_process):
            if step_r % log_every == 0:
                pct = 100.0 * step_r / len(rows_to_process) if len(rows_to_process) > 0 else 0
                logger.info(
                    "LRX progress: row %d / %d (%.0f%%) | scored=%d skipped_bg=%d",
                    r, H, pct, n_scored, n_skipped_bg,
                )

            r0 = max(0, r - outer)
            r1 = min(H, r + outer + 1)

            for c in cols_to_process:
                if not spatial_valid[r, c]:
                    n_skipped_invalid += 1
                    continue

                t0 = time.time()

                c0 = max(0, c - outer)
                c1 = min(W, c + outer + 1)

                win_valid = spatial_valid[r0:r1, c0:c1]
                win_sub   = sub[:, r0:r1, c0:c1]

                bg_mask = win_valid.copy()
                gr0 = max(0, r - inner) - r0
                gr1 = min(H, r + inner + 1) - r0
                gc0 = max(0, c - inner) - c0
                gc1 = min(W, c + inner + 1) - c0
                bg_mask[gr0:gr1, gc0:gc1] = False

                n_bg = int(bg_mask.sum())
                if n_bg < min_bg:
                    n_skipped_bg += 1
                    t_extract += time.time() - t0
                    continue

                X_bg = win_sub[:, bg_mask].T   # (n_bg, B)

                batch_X_bg[batch_idx, :n_bg] = X_bg
                batch_n_bg[batch_idx] = n_bg
                batch_x_test[batch_idx] = sub[:, r, c]
                batch_coords.append((r, c))
                batch_idx += 1
                t_extract += time.time() - t0

                if batch_idx == batch_size:
                    _flush_batch()

        # Flush remaining
        _flush_batch()

        logger.info(
            "LRX compute done | scored=%d skipped_bg=%d skipped_invalid=%d | "
            "batches=%d",
            n_scored, n_skipped_bg, n_skipped_invalid, n_batches_flushed,
        )
        logger.info(
            "LRX timing | extract=%.2fs solve=%.2fs (%.1f%% on device)",
            t_extract, t_solve,
            100.0 * t_solve / (t_extract + t_solve) if (t_extract + t_solve) > 0 else 0,
        )

        # If stride > 1, interpolate NaN-free scored pixels to full resolution
        if stride > 1:
            t_interp = time.time()
            score_map = self._interpolate_strided(score_map, spatial_valid, stride)
            t_interp = time.time() - t_interp
            logger.info("LRX: stride=%d interpolation done in %.2fs", stride, t_interp)

        n_scored_final = int(np.sum(~np.isnan(score_map) & spatial_valid))
        t_total = time.time() - t_total
        logger.info(
            "LRX: total %.2fs | scored %d / %d valid pixels",
            t_total, n_scored_final, int(spatial_valid.sum()),
        )

        self._last_result = LocalRXResult(
            lrx_score_map=score_map,
            spatial_mask=mask,
            computed_mask=(~np.isnan(score_map)) & mask,
            good_band_indices=good,
            good_band_wavelengths=self._good_wavelengths,
            n_valid_pixels=int(mask.sum()),
            n_scored_pixels=n_scored_final,
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

        sub = score_map[::stride, ::stride]
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
