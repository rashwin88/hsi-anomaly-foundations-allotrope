"""
MNF Compression + Local RX anomaly detector.

Applies Minimum Noise Fraction (MNF) dimensionality reduction before
computing Local RX scores.  MNF orders components by signal-to-noise
ratio (descending), so the first ``n_components`` capture the most
information-rich spectral variation while suppressing sensor noise
and striping artefacts.

The noise covariance is estimated from spatial first-differences
(shift-difference method), following Green et al. 1988.

Pipeline:
    fit()   â†’ band filtering + spatial mask + estimate noise covariance
              + compute MNF transform matrix
    detect()â†’ project good-band cube into MNF space (n_components)
              + run Local RX (annulus-based Mahalanobis) on the compressed cube
"""

import logging
import time
from typing import List, Dict

import numpy as np
import torch
from scipy.ndimage import zoom

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.detectors._local_background import batch_mahalanobis, select_device
from app.models.anomaly_detection.mnf_lrx_result import MNFCompressionLRXResult
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter

logger = logging.getLogger(__name__)

DEFAULT_BAND_FAILURE_THRESHOLD = 0.05
DEFAULT_N_COMPONENTS = 10
DEFAULT_OUTER_WINDOW = 25
DEFAULT_INNER_WINDOW = 5
DEFAULT_REGULARIZATION = 1e-4
DEFAULT_BATCH_SIZE = 256



class MNFCompressionLRXDetector(AnomalyDetector):
    """
    Local RX with MNF dimensionality reduction.

    Call fit() then detect().  ``n_components`` controls how many MNF
    bands are retained (default 10).  LRX parameters (outer_window,
    inner_window, stride, etc.) are passed to detect().
    """

    def __init__(self, vendable: VendableDataset):
        super().__init__(vendable)
        self._good_indices: List[int] | None = None
        self._good_wavelengths: List[float] | None = None
        self._spatial_mask: np.ndarray | None = None
        self._mnf_components: np.ndarray | None = None  # (n_components, n_bands)
        self._mnf_mean: np.ndarray | None = None        # (n_bands,)
        self._n_components: int = DEFAULT_N_COMPONENTS
        self._eigenvalues: np.ndarray | None = None
        self._logs: Dict = {}

    @property
    def logs(self) -> Dict:
        return self._logs

    # ------------------------------------------------------------------ fit
    def fit(self, **kwargs) -> None:
        """
        Band filtering, spatial masking, and MNF transform estimation.

        kwargs:
            n_components (int): number of MNF components to retain.
                Default 10.
            band_failure_threshold (float): max per-pixel failure rate
                for a band to survive stage 2. Default 0.05.
            exclusion_ranges: wavelength exclusion ranges (nm).
            min_band_coverage (float): coverage-fraction threshold.
                Default 0.95.
        """
        self._logs = {}
        self._n_components = kwargs.get("n_components", DEFAULT_N_COMPONENTS)
        threshold = kwargs.get(
            "band_failure_threshold", DEFAULT_BAND_FAILURE_THRESHOLD
        )
        min_band_coverage = kwargs.get("min_band_coverage", 0.95)
        exclusion_ranges = kwargs.get("exclusion_ranges", None)

        validity = self._vendable.validity_cube

        # ---- Stage 1: band-level filtering by flag + wavelength ----
        band_filter = SpectralBandFilter(
            band_wavelengths=self._vendable.band_cw_order,
            band_validity_flags=self._vendable.band_validity_by_position,
            exclusion_ranges=exclusion_ranges,
        )
        stage1 = band_filter.get_good_band_indices()

        # ---- Stage 2: per-pixel failure rate ----
        any_valid = validity.any(axis=0)
        n_spatial = int(any_valid.sum())
        good = []
        self._logs["band_wise_failure_ratio"] = {}
        for idx in stage1:
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
            "MNF-LRX: %d / %d bands survive stage-2 (threshold=%.1f%%)",
            len(good), len(stage1), threshold * 100,
        )

        # ---- Spatial mask ----
        n_good = len(good)
        if n_good == 0:
            self._spatial_mask = np.zeros(validity.shape[1:], dtype=bool)
            return

        valid_count = validity[good].sum(axis=0)
        coverage_fraction = valid_count.astype(np.float32) / n_good
        self._spatial_mask = coverage_fraction >= min_band_coverage

        n_valid = int(self._spatial_mask.sum())
        logger.info(
            "MNF-LRX: spatial mask â€” %d valid pixels, %d bands", n_valid, n_good,
        )

        if n_valid < n_good + 1:
            logger.warning(
                "Too few valid pixels (%d) for %d bands â€” MNF will fail.",
                n_valid, n_good,
            )
            return

        # ---- Build pixel matrix (N, B) for valid pixels ----
        cube = self._vendable.normalized_hyperspectral_cube
        working = cube[good].copy()  # (B_good, H, W)

        mask = self._spatial_mask
        for b_idx, b in enumerate(good):
            band_valid = validity[b].astype(bool)
            needs_fill = mask & (~band_valid)
            n_fill = int(needs_fill.sum())
            if n_fill > 0:
                donor = band_valid & mask
                fill_value = float(working[b_idx][donor].mean())
                working[b_idx][needs_fill] = fill_value

        pixels = working[:, mask].T  # (N, B_good)

        # ---- Estimate noise covariance (shift-difference) ----
        t_mnf = time.time()
        noise_cov = self._estimate_noise_covariance(working, mask)

        # ---- MNF transform ----
        self._compute_mnf_transform(pixels, noise_cov)
        logger.info(
            "MNF-LRX: transform computed in %.2fs â€” retaining %d / %d components",
            time.time() - t_mnf, self._n_components, n_good,
        )

        if self._eigenvalues is not None:
            top = self._eigenvalues[: self._n_components]
            self._logs["mnf_eigenvalues_retained"] = top.tolist()
            logger.info(
                "MNF-LRX: eigenvalue range [%.4f, %.4f] for retained components",
                float(top.min()), float(top.max()),
            )

    # ---------------------------------------------------------------- detect
    def detect(self, cube, validity_mask=None, **kwargs):
        """
        MNF-compress the cube, then run Local RX.

        Args:
            cube:           (B, H, W) reflectance cube.
            validity_mask:  (B, H, W) binary mask. If None, uses vendable's.
            outer_window:   Background window half-size. Default 25.
            inner_window:   Guard window half-size. Default 5.
            regularization: Ridge term for covariance. Default 1e-4.
            min_bg_pixels:  Minimum background pixels to score. Default n_components+1.
            stride:         Spatial subsampling factor. Default 1.
            batch_size:     Pixels per GPU batch. Default 256.

        Returns:
            (H, W) score map with NaN for invalid pixels.
        """
        if self._good_indices is None or self._spatial_mask is None:
            raise RuntimeError("Call fit() before detect().")
        if self._mnf_components is None:
            raise RuntimeError("MNF transform not computed â€” check fit() logs.")

        validity = (
            validity_mask if validity_mask is not None
            else self._vendable.validity_cube
        )

        _, H, W = cube.shape
        good = self._good_indices
        mask = self._spatial_mask
        n_components = self._n_components

        outer = int(kwargs.get("outer_window", DEFAULT_OUTER_WINDOW))
        inner = int(kwargs.get("inner_window", DEFAULT_INNER_WINDOW))
        reg = float(kwargs.get("regularization", DEFAULT_REGULARIZATION))
        stride = int(kwargs.get("stride", 1))
        min_bg = int(kwargs.get("min_bg_pixels", n_components + 1))

        t_total = time.time()

        # ---- MNF projection ----
        t_proj = time.time()
        working = cube[good].copy()
        for b_idx, b in enumerate(good):
            band_valid = validity[b].astype(bool)
            needs_fill = mask & (~band_valid)
            n_fill = int(needs_fill.sum())
            if n_fill > 0:
                donor = band_valid & mask
                fill_value = float(working[b_idx][donor].mean())
                working[b_idx][needs_fill] = fill_value

        pixels = working[:, mask].T  # (N, B_good)
        pixels_centered = pixels - self._mnf_mean
        mnf_pixels = pixels_centered @ self._mnf_components.T  # (N, n_comp)

        # Build MNF cube (n_comp, H, W) for spatial windowing
        mnf_cube = np.full((n_components, H, W), 0.0, dtype=np.float64)
        mnf_cube[:, mask] = mnf_pixels.T

        logger.info(
            "MNF-LRX: %d pixels projected to %d components in %.2fs",
            mnf_pixels.shape[0], n_components, time.time() - t_proj,
        )

        # ---- Device selection + batch size ----
        device = select_device()
        compute_dtype = "float32" if device.type != "cpu" else "float64"

        if "batch_size" in kwargs:
            batch_size = int(kwargs["batch_size"])
        elif device.type == "cuda":
            max_bg = (2 * outer + 1) ** 2 - (2 * inner + 1) ** 2
            bytes_per_elem = (2 * max_bg * n_components + n_components ** 2) * 4
            vram = torch.cuda.get_device_properties(device).total_memory
            batch_size = max(256, int(0.25 * vram / bytes_per_elem))
        else:
            batch_size = DEFAULT_BATCH_SIZE

        logger.info(
            "MNF-LRX: device=%s | compute_dtype=%s | batch_size=%d",
            device, compute_dtype, batch_size,
        )
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            logger.info("MNF-LRX: GPU=%s (%.1f GB)", gpu_name, gpu_mem)

        # ---- LRX sliding window ----
        max_bg = (2 * outer + 1) ** 2 - (2 * inner + 1) ** 2
        B = n_components
        spatial_valid = mask
        score_map = np.full((H, W), np.nan, dtype=np.float64)

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
            "MNF-LRX: outer=%d inner=%d stride=%d reg=%.0e min_bg=%d | "
            "grid=%d rows x %d cols = %d pixels | max_bg=%d | B=%d (MNF components)",
            outer, inner, stride, reg, min_bg,
            len(rows_to_process), len(cols_to_process),
            total_pixels, max_bg, B,
        )

        def _flush_batch():
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

        for step_r, r in enumerate(rows_to_process):
            if step_r % log_every == 0:
                pct = 100.0 * step_r / len(rows_to_process) if len(rows_to_process) > 0 else 0
                logger.info(
                    "MNF-LRX progress: row %d / %d (%.0f%%) | scored=%d skipped_bg=%d",
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
                win_sub = mnf_cube[:, r0:r1, c0:c1]

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
                batch_x_test[batch_idx] = mnf_cube[:, r, c]
                batch_coords.append((r, c))
                batch_idx += 1
                t_extract += time.time() - t0

                if batch_idx == batch_size:
                    _flush_batch()

        _flush_batch()

        logger.info(
            "MNF-LRX compute done | scored=%d skipped_bg=%d skipped_invalid=%d | "
            "batches=%d",
            n_scored, n_skipped_bg, n_skipped_invalid, n_batches_flushed,
        )
        logger.info(
            "MNF-LRX timing | extract=%.2fs solve=%.2fs (%.1f%% on device)",
            t_extract, t_solve,
            100.0 * t_solve / (t_extract + t_solve) if (t_extract + t_solve) > 0 else 0,
        )

        # If stride > 1, interpolate back to full resolution
        if stride > 1:
            t_interp = time.time()
            score_map = self._interpolate_strided(score_map, spatial_valid, stride)
            logger.info(
                "MNF-LRX: stride=%d interpolation done in %.2fs",
                stride, time.time() - t_interp,
            )

        n_scored_final = int(np.sum(~np.isnan(score_map) & spatial_valid))
        t_total = time.time() - t_total
        logger.info(
            "MNF-LRX: total %.2fs | scored %d / %d valid pixels",
            t_total, n_scored_final, int(spatial_valid.sum()),
        )

        self._last_result = MNFCompressionLRXResult(
            lrx_score_map=score_map,
            spatial_mask=mask,
            computed_mask=(~np.isnan(score_map)) & mask,
            good_band_indices=good,
            good_band_wavelengths=self._good_wavelengths,
            n_valid_pixels=int(mask.sum()),
            n_scored_pixels=n_scored_final,
            n_good_bands=len(good),
            n_components=self._n_components,
            mnf_eigenvalues=self._eigenvalues.tolist() if self._eigenvalues is not None else [],
            outer_window=outer,
            inner_window=inner,
        )

        return score_map

    @property
    def result(self) -> MNFCompressionLRXResult:
        if not hasattr(self, "_last_result"):
            raise RuntimeError("Call detect() before accessing result.")
        return self._last_result

    # ------------------------------------------------------------ internals
    def _estimate_noise_covariance(
        self, working: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate noise covariance via spatial shift-difference.

        Uses horizontal first-differences of valid pixels.  The noise
        covariance is half the covariance of the differences (Green et
        al. 1988).
        """
        B, H, W = working.shape
        left_mask = mask[:, :-1] & mask[:, 1:]
        diffs = working[:, :, 1:] - working[:, :, :-1]
        diff_pixels = diffs[:, left_mask].T

        logger.info(
            "MNF-LRX: noise covariance from %d spatial differences",
            diff_pixels.shape[0],
        )

        noise_cov = np.cov(diff_pixels, rowvar=False) / 2.0
        return noise_cov

    def _compute_mnf_transform(
        self, pixels: np.ndarray, noise_cov: np.ndarray
    ) -> None:
        """
        Compute MNF transform matrix.

        MNF = PCA in noise-whitened space:
            1. Whiten by noise covariance: W = Sigma_n^{-1/2}
            2. Compute data covariance in whitened space
            3. Eigenvectors of whitened covariance, sorted by descending
               eigenvalue -> MNF components
        """
        n_bands = pixels.shape[1]
        n_components = min(self._n_components, n_bands)
        self._n_components = n_components

        self._mnf_mean = pixels.mean(axis=0)
        centered = pixels - self._mnf_mean

        noise_eigvals, noise_eigvecs = np.linalg.eigh(noise_cov)
        noise_eigvals = np.maximum(noise_eigvals, 1e-10)
        whiten = noise_eigvecs @ np.diag(1.0 / np.sqrt(noise_eigvals)) @ noise_eigvecs.T

        whitened = centered @ whiten.T

        data_cov_w = np.cov(whitened, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(data_cov_w)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self._eigenvalues = eigvals

        combined = (eigvecs[:, :n_components].T) @ whiten
        self._mnf_components = combined

        logger.info(
            "MNF-LRX: eigenvalue ratio (1st/last retained): %.2f",
            float(eigvals[0] / eigvals[n_components - 1])
            if eigvals[n_components - 1] > 0 else float("inf"),
        )

    @staticmethod
    def _interpolate_strided(
        score_map: np.ndarray,
        spatial_valid: np.ndarray,
        stride: int,
    ) -> np.ndarray:
        """Bilinear interpolation of a strided score map back to full resolution."""
        H, W = score_map.shape

        sub = score_map[::stride, ::stride]
        sub_valid = ~np.isnan(sub)

        filled = sub.copy()
        filled[~sub_valid] = 0.0

        zoomed_scores = zoom(filled, stride, order=1)
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
