"""
MNF Compression + Global RX anomaly detector.

Applies Minimum Noise Fraction (MNF) dimensionality reduction before
computing Global RX scores.  MNF orders components by signal-to-noise
ratio (descending), so the first ``n_components`` capture the most
information-rich spectral variation while suppressing sensor noise.

The noise covariance is estimated from spatial first-differences
(shift-difference method), following Green et al. 1988.

Pipeline:
    fit()   → band filtering + spatial mask (delegated to GlobalRXDetector)
              + estimate noise covariance + compute MNF transform matrix
    detect()→ project good-band cube into MNF space (n_components)
              + run spectral.rx on the compressed cube
"""

import logging
import time
from typing import List, Dict

import numpy as np
import spectral

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.models.anomaly_detection.mnf_rx_result import MNFCompressionRXResult
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter

logger = logging.getLogger(__name__)

DEFAULT_BAND_FAILURE_THRESHOLD = 0.05
DEFAULT_N_COMPONENTS = 10


class MNFCompressionDetector(AnomalyDetector):
    """
    Global RX with MNF dimensionality reduction.

    Call fit() then detect().  ``n_components`` controls how many MNF
    bands are retained (default 10).
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
            "MNF: %d / %d bands survive stage-2 (threshold=%.1f%%)",
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
            "MNF: spatial mask — %d valid pixels, %d bands", n_valid, n_good,
        )

        if n_valid < n_good + 1:
            logger.warning(
                "Too few valid pixels (%d) for %d bands — MNF will fail.",
                n_valid, n_good,
            )
            return

        # ---- Build pixel matrix (N, B) for valid pixels ----
        cube = self._vendable.normalized_hyperspectral_cube
        working = cube[good].copy()  # (B_good, H, W)

        # Band-mean fill for valid pixels missing individual bands
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
            "MNF: transform computed in %.2fs — retaining %d / %d components",
            time.time() - t_mnf, self._n_components, n_good,
        )

        # Log eigenvalue info
        if self._eigenvalues is not None:
            top = self._eigenvalues[: self._n_components]
            self._logs["mnf_eigenvalues_retained"] = top.tolist()
            logger.info(
                "MNF: eigenvalue range [%.4f, %.4f] for retained components",
                float(top.min()), float(top.max()),
            )

    def detect(self, cube, validity_mask=None):
        """
        MNF-compress the cube, then run Global RX.

        Returns (H, W) score map with NaN for invalid pixels.
        """
        if self._good_indices is None or self._spatial_mask is None:
            raise RuntimeError("Call fit() before detect().")
        if self._mnf_components is None:
            raise RuntimeError("MNF transform not computed — check fit() logs.")

        validity = (
            validity_mask if validity_mask is not None
            else self._vendable.validity_cube
        )

        _, H, W = cube.shape
        good = self._good_indices
        mask = self._spatial_mask
        n_valid = int(mask.sum())

        t_total = time.time()

        # Extract good bands + band-mean fill
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

        # MNF projection: (N, B_good) @ (B_good, n_components) → (N, n_components)
        pixels_centered = pixels - self._mnf_mean
        mnf_pixels = pixels_centered @ self._mnf_components.T  # (N, n_comp)

        logger.info(
            "MNF-GRX: %d pixels projected to %d components",
            mnf_pixels.shape[0], mnf_pixels.shape[1],
        )

        # Run spectral.rx on MNF-compressed data
        t_rx = time.time()
        rx_scores_flat = spectral.rx(
            np.asarray(mnf_pixels[:, np.newaxis, :])
        ).ravel()
        logger.info("MNF-GRX: spectral.rx done in %.2fs", time.time() - t_rx)

        # Map back to (H, W)
        score_map = np.full((H, W), np.nan, dtype=np.float64)
        score_map[mask] = rx_scores_flat

        valid_scores = rx_scores_flat[np.isfinite(rx_scores_flat)]
        if len(valid_scores) > 0:
            logger.info(
                "MNF-GRX: score range [%.4f, %.4f] median=%.4f p99=%.4f",
                float(valid_scores.min()), float(valid_scores.max()),
                float(np.median(valid_scores)),
                float(np.percentile(valid_scores, 99)),
            )
        logger.info("MNF-GRX: total %.2fs", time.time() - t_total)

        self._last_result = MNFCompressionRXResult(
            rx_score_map=score_map,
            spatial_mask=mask,
            good_band_indices=good,
            good_band_wavelengths=self._good_wavelengths,
            n_valid_pixels=n_valid,
            n_good_bands=len(good),
            n_components=self._n_components,
            mnf_eigenvalues=self._eigenvalues.tolist() if self._eigenvalues is not None else [],
        )

        return score_map

    @property
    def result(self) -> "MNFCompressionRXResult":
        """Full result with metadata. Available after detect()."""
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

        Args:
            working: (B, H, W) good-band cube with band-mean fill applied.
            mask:    (H, W) spatial validity mask.

        Returns:
            (B, B) noise covariance matrix.
        """
        B, H, W = working.shape

        # Horizontal differences where both the pixel and its right
        # neighbour are valid
        left_mask = mask[:, :-1] & mask[:, 1:]
        diffs = working[:, :, 1:] - working[:, :, :-1]  # (B, H, W-1)
        diff_pixels = diffs[:, left_mask].T  # (N_diff, B)

        logger.info(
            "MNF: noise covariance from %d spatial differences", diff_pixels.shape[0],
        )

        noise_cov = np.cov(diff_pixels, rowvar=False) / 2.0
        return noise_cov

    def _compute_mnf_transform(
        self, pixels: np.ndarray, noise_cov: np.ndarray
    ) -> None:
        """
        Compute MNF transform matrix.

        MNF = PCA in noise-whitened space:
            1. Whiten by noise covariance: W = Σ_n^{-1/2}
            2. Compute data covariance in whitened space
            3. Eigenvectors of whitened covariance, sorted by descending
               eigenvalue → MNF components

        Stores:
            self._mnf_components: (n_components, n_bands) projection rows
            self._mnf_mean:       (n_bands,) mean for centering
            self._eigenvalues:    all eigenvalues (descending)
        """
        n_bands = pixels.shape[1]
        n_components = min(self._n_components, n_bands)
        self._n_components = n_components

        self._mnf_mean = pixels.mean(axis=0)
        centered = pixels - self._mnf_mean

        # Noise whitening: eigendecompose noise covariance
        noise_eigvals, noise_eigvecs = np.linalg.eigh(noise_cov)
        # Clamp small eigenvalues for numerical stability
        noise_eigvals = np.maximum(noise_eigvals, 1e-10)
        # W = V_n @ diag(1/sqrt(λ_n)) @ V_n^T
        whiten = noise_eigvecs @ np.diag(1.0 / np.sqrt(noise_eigvals)) @ noise_eigvecs.T

        # Whiten the centered data
        whitened = centered @ whiten.T  # (N, B)

        # Data covariance in whitened space
        data_cov_w = np.cov(whitened, rowvar=False)

        # Eigendecompose → MNF components
        eigvals, eigvecs = np.linalg.eigh(data_cov_w)
        # Sort descending (highest SNR first)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self._eigenvalues = eigvals

        # MNF transform: project original (centered) data
        # combined_transform = eigvecs[:, :n_components].T @ whiten
        # so that: mnf_pixel = (x - mean) @ combined_transform.T
        combined = (eigvecs[:, :n_components].T) @ whiten  # (n_comp, B)
        self._mnf_components = combined

        logger.info(
            "MNF: eigenvalue ratio (1st/last retained): %.2f",
            float(eigvals[0] / eigvals[n_components - 1])
            if eigvals[n_components - 1] > 0 else float("inf"),
        )
