"""
Thermal Global RX anomaly detector for single-band thermal data.

Operates on VendableThermalDataset (e.g. Landsat 9 B10).  Since thermal
data has a single band with no spectral metadata, band filtering is
skipped entirely.  The spatial mask is derived directly from the
validity cube.

For a single-band cube, Global RX reduces to a z-score: each pixel's
anomaly score is its squared deviation from the global mean temperature,
scaled by the inverse variance.

Pipeline:
    fit()   → build spatial mask from validity_cube
    detect()→ run spectral.rx on valid pixels, map scores to (H, W)
"""

import logging
import time

import numpy as np
import spectral

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.models.anomaly_detection.thermal_rx_result import ThermalGRXResult

logger = logging.getLogger(__name__)


class ThermalGRXDetector(AnomalyDetector):
    """
    Global RX detector for single-band thermal cubes.

    Call fit() then detect().
    """

    def __init__(self, vendable: VendableDataset):
        super().__init__(vendable)
        self._spatial_mask: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, **kwargs) -> None:
        """
        Build spatial mask from validity cube.

        No band filtering — thermal data has a single band with no
        spectral metadata.

        kwargs:
            (reserved for future use)
        """
        validity = self._vendable.validity_cube  # (1, H, W)
        self._spatial_mask = validity[0].astype(bool)  # (H, W)

        n_valid = int(self._spatial_mask.sum())
        n_total = int(self._spatial_mask.size)
        logger.info(
            "Thermal GRX: spatial mask — %d / %d pixels valid (%.1f%%)",
            n_valid, n_total, 100.0 * n_valid / n_total if n_total > 0 else 0,
        )

    # --------------------------------------------------------------- detect
    def detect(self, cube, validity_mask=None, **kwargs):
        """
        Run Global RX on the thermal cube.

        Args:
            cube:           (C, H, W) thermal cube (C=1 for Landsat B10).
            validity_mask:  (C, H, W) binary mask. If None, uses vendable's.

        Returns:
            (H, W) score map with NaN for invalid pixels.
        """
        if self._spatial_mask is None:
            raise RuntimeError("Call fit() before detect().")

        if validity_mask is not None:
            mask = validity_mask[0].astype(bool)
        else:
            mask = self._spatial_mask

        C, H, W = cube.shape
        n_valid = int(mask.sum())

        t_total = time.time()

        # Extract valid pixels: (N, C)
        pixels = cube[:, mask].T.astype(np.float64)  # (N, C)

        logger.info(
            "Thermal GRX: %d valid pixels, %d band(s)", n_valid, C,
        )

        t_rx = time.time()
        if C == 1:
            # Single-band: spectral.rx needs ≥2D covariance.
            # RX for 1 band reduces to squared z-score: (x - μ)² / σ²
            vals = pixels[:, 0]
            mu = vals.mean()
            var = vals.var()
            if var > 0:
                rx_scores_flat = ((vals - mu) ** 2) / var
            else:
                rx_scores_flat = np.zeros_like(vals)
        else:
            # Multi-band: use spectral.rx as usual
            rx_scores_flat = spectral.rx(
                np.asarray(pixels[:, np.newaxis, :])
            ).ravel()
        logger.info("Thermal GRX: RX done in %.2fs", time.time() - t_rx)

        # Map back to (H, W)
        score_map = np.full((H, W), np.nan, dtype=np.float64)
        score_map[mask] = rx_scores_flat

        valid_scores = rx_scores_flat[np.isfinite(rx_scores_flat)]
        if len(valid_scores) > 0:
            logger.info(
                "Thermal GRX: score range [%.4f, %.4f] median=%.4f p99=%.4f",
                float(valid_scores.min()), float(valid_scores.max()),
                float(np.median(valid_scores)),
                float(np.percentile(valid_scores, 99)),
            )
        logger.info("Thermal GRX: total %.2fs", time.time() - t_total)

        self._last_result = ThermalGRXResult(
            rx_score_map=score_map,
            spatial_mask=mask,
            n_valid_pixels=n_valid,
        )

        return score_map

    @property
    def result(self) -> ThermalGRXResult:
        if not hasattr(self, "_last_result"):
            raise RuntimeError("Call detect() before accessing result.")
        return self._last_result
