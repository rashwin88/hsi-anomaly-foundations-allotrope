"""
Combined destriper: FFT notch (high-frequency) + tilted moment-matching
(low-frequency).

The FFT notch removes fine periodic stripes and detects the stripe angle.
Moment-matching along that angle removes the broad per-detector gain/offset
variation. They target different parts of the stripe spectrum.

The two underlying transformers are independent and composable — this
orchestrator composes them but either can be used standalone.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.abstract_classes.data_transformer import DataTransformer
from app.models.dataset.transformations import Transformation
from app.utils.data_transformations.frequency_domain_destriper import (
    FrequencyDomainDestriper,
    FrequencyDestripeDiagnostics,
)
from app.utils.data_transformations.moment_matching_destriper import (
    MomentMatchingDestriper,
)

logger = logging.getLogger(__name__)


@dataclass
class CombinedDestripeDiagnostics:
    """Diagnostics from both stages plus 3-stage σ tracking."""

    fft_diagnostics: FrequencyDestripeDiagnostics = field(default=None)
    detected_angles: list[float] = field(default=None)

    # Column-mean σ at three stages for the representative band
    sigma_original: float = field(default=None)
    sigma_after_fft: float = field(default=None)
    sigma_after_combined: float = field(default=None)


class CombinedDestriper(DataTransformer):
    """
    Two-stage destriper:
        1. FrequencyDomainDestriper — removes high-frequency stripe component.
        2. MomentMatchingDestriper — removes low-frequency gain/offset along
           the stripe angle detected in stage 1.

    If stage 1 finds no significant stripe, the cube is returned unchanged.
    """

    def __init__(self, **fft_kwargs):
        super().__init__(
            transformation_category=Transformation.COMPOSITE_DESTRIPE
        )
        self._fft_destriper = FrequencyDomainDestriper(**fft_kwargs)
        self._mm_destriper = MomentMatchingDestriper()
        self._diagnostics: CombinedDestripeDiagnostics | None = None

    @property
    def detected_angles(self) -> list[float] | None:
        if self._diagnostics is None or self._diagnostics.detected_angles is None:
            return None
        return self._diagnostics.detected_angles

    @property
    def diagnostics(self) -> CombinedDestripeDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError(
                "No diagnostics available. Call transform() with diagnostics=True."
            )
        return self._diagnostics

    def transform(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Destripe a BSQ reflectance cube.

        Args:
            input_data:     (B, H, W) reflectance cube.
            validity_mask:  (B, H, W) binary mask, 1 = valid. Required.
            diagnostics:    bool, default False.

        Returns:
            Destriped cube, same shape and dtype as input_data.
        """
        validity_mask = kwargs.get("validity_mask")
        if validity_mask is None:
            raise ValueError("validity_mask is required.")

        capture = kwargs.get("diagnostics", False)
        band_wavelengths      = kwargs.get("band_wavelengths", None)
        exclusion_ranges      = kwargs.get("exclusion_ranges", None)
        spectral_family_order = kwargs.get("spectral_family_order", None)

        diag = CombinedDestripeDiagnostics() if capture else None
        self._diagnostics = diag

        # Stage 1: FFT notch (also detects the stripe angle)
        t0 = time.time()
        logger.info("CombinedDestriper: starting FFT destripe stage…")
        fft_result = self._fft_destriper.transform(
            input_data,
            validity_mask=validity_mask,
            diagnostics=capture,
            band_wavelengths=band_wavelengths,
            exclusion_ranges=exclusion_ranges,
            spectral_family_order=spectral_family_order,
        )
        logger.info("CombinedDestriper: FFT stage done in %.2fs", time.time() - t0)

        fft_diag = self._fft_destriper._diagnostics
        angles = self._fft_destriper._last_detected_angles or None

        if diag is not None:
            diag.fft_diagnostics = fft_diag
            diag.detected_angles = angles

        if not angles:
            logger.info("No stripes detected — skipping moment-matching.")
            return fft_result

        # σ tracking — always compute rep band stats for the safety guard.
        # Prefer the band the FFT destriper used (most meaningful for stripe
        # strength); fall back to the midpoint band if diagnostics were off.
        if fft_diag is not None and fft_diag.representative_band_index is not None:
            rep = fft_diag.representative_band_index
        else:
            rep = input_data.shape[0] // 2
        rep_validity = validity_mask[rep].astype(bool)

        if diag is not None:
            diag.sigma_original = self._column_mean_sigma(
                input_data[rep], rep_validity
            )
            diag.sigma_after_fft = self._column_mean_sigma(
                fft_result[rep], rep_validity
            )

        # Stage 2: Tilted moment-matching — sequentially for each angle
        # with σ safety guard: skip any angle that makes things worse
        logger.info("CombinedDestriper: starting moment-matching for %d angle(s)…", len(angles))
        t_mm = time.time()
        result = fft_result
        for angle in angles:
            logger.info(
                "Applying moment-matching along %.1f° stripe direction.", angle
            )
            candidate = self._mm_destriper.transform(
                result,
                validity_mask=validity_mask,
                stripe_angle=angle,
            )

            if True:  # σ guard always runs
                sigma_before = self._column_mean_sigma(result[rep], rep_validity)
                sigma_after = self._column_mean_sigma(candidate[rep], rep_validity)

                if sigma_after > sigma_before:
                    logger.warning(
                        "Moment-matching at %.1f° increased σ (%.6f → %.6f). "
                        "Skipping this angle.",
                        angle, sigma_before, sigma_after,
                    )
                    continue

            result = candidate

        logger.info("CombinedDestriper: moment-matching done in %.2fs", time.time() - t_mm)

        if diag is not None:
            diag.sigma_after_combined = self._column_mean_sigma(
                result[rep], rep_validity
            )
            logger.info(
                "Column-mean σ: original=%.6f → FFT=%.6f → combined=%.6f",
                diag.sigma_original, diag.sigma_after_fft, diag.sigma_after_combined,
            )

        return result

    @staticmethod
    def _column_mean_sigma(band: np.ndarray, validity: np.ndarray) -> float:
        """Compute std of per-column means across valid pixels."""
        H, W = band.shape
        col_means = []
        for j in range(W):
            col_valid = validity[:, j]
            if col_valid.sum() > 0:
                col_means.append(band[col_valid, j].mean())
        return float(np.std(col_means)) if col_means else 0.0
