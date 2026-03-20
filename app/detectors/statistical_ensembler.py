"""
Statistical Ensembler — runs Global RX and Local RX with optional
composite destriping and fuses their CDF-normalized score maps.

Pipeline
--------
1. fit()  — band filtering (shared between GRX and LRX).
2. detect() —
   a. Optionally destripe the cube via CombinedDestriper.
   b. Run GlobalRXDetector.detect() on the (destriped) cube.
   c. Run LocalRXDetector.detect() on the (destriped) cube.
   d. CDF-normalize both score maps to [0, 1].
   e. Fuse using product, maximum, and mean strategies.

The default fusion strategy returned by detect() is "product" (conservative:
a pixel must look anomalous under both global and local statistics).
"""

import logging
from typing import List

import numpy as np

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.detectors.global_rx_detector import GlobalRXDetector
from app.detectors.local_rx_detector import LocalRXDetector
from app.models.anomaly_detection.ensemble_rx_result import (
    EnsembleRXResult,
    cdf_normalize,
    fuse_maps,
)
from app.utils.data_transformations.composite_destriper import CombinedDestriper

logger = logging.getLogger(__name__)


class StatisticalEnsembler(AnomalyDetector):
    """
    Ensemble detector that combines Global RX and Local RX scores.

    Usage::

        ensembler = StatisticalEnsembler(vendable)
        ensembler.fit()
        score_map = ensembler.detect(
            vendable.reflectance_cube,
            destripe=True,
        )
        result = ensembler.result          # full EnsembleRXResult
        product_map = result.fused_score_maps["product"]
    """

    def __init__(self, vendable: VendableDataset):
        super().__init__(vendable)
        self._grx = GlobalRXDetector(vendable)
        self._lrx = LocalRXDetector(vendable)
        self._good_indices: List[int] | None = None
        self._good_wavelengths: List[float] | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, **kwargs) -> None:
        """
        Fit both sub-detectors (shared band filtering kwargs).

        kwargs are forwarded to both GlobalRXDetector.fit() and
        LocalRXDetector.fit() (band_failure_threshold, exclusion_ranges).
        """
        logger.info("StatisticalEnsembler: fitting GRX…")
        self._grx.fit(**kwargs)

        logger.info("StatisticalEnsembler: fitting LRX…")
        self._lrx.fit(**kwargs)

        # Store the intersection of surviving bands for metadata
        grx_good = set(self._grx._good_indices)
        lrx_good = set(self._lrx._good_indices)
        shared = sorted(grx_good & lrx_good)
        self._good_indices = shared
        self._good_wavelengths = [
            self._vendable.band_cw_order[i] for i in shared
        ]
        self._fitted = True

        logger.info(
            "StatisticalEnsembler: GRX kept %d bands, LRX kept %d bands, "
            "%d shared.",
            len(grx_good), len(lrx_good), len(shared),
        )

    # ------------------------------------------------------------------
    # detect
    # ------------------------------------------------------------------

    def detect(self, cube: np.ndarray, validity_mask: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Run the full ensemble pipeline.

        Args:
            cube:            (B, H, W) reflectance cube.
            validity_mask:   (B, H, W) binary mask. If None, uses vendable's.
            destripe:        bool, default True. Apply CombinedDestriper first.
            fft_kwargs:      dict of kwargs forwarded to CombinedDestriper.
            default_strategy: fusion strategy returned as the primary score map.
                              One of "product", "maximum", "mean". Default "product".
            **kwargs:        remaining kwargs forwarded to LRX detect()
                             (outer_window, inner_window, stride, etc.).

        Returns:
            (H, W) fused score map for the chosen strategy.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before detect().")

        do_destripe = kwargs.pop("destripe", True)
        fft_kwargs = kwargs.pop("fft_kwargs", {})
        default_strategy = kwargs.pop("default_strategy", "product")

        validity = (
            validity_mask if validity_mask is not None
            else self._vendable.validity_cube
        )

        # --- optional destriping -----------------------------------------
        if do_destripe:
            logger.info("StatisticalEnsembler: applying CombinedDestriper…")
            destriper = CombinedDestriper(**fft_kwargs)
            working_cube = destriper.transform(
                cube,
                validity_mask=validity,
                band_wavelengths=self._vendable.band_cw_order,
                spectral_family_order=getattr(
                    self._vendable, "spectral_family_order", None
                ),
                diagnostics=True,
            )
            logger.info("StatisticalEnsembler: destriping complete.")
        else:
            working_cube = cube

        # --- GRX ---------------------------------------------------------
        logger.info("StatisticalEnsembler: running Global RX…")
        grx_scores = self._grx.detect(working_cube, validity_mask=validity)
        grx_result = self._grx.result

        # --- LRX ---------------------------------------------------------
        logger.info("StatisticalEnsembler: running Local RX…")
        lrx_scores = self._lrx.detect(working_cube, validity_mask=validity, **kwargs)
        lrx_result = self._lrx.result

        # --- normalization & fusion --------------------------------------
        combined_mask = grx_result.spatial_mask & lrx_result.computed_mask

        norm_grx = cdf_normalize(grx_scores, combined_mask)
        norm_lrx = cdf_normalize(lrx_scores, combined_mask)

        fused = fuse_maps(norm_grx, norm_lrx, combined_mask)
        n_valid = int(combined_mask.sum())

        logger.info(
            "StatisticalEnsembler: %d pixels scored by both detectors.", n_valid
        )

        self._last_result = EnsembleRXResult(
            grx_result=grx_result,
            lrx_result=lrx_result,
            normalized_grx_map=norm_grx,
            normalized_lrx_map=norm_lrx,
            fused_score_maps=fused,
            spatial_mask=combined_mask,
            good_band_indices=self._good_indices,
            good_band_wavelengths=self._good_wavelengths,
            n_valid_pixels=n_valid,
            n_good_bands=len(self._good_indices),
            destriped=do_destripe,
        )

        return fused[default_strategy]

    # ------------------------------------------------------------------
    # result
    # ------------------------------------------------------------------

    @property
    def result(self) -> EnsembleRXResult:
        if not hasattr(self, "_last_result"):
            raise RuntimeError("Call detect() before accessing result.")
        return self._last_result
