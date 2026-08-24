"""
Inference harness for anomaly detection.

Bridges the vendable dataset layer to the AnomalyDetector ABC.
Handles full-scene and patch-level inference with overlap-averaged
reassembly. Framework-agnostic — operates entirely in numpy.
"""

import logging

import numpy as np

from app.abstract_classes.anomaly_detector import AnomalyDetector, VendableDataset
from app.models.anomaly_detection.harness_config import InferenceHarnessConfig
from app.models.anomaly_detection.detection_result import AnomalyDetectionResult
# The concrete types are still needed for the isinstance dispatch in
# _extract_cube_and_validity; VendableDataset above is only the union alias.
from app.models.dataset.vendables import (
    VendableHyperspectralDataset,
    VendableEnmapHyperspectralDataset,
    VendableThermalDataset,
)
from app.models.patches.patching_request import PatchRequest
from app.models.patches.patching_response import PatchingPlan
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator

logger = logging.getLogger(__name__)


class InferenceHarness:
    """
    Orchestrates anomaly detection inference on a VendableDataset.

    Responsibilities:
        1. Extract the data cube and validity mask from any vendable type.
        2. Optionally call detector.fit on the full scene.
        3. Run detection — full-scene or patched with overlap-averaged reassembly.
    """

    def __init__(self, config: InferenceHarnessConfig):
        self._config = config
        self._patch_planner = PatchPlanGenerator()

    def run(
        self,
        vendable: VendableDataset,
        detector: AnomalyDetector,
    ) -> AnomalyDetectionResult:
        cube, validity = self._extract_cube_and_validity(vendable)

        if self._config.fit_on_full_scene:
            detector.fit()

        if self._config.patch_config is None:
            return self._run_full_scene(cube, validity, detector)

        return self._run_patched(cube, validity, detector)

    @staticmethod
    def _extract_cube_and_validity(
        vendable: VendableDataset,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Pull the primary data cube and validity mask from any vendable type."""
        if isinstance(vendable, VendableHyperspectralDataset):
            return vendable.normalized_hyperspectral_cube, vendable.validity_cube
        if isinstance(vendable, VendableEnmapHyperspectralDataset):
            return vendable.normalized_hyperspectral_cube, vendable.validity_cube
        if isinstance(vendable, VendableThermalDataset):
            return vendable.normalized_thermal_cube, vendable.validity_cube
        raise TypeError(f"Unsupported vendable type: {type(vendable)}")


    @staticmethod
    def _run_full_scene(
        cube: np.ndarray,
        validity: np.ndarray | None,
        detector: AnomalyDetector,
    ) -> AnomalyDetectionResult:
        score_map = detector.detect(cube, validity)
        return AnomalyDetectionResult(
            score_map=score_map,
            source_cube_shape=cube.shape,
            patches_processed=1,
        )

    def _run_patched(
        self,
        cube: np.ndarray,
        validity: np.ndarray | None,
        detector: AnomalyDetector,
    ) -> AnomalyDetectionResult:
        pc = self._config.patch_config
        plan = self._build_patch_plan(cube.shape)
        coords = plan.patch_coordinates

        _, H, W = cube.shape
        score_accumulator = np.zeros((H, W), dtype=np.float64)
        count_accumulator = np.zeros((H, W), dtype=np.float64)

        for batch_start in range(0, len(coords), self._config.batch_size):
            batch_coords = coords[batch_start : batch_start + self._config.batch_size]

            cube_patches = np.stack(
                [
                    cube[:, r : r + pc.patch_height, c : c + pc.patch_width]
                    for r, c in batch_coords
                ]
            )

            validity_patches = None
            if validity is not None:
                validity_patches = np.stack(
                    [
                        validity[:, r : r + pc.patch_height, c : c + pc.patch_width]
                        for r, c in batch_coords
                    ]
                )

            batch_scores = detector.detect_batch(cube_patches, validity_patches)

            for idx, (r, c) in enumerate(batch_coords):
                score_accumulator[r : r + pc.patch_height, c : c + pc.patch_width] += batch_scores[idx]
                count_accumulator[r : r + pc.patch_height, c : c + pc.patch_width] += 1.0

        count_accumulator = np.maximum(count_accumulator, 1.0)
        final_scores = score_accumulator / count_accumulator

        return AnomalyDetectionResult(
            score_map=final_scores,
            source_cube_shape=cube.shape,
            patches_processed=len(coords),
        )

    def _build_patch_plan(self, cube_shape: tuple[int, ...]) -> PatchingPlan:
        pc = self._config.patch_config
        request = PatchRequest(
            input_cube=cube_shape,
            width=pc.patch_width,
            height=pc.patch_height,
            stride=pc.stride,
        )
        return self._patch_planner.generate_patching_plan(request)
