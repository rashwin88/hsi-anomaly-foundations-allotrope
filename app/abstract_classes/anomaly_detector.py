"""
Abstract base class for anomaly detectors.

Defines the contract that all anomaly detection algorithms must satisfy,
whether classical (RX, CRD, LRASR) or learned (autoencoders, etc.).

The interface operates on numpy arrays to stay consistent with the vendable
dataset layer. Detectors that use torch, sklearn, or any other framework
handle their own conversion internally.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from app.models.dataset.vendables import (
    VendableHyperspectralDataset,
    VendableEnmapHyperspectralDataset,
    VendableThermalDataset,
)

VendableDataset = Union[
    VendableHyperspectralDataset, VendableEnmapHyperspectralDataset, VendableThermalDataset
]


class AnomalyDetector(ABC):
    """
    Contract for a single-scene anomaly detector.

    Every detector is initialised with a VendableDataset, giving it access
    to the data cube, validity masks, and sensor-specific metadata.

    Subclasses MUST implement `detect`. They MAY override `fit` (for
    detector-specific configuration and background statistics) and
    `detect_batch` (for batched computation).
    """

    def __init__(self, vendable: VendableDataset):
        self._vendable = vendable

    @abstractmethod
    def detect(
        self,
        cube: Union[np.ndarray, np.ma.MaskedArray],
        validity_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Produce per-pixel anomaly scores for a single cube.

        Args:
            cube:           (C, H, W) BSQ spectral/thermal cube.
            validity_mask:  (C, H, W) or (1, H, W) binary mask. 1 = valid.
                            None means all pixels are valid.

        Returns:
            (H, W) array of anomaly scores. Higher = more anomalous.
        """
        ...

    def detect_batch(
        self,
        cubes: np.ndarray,
        validity_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Score a batch of cubes.

        Default implementation iterates over the batch dimension and calls
        `detect` per element. Override for detectors that benefit from
        vectorised or GPU-batched computation.

        Args:
            cubes:           (B, C, H, W)
            validity_masks:  (B, C, H, W) or (B, 1, H, W) or None.

        Returns:
            (B, H, W) anomaly scores.
        """
        results: list[np.ndarray] = []
        for i in range(cubes.shape[0]):
            mask = validity_masks[i] if validity_masks is not None else None
            results.append(self.detect(cubes[i], mask))
        return np.stack(results)

    def fit(self, **kwargs) -> None:
        """
        Optional: configure the detector and learn background statistics.

        Detector-specific parameters (thresholds, hyperparameters) are
        passed as kwargs. The vendable's cube and validity mask are
        available via self._vendable.

        Default is a no-op.
        """
