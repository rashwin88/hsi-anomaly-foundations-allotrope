"""
Factory that maps ADModel enum values to AnomalyDetector classes.
"""

from typing import Type

from app.abstract_classes.anomaly_detector import AnomalyDetector
from app.models.ad_models.ad_model import ADModel
from app.detectors.global_rx_detector import GlobalRXDetector
from app.detectors.local_rx_detector import LocalRXDetector
from app.detectors.mnf_compression_detector import MNFCompressionDetector
from app.detectors.mnf_compression_lrx_detector import MNFCompressionLRXDetector
from app.detectors.thermal_grx_detector import ThermalGRXDetector

# Registry: ADModel → concrete AnomalyDetector class.
# Populate as detectors are implemented.
_REGISTRY: dict[ADModel, Type[AnomalyDetector]] = {
    ADModel.RX:  GlobalRXDetector,
    ADModel.LRX: LocalRXDetector,
    ADModel.MNF_RX: MNFCompressionDetector,
    ADModel.MNF_LRX: MNFCompressionLRXDetector,
    ADModel.THERMAL_GRX: ThermalGRXDetector,
    # ADModel.CRD: CRDDetector,
}


def get_detector(model: ADModel) -> Type[AnomalyDetector]:
    """
    Look up the AnomalyDetector class for the given model.

    Returns the class — caller is responsible for instantiation.

    Raises:
        KeyError: if the model has no registered implementation.
    """
    detector_cls = _REGISTRY.get(model)
    if detector_cls is None:
        raise KeyError(
            f"No detector registered for {model.value!r}. "
            f"Available: {[m.value for m in _REGISTRY]}"
        )
    return detector_cls


def register_detector(
    model: ADModel, detector_cls: Type[AnomalyDetector]
) -> None:
    """
    Register a detector class for a given ADModel.

    Call at import time in the detector's module to wire it into the factory.
    """
    if model in _REGISTRY:
        raise ValueError(f"{model.value!r} is already registered.")
    _REGISTRY[model] = detector_cls
