"""
Enumerating all anomaly detection models.
"""

from enum import Enum


class ADModel(str, Enum):
    """
    Registry of anomaly detection models implemented in the pipeline.

    Each entry corresponds to a concrete AnomalyDetector subclass.
    The string value is a stable identifier used for logging,
    serialization, and result provenance.
    """

    # Classical detectors
    RX = "rx"
    LRX = "lrx"
    CRD = "crd"
    MNF_RX = "mnf_rx"
    MNF_LRX = "mnf_lrx"
    THERMAL_GRX = "thermal_grx"

    # Deep learning detectors
    # (add as implemented)
