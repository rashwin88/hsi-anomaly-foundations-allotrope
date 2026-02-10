"""
Enumerating all base models used in the anomaly detection
pipeline
"""

from enum import Enum


class BaseModel(str, Enum):
    """
    All base models used in the anomaly detection pipeline
    """

    # Cloud masking from B10 data
    ALLOTROPE_B10_ADAPTIVE_CLOUD_MASKER = "allotrope-b10-adaptive-cloud-masker"
