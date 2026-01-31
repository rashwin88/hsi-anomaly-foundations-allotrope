"""
Defines the different dataset categories
"""

from enum import Enum


class DatasetCategory(Enum):
    """
    Different dataset categories that can be used in anomaly detection.
    """

    HYPERSPECTRAL = "hyperspectral"
    THERMAL = "thermal"
