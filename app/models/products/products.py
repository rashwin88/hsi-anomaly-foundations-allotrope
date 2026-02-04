"""
Different products that we are dealing with
"""

from enum import Enum


class Product(Enum):
    """
    Canonical products that we will be using for anomaly detection
    """

    PRISMA = "prisma"
    ENMAP = "EnMap"
    LANDSAT = "LandSat"
