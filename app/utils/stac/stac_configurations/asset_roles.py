"""
Enumerates the different Asset Roles
"""

from enum import Enum


class AssetRole(str, Enum):
    """
    Enumerates the different asset roles
    """

    HYPERSPECTRAL = "hyperspectral"
    THERMAL = "thermal"
