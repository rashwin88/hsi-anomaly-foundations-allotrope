"""
Defines the different processing levels
"""

from enum import Enum


class ProcessingLevels(str, Enum):
    """
    Defines the different processing levels
    """

    L2SP = "L2SP"
    L2D = "L2D"
