"""
Define different types of transformations
that can be applied to datasets
"""

from enum import Enum


class Transformation(str, Enum):
    """
    Define different types of transformations
    """

    # Transformation that takes digital numbers in the LC09 B10 Thermal
    # Dataset and converts it to a surface temperature.
    LC09_DN_TO_ST = "LC09_DN_TO_ST"
    PRS_L2D_DN_TO_SR = "PRS_L2D_DN_TO_SR"
