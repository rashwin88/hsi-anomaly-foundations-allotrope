"""
Defines different units of surface temperature
"""

from enum import Enum


class Temperature(str, Enum):
    """
    Surface Temperature Units
    """

    KELVIN = "KELVIN"
    CELSIUS = "CELSIUS"
    FAHRENHEIT = "FAHRENHEIT"
