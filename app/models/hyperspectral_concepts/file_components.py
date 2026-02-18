"""
Defines the different hyperspectral file components
"""

from enum import Enum


class HyperspectralFileComponents(str, Enum):
    """
    Defines the different hyperspectral file components
    """

    SWIR_CUBE_DATA = "SWIR_CUBE_DATA"
    VNIR_CUBE_DATA = "VNIR_CUBE_DATA"
    VNIR_PIXEL_ERR_MATRIX = "VNIR_PIXEL_ERR_MATRIX"
    SWIR_PIXEL_ERR_MATRIX = "SWIR_PIXEL_ERR_MATRIX"

    SWIR_CENTRAL_WAVELENGTH_LIST = "SWIR_CENTRAL_WAVELENGTH_LIST"
    SWIR_CENTRAL_WAVELENGTH_FLAGS = "SWIR_CENTRAL_WAVELENGTH_FLAGS"
    SWIR_FWHM_LIST = "SWIR_FWHM_LIST"

    VNIR_CENTRAL_WAVELENGTH_LIST = "VNIR_CENTRAL_WAVELENGTH_LIST"
    VNIR_CENTRAL_WAVELENGTH_FLAGS = "VNIR_CENTRAL_WAVELENGTH_FLAGS"
    VNIR_FWHM_LIST = "VNIR_FWHM_LIST"

    # Scaling factors for conversion
    L2_SCALE_MAX_VNIR = "L2_SCALE_MAX_VNIR"
    L2_SCALE_MIN_VNIR = "L2_SCALE_MIN_VNIR"
    L2_SCALE_MAX_SWIR = "L2_SCALE_MAX_SWIR"
    L2_SCALE_MIN_SWIR = "L2_SCALE_MIN_SWIR"


class ThermalComponents(Enum):
    """
    Defines the different components and properties
    of a thermal dataset
    """

    COORDINATE_REFERENCE_SYSTEM = "COORDINATE_REFERENCE_SYSTEM"
    AREA_OR_POINT = "AREA_OR_POINT"
    BOUNDS = "BOUNDS"
    TRANSFORM = "TRANSFORM"
    WIDTH = "WIDTH"
    HEIGHT = "HEIGHT"
