from enum import Enum


class SpectralFamily(Enum):
    """
    Defines different spectral families.
    Typically the kind of information we will receive from a hyperspectral file
    """

    SWIR = "SWIR"  # Shorwave InfraRed
    VNIR = "VNIR"  # Visual and Near InfraRed
    PANCHROMATIC = "PANCHROMATIC"
