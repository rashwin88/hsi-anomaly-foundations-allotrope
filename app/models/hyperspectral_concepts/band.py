from typing import Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field


class WavelengthMeasurementUnits(Enum):
    """
    Different units in which wavelengths are measured.
    """

    NANO_METERS = "nm"
    MICRO_METERS = "um"


class HyperSpectralBand(BaseModel):
    """
    Defines a hyperspectral band.
    This is basically a single slice in any kind of hyperspectral cube
    """

    wavelength: float = Field(..., description="The spectral wavelength of the band")

    wavelength_measurement_unit: WavelengthMeasurementUnits = Field(
        ..., description="The unit of measurement of the wavelength"
    )

    band_index: int = Field(..., description="The numerical index of the band")

    full_width_at_half_maximum: Optional[float] = Field(
        ..., description="The FWHM of the band"
    )

    is_valid: Optional[bool] = Field(
        default=False, description=" Whether the band in the dataset is valid or not."
    )


class HyperpectralBandInformation(BaseModel):
    """
    Collected Hyperspectral band information at the level of a given spectral family.
    Defined for each spectral family
    """

    bands_by_wavelength: Dict[float, HyperSpectralBand] = Field(
        ...,
        description="The mapping between wavelengths and the respective band information.",
    )

    bands_by_index: Dict[int, HyperSpectralBand] = Field(
        ...,
        description="The mapping between a band index and the respective band information",
    )
