"""
What a single hyperspectral band is - wavelength, width, and whether to trust it.

Populated by the dataset builders from sensor metadata (PRISMA's List_Cw_*
attributes, EnMAP's METADATA.XML) and consumed by SpectralBandFilter when it
decides which bands to keep.

The three fields that matter downstream:
  wavelength  centre wavelength, the band's position in the spectrum
  fwhm        full width at half maximum - how wide a slice this band samples.
              The spectral-match resampler needs it to model the sensor's
              response function; without it, lab spectra cannot be convolved
              onto this sensor's grid.
  is_valid    the sensor's own bad-band flag. Stage 1 of the band pipeline.

Gotcha: PRISMA's cube contains a slice for EVERY wavelength including the ones
it flags bad, and those carry fwhm = 0.0. Filter on is_valid before trusting
fwhm.

Note the class name HyperpectralBandInformation is misspelled (missing 's').
The typo is load-bearing - it appears in the DatasetBuilder ABC signature and in
every concrete builder, so fixing it is a rename across the package, not a
one-line edit.
"""

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
