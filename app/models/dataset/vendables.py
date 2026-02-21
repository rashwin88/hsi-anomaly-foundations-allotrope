"""
Defines vendable datasets for each dataset builder
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, SkipValidation
import numpy as np

from app.models.hyperspectral_concepts.spectral_family import SpectralFamily


class VendableHyperspectralDataset(BaseModel):
    """
    Models a vendable Hyperspectral Dataset that can be used by downstream applications
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    normalized_hyperspectral_cube: SkipValidation[np.ndarray] = Field(
        ..., description="A fully normalized hyperspectral cube"
    )

    validity_cube: SkipValidation[np.ndarray] = Field(
        ...,
        description="The full validity cube. If a band is is in valid then every pixel in that band must be invalid.",
    )

    spectral_family_order: List[SpectralFamily] = Field(
        ...,
        description="An ordered list of spectral families to which each band belongs",
    )

    band_cw_order: List[float] = Field(
        ...,
        description="A list which has the CW of each band in order of occurence in the cube.",
    )

    band_fwhm_order: Optional[List[float]] = Field(
        default=[], description="An ordered list of FWHM of the wavelengths"
    )


class VendableThermalDataset(BaseModel):
    """
    Defines a vendable dataset for Landsat
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    normalized_thermal_cube: SkipValidation[np.ndarray] = Field(
        ...,
        description="A fully normalized thermal cube with surface temperatures in celsius",
    )

    validity_cube: SkipValidation[np.ndarray] = Field(
        ...,
        description="The full validity cube. Here validity refers to the presence or absence of clouds.",
    )

    cloud_mask: Optional[SkipValidation[np.ndarray]] = Field(
        ..., description="The pure cloud mask where 0 means cloud and 1 means clear."
    )

    pure_validity_mask: Optional[SkipValidation[np.ndarray]] = Field(
        ..., description="Pure validity mask"
    )

    #### Provider specific data - we dont have access to this usually (use with care)

    provider_cloud_presence: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description="The cloud mask that comes from the provider of the data 1 means no cloud 0 means cloud",
    )

    provider_water_presence: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description="The water mask that comes from the provider of the data 1 means no water 0 means water",
    )

    provider_snow_presence: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description="The snow mask that comes from the provider of the data 1 means no snow 0 means snow",
    )
