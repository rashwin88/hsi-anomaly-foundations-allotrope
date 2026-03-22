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

    band_validity_by_position : List[int] = Field(
        ..., description="The band validity order."
    )

    band_level_validity_score: List[float] = Field(
        ...,
        description="Percentage of valid pixels in each band (0.0 to 100.0), derived from the validity cube.",
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
        description="The cloud mask that comes from the provider of the data 1 means cloud 0 means no cloud",
    )

    provider_water_presence: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description="The water mask that comes from the provider of the data 1 means water 0 means no water",
    )

    provider_snow_presence: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description="The snow mask that comes from the provider of the data 1 means snow 0 means no snow",
    )


class VendableEnmapHyperspectralDataset(BaseModel):
    """
    Vendable dataset for EnMAP L2A hyperspectral data.
    Includes separate quality layer masks (cloud, cirrus, haze, cloud shadow, snow)
    and detector boundary info for VNIR/SWIR channel tracking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    normalized_hyperspectral_cube: SkipValidation[np.ndarray] = Field(
        ..., description="224-band surface reflectance cube in BSQ (C, H, W)"
    )

    validity_cube: SkipValidation[np.ndarray] = Field(
        ...,
        description="Combined validity mask (pixel mask AND nodata). 1=valid, 0=invalid.",
    )

    spectral_family_order: List[SpectralFamily] = Field(
        ..., description="Per-band spectral family assignment (VNIR or SWIR)"
    )

    band_cw_order: List[float] = Field(
        ..., description="Center wavelength of each band in order (nm)"
    )

    band_fwhm_order: List[float] = Field(
        default=[], description="FWHM of each band in order (nm)"
    )

    band_validity_by_position: List[int] = Field(
        ..., description="1 if band is valid, 0 otherwise"
    )

    band_level_validity_score: List[float] = Field(
        ...,
        description="Percentage of valid pixels in each band (0.0 to 100.0), derived from the validity cube.",
    )

    # EnMAP quality layer masks (each is H x W, uint8)
    cloud_mask: Optional[SkipValidation[np.ndarray]] = Field(
        default=None, description="Cloud mask from QL_QUALITY_CLOUD"
    )

    cirrus_mask: Optional[SkipValidation[np.ndarray]] = Field(
        default=None, description="Cirrus mask from QL_QUALITY_CIRRUS"
    )

    haze_mask: Optional[SkipValidation[np.ndarray]] = Field(
        default=None, description="Haze mask from QL_QUALITY_HAZE"
    )

    cloud_shadow_mask: Optional[SkipValidation[np.ndarray]] = Field(
        default=None, description="Cloud shadow mask from QL_QUALITY_CLOUDSHADOW"
    )

    snow_mask: Optional[SkipValidation[np.ndarray]] = Field(
        default=None, description="Snow mask from QL_QUALITY_SNOW"
    )

    vnir_channel_indices: List[int] = Field(
        ..., description="1-based channel numbers belonging to the VNIR detector"
    )

    swir_channel_indices: List[int] = Field(
        ..., description="1-based channel numbers belonging to the SWIR detector"
    )
