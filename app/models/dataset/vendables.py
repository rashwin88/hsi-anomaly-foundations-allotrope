"""
The VendableDataset types - the currency of the whole system.

A "vendable" is a normalised cube plus its validity masks, produced by a
DatasetBuilder from a raw scene file. Every detector, model and Action consumes
one of these; nothing downstream touches raw sensor formats. Three variants:
hyperspectral (PRISMA, AVIRIS-NG), EnMAP hyperspectral (adds five quality
masks), and thermal (Landsat 9, HotSat-1, in Celsius).

Also defines BandFilterConfig, which drives the 8-stage band pipeline, and
DEFAULT_COMMON_WAVELENGTH_GRID - the 165-band, 10 nm, 460-2450 nm grid that
every hyperspectral sensor is resampled onto. That shared grid is what lets one
model train on shards mixing PRISMA and EnMAP.

Two things that will surprise you:

  1. Vendables carry NO spatial reference. Every GeoTIFF an Action writes has an
     identity transform; CRS and affine are recovered at export time by
     re-reading the raw file (app/georef/). Do not add a transform field here
     expecting it to be populated.

  2. These classes are PICKLED to disk - band_filter_apply writes
     filtered_vendable.pkl and the api unpickles it. Pickle stores the class
     path, so renaming or moving any class in this module breaks every stored
     vendable on disk. Treat the class names and module path as a wire format.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, SkipValidation
import numpy as np

from app.models.hyperspectral_concepts.spectral_family import SpectralFamily


def build_common_wavelength_grid(
    start_nm: float = 460.0,
    end_nm: float = 2450.0,
    spacing_nm: float = 10.0,
    exclusion_ranges: list = None,
) -> np.ndarray:
    """
    Builds a common wavelength grid that skips atmospheric absorption windows.

    The grid only contains wavelengths where real sensor data exists,
    avoiding PCHIP interpolation across gaps where bands were deliberately
    removed (water vapor, CO2 absorption, etc.).

    Returns a 1-D float64 array of wavelengths in nm, ascending.
    """
    if exclusion_ranges is None:
        exclusion_ranges = [
            (0, 450),       # low SNR
            (912, 978),     # water vapor
            (1131, 1152),   # water vapor
            (1350, 1450),   # water vapor absorption
            (1800, 1950),   # water vapor + CO2
        ]
    full_grid = np.arange(start_nm, end_nm + spacing_nm / 2, spacing_nm)
    mask = np.ones(len(full_grid), dtype=bool)
    for lo, hi in exclusion_ranges:
        mask &= ~((full_grid >= lo) & (full_grid <= hi))
    return full_grid[mask]


# 165 bands across 5 clean spectral segments, 10nm spacing.
# Respects the coarsest sensor resolution (PRISMA ~12nm) and excludes
# atmospheric absorption windows to avoid fabricating spectral data.
DEFAULT_COMMON_WAVELENGTH_GRID = build_common_wavelength_grid()


class BandFilterConfig(BaseModel):
    """
    Configuration for spectral band filtering applied during dataset vending.

    Controls four filtering stages:
    1. Sensor-flagged bad bands (always applied)
    2. Wavelength exclusion ranges (atmospheric absorption windows)
    3. Edge band trimming per detector
    4. Coverage-aware pruning by valid pixel percentage
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    exclusion_ranges: List[Tuple[float, float]] = Field(
        default=[
            (0, 450),       # low SNR, detector noise
            (912, 978),     # water vapor
            (1131, 1152),   # water vapor
            (1350, 1450),   # water vapor absorption
            (1800, 1950),   # water vapor + CO2 absorption
        ],
        description="Wavelength ranges (nm) to exclude. Each tuple is (low, high) inclusive.",
    )

    edge_bands_to_trim: int = Field(
        default=3,
        ge=0,
        description="Number of bands to trim from each end of each detector (VNIR, SWIR).",
    )

    min_valid_pixel_pct: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Minimum valid pixel percentage to keep a band (0.0–100.0).",
    )

    max_invalid_voxel_fraction: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum fraction of invalid/missing voxels a pixel may have across "
            "all bands before the entire pixel column is marked invalid. "
            "E.g. 0.4 means pixels with >40%% invalid bands are fully invalidated."
        ),
    )

    quality_masks_to_apply: List[str] = Field(
        default=["cloud", "cloud_shadow", "haze"],
        description=(
            "Quality mask layers to use for spatial invalidation (EnMAP only). "
            "Pixels flagged (>0) by any of these masks have their entire validity "
            "column zeroed out. Applied before voxel-fraction spatial masking. "
            "Available masks: cloud, cirrus, haze, cloud_shadow, snow."
        ),
    )

    common_wavelength_grid: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description=(
            "Target wavelength grid (1-D array, nm, ascending) to resample all "
            "sensors onto. When set, the vendable cube is spectrally resampled "
            "so that every sensor produces identical band count and wavelengths. "
            "Use DEFAULT_COMMON_WAVELENGTH_GRID for the standard 10nm grid "
            "(450–2450nm, 201 bands). None disables resampling."
        ),
    )


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

    custom_quality_mask: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description=(
            "QA_PIXEL-derived quality mask. 0 = invalid (fill, cloud, dilated cloud, "
            "cloud shadow, or medium/high cirrus), 1 = valid. "
            "Must be multiplied with pure_validity_mask before use."
        ),
    )

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

    quality_classes_mask: Optional[SkipValidation[np.ndarray]] = Field(
        default=None,
        description=(
            "Land/water classification from QL_QUALITY_CLASSES. Unlike the "
            "masks above this is not binary — per the EnMAP product "
            "specification (EN-PCV-ICD-2009-2): 0 = background inside the "
            "scene (missing values or errors), 1 = land, 2 = water, "
            "3 = background outside the scene (map/sensor geometry "
            "resampling artefact, consistently ~25% of every raster). "
            "Only 1 and 2 are real surface classes; 0 and 3 are both "
            "'no data' and must be excluded from any training loss."
        ),
    )

    vnir_channel_indices: List[int] = Field(
        ..., description="1-based channel numbers belonging to the VNIR detector"
    )

    swir_channel_indices: List[int] = Field(
        ..., description="1-based channel numbers belonging to the SWIR detector"
    )
