"""
Typed shape of everything EnMAP's METADATA.XML sidecar tells us.

Produced by EnmapXmlParser, consumed by EnmapHelper and EnmapDatasetBuilder.
The GeoTIFF holds 224 unlabelled band planes; these models carry the facts that
make those planes interpretable.

    EnmapBandCharacterisation  per band: centre wavelength, FWHM, gain, offset
    EnmapDetectorBoundary      where VNIR ends and SWIR begins
    EnmapQualityFlags          which mask layers this product ships
    EnmapSpatialInfo           footprint and projection

Two fields do more work than their size suggests:

  fwhm      required for spectral library matching. Resampling a lab spectrum
            onto this sensor means convolving it with each band's response
            function, and FWHM is that function's width. It is recorded nowhere
            else, so a scene without it cannot be material-matched.

  detector  VNIR and SWIR overlap in wavelength, so ascending band order alone
  boundary  cannot tell you which detector a band came from. Band filtering
            trims detector EDGES, which requires knowing where the edges are.

Everything here is descriptive - no conversion logic. The gain/offset values are
carried, but applying them is the transformer's job.
"""

from typing import Any, List, Dict, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class EnmapBandCharacterisation(BaseModel):
    """Per-band spectral characterisation from the XML bandCharacterisation section."""

    band_number: int = Field(..., description="1-based band number")
    wavelength_center: float = Field(..., description="Central wavelength in nm")
    fwhm: float = Field(..., description="Full width at half maximum in nm")
    gain: float = Field(..., description="Gain factor for DN to reflectance conversion")
    offset: float = Field(..., description="Offset for DN to reflectance conversion")


class EnmapQualityFlags(BaseModel):
    """Scene-level quality percentages and pixel counts from the XML qualityFlag section."""

    cloud_cover: float
    haze_cover: float
    cirrus_cover: float
    snow_cover: float
    water_cover: float
    cloud_shadow: float
    dead_pixels_vnir: int
    dead_pixels_swir: int


class EnmapDetectorBoundary(BaseModel):
    """Detector boundary information identifying which channels belong to VNIR vs SWIR."""

    num_vnir_bands: int
    num_swir_bands: int
    vnir_expected_channels: List[int] = Field(
        ..., description="1-based channel numbers belonging to the VNIR detector"
    )
    swir_expected_channels: List[int] = Field(
        ..., description="1-based channel numbers belonging to the SWIR detector"
    )


class EnmapSpatialInfo(BaseModel):
    """Spatial extent and dimension information."""

    bounding_polygon: List[Dict[str, Any]] = Field(
        ...,
        description="List of bounding polygon points with 'frame', 'latitude', and 'longitude' keys",
    )
    width: int = Field(..., description="Scene width in pixels")
    height: int = Field(..., description="Scene height in pixels")
    background_value: int = Field(
        ..., description="Nodata/background value in spectral image"
    )


class EnmapMetadata(BaseModel):
    """Complete metadata for an EnMAP L2A scene, parsed from the XML sidecar file."""

    processing_level: str
    mission: str
    satellite_id: str
    tile_id: str
    datatake_id: str
    start_time: datetime
    stop_time: datetime
    band_characterisation: List[EnmapBandCharacterisation]
    quality_flags: EnmapQualityFlags
    detector_boundary: EnmapDetectorBoundary
    spatial_info: EnmapSpatialInfo
