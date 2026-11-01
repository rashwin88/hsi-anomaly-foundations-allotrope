"""
Pydantic model for a scene read by the generic hyperspectral reader (`sensor_type="generic_hsi"`).

Filled by app/utils/files/generic_hsi_helper.py and read by
GenericHsiDatasetBuilder, scene onboarding (Scene row + pre-flight report) and
the georef resolver. No science here: this is "what we decided the file is".

Gotchas:
  - Every per-band list is in ASCENDING wavelength order, not file order.
    `file_band_index[i]` is the 0-based band in the file for sorted band i.
    Bands dropped while reading the table (duplicate or non-finite
    wavelength) are not in these lists at all.
  - `scale_factor` MULTIPLIES the stored value: reflectance = raw * factor +
    offset. ENVI's `reflectance scale factor` divides, so it arrives here
    inverted.
  - `reflectance` False means the cube is in whatever the file holds (DN,
    radiance); the vendable then carries units="not_reflectance".
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenericHsiMetadata(BaseModel):
    """Everything the builder, onboarding and export need without reading the cube."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(..., description="The picked upload file: .tif, ENVI .hdr, or the HDF5/NetCDF container.")
    data_path: str = Field(..., description="What GDAL opens: the GeoTIFF, the ENVI data file, or an HDF subdataset name.")
    driver: str = Field(..., description="GDAL driver of data_path: GTiff, ENVI, HDF5, netCDF, ...")
    scene_id: str
    acquisition_at: Optional[datetime] = None
    bands: int = Field(..., gt=0, description="Bands kept (sorted, de-duplicated).")
    file_bands: int = Field(..., gt=0, description="Bands GDAL reads in the file.")
    height: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    dtype: str
    file_band_index: List[int]
    wavelengths_nm: List[float]
    fwhm_nm: List[float] = Field(..., description="Every value > 0; see fwhm_source for read vs derived.")
    good_band: List[int]
    fwhm_source: str
    fill_value: Optional[float] = Field(default=None, description="Declared no-data (header / GDAL); None if none.")
    scale_factor: float = 1.0
    scale_offset: float = 0.0
    reflectance: bool = True
    scale_decision: str
    georef_source: Literal["file", "igm", "none"] = "none"
    epsg: Optional[int] = None
    bbox: Optional[List[float]] = Field(default=None, description="WGS-84 [min_lon, min_lat, max_lon, max_lat].")
    platform: Optional[str] = Field(default=None, description="'GHOSt' when the metadata names Orbital Sidekick.")
    preflight: Dict[str, Any] = Field(..., description="The pre-flight report shown at upload and on the scene page.")
