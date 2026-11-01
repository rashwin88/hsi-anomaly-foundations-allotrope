"""
Pydantic models for a Planet Tanager ortho surface-reflectance scene, HDF-EOS5 or ENVI.

Filled by app/utils/files/tanager_helper.py (TanagerHelper) and read by
TanagerDatasetBuilder and the Tanager georef resolver in app/georef/. Mirrors
envi_metadata.py: "what the raw metadata of this file format looks like once
parsed" - no science here.

Gotchas:
  - Every per-band list is in ASCENDING wavelength order, not file order.
    `file_band_index[i]` is the slice in the file's cube for sorted band i.
    The file's own array is not guaranteed ascending (one scene had band 3
    at 366.38 nm where 386.38 nm belonged), so the helper sorts it once.
  - The grid corners are the OUTER corners of the outer pixels
    (GridOrigin=HE5_HDFE_GD_UL, PixelRegistration=HE5_HDFE_CORNER), so the
    affine origin is the upper-left corner itself, with no half-pixel shift.
"""

from datetime import datetime
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class TanagerGrid(BaseModel):
    """The HDF-EOS5 GRID geometry parsed from `StructMetadata.0`."""

    model_config = ConfigDict(extra="forbid")

    x_dim: int = Field(..., gt=0, description="Columns (W).")
    y_dim: int = Field(..., gt=0, description="Rows (H).")
    upper_left_m: Tuple[float, float] = Field(..., description="(easting, northing) of the outer UL corner, metres.")
    lower_right_m: Tuple[float, float] = Field(..., description="(easting, northing) of the outer LR corner, metres.")
    projection: str = Field(..., description="GCTP projection name, e.g. HE5_GCTP_UTM.")
    zone_code: Optional[int] = Field(default=None, description="UTM zone; negative means southern hemisphere (GCTP).")
    epsg: Optional[int] = Field(default=None, description="EPSG code from the grid group's `epsg_code` attribute.")

    @property
    def pixel_size_m(self) -> Tuple[float, float]:
        """(dx, dy), both positive."""
        dx = (self.lower_right_m[0] - self.upper_left_m[0]) / self.x_dim
        dy = (self.upper_left_m[1] - self.lower_right_m[1]) / self.y_dim
        return dx, dy


class TanagerMetadata(BaseModel):
    """Everything the builder needs without reading the cube."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    scene_id: str = Field(..., description="Planet item id: strip_id without its '_strip' suffix, else the file stem.")
    strip_id: Optional[str] = None
    acquisition_at: Optional[datetime] = Field(default=None, description="UTC, parsed from the YYYYMMDD_HHMMSS id prefix.")
    bands: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    fill_value: float = Field(..., description="`_FillValue` (-9999 in every file seen) or ENVI `data ignore value`; NaN if none.")
    wavelengths_nm: List[float] = Field(..., description="Band centres, ascending.")
    fwhm_nm: List[float] = Field(..., description="Per band, ascending-wavelength order, every value > 0.")
    good_band: List[int] = Field(..., description="1 = good, 0 = flagged by Planet (`good_wavelengths`) or by the helper.")
    file_band_index: List[int] = Field(..., description="Cube slice in the file for each sorted band.")
    grid: Optional[TanagerGrid] = Field(default=None, description="None only for an ENVI scene without a north-up EPSG map info.")
    # ENVI path (tanager_envi.py); the HDF5 path leaves the defaults.
    file_format: Literal["hdf5", "envi"] = "hdf5"
    data_path: Optional[str] = Field(default=None, description="ENVI raw data file; source_path is its .hdr.")
    interleave: Optional[str] = Field(default=None, description="ENVI bsq / bil / bip.")
    reflectance_scale_factor: float = Field(default=1.0, gt=0, description="Stored value / this = 0-1 reflectance.")
    reflectance_scale_source: str = Field(default="file_is_0_1", description="header | detected_0_1 | detected_0_10000 | ...")
    fwhm_derived: bool = Field(default=False, description="True when FWHM came from band spacing, not the file.")
    band_names: Optional[List[str]] = Field(default=None, description="ENVI band names, ascending-wavelength order.")
