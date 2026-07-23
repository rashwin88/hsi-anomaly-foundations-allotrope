"""
Pydantic models for ENVI-format scene metadata (AVIRIS-NG and any other
sensor that ships flat .bin + .hdr pairs).

The ENVI header is plain text — `key = value` and `key = { csv-list }`.
These models capture the typed shape of a parsed header. They are
content-agnostic: parsing fills them, but classification of bands into
spectral families / production of a vendable cube / lat-lon projection
all live in the AVIRIS-NG DatasetBuilder, not here.

Mirrors the pattern of `file_metadata_models.py` (He5Metadata,
TIFMetadata) and `enmap_metadata.py`:
    Ring 2 = "what does the raw metadata for THIS file format look like
              once parsed?"
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ENVIHeaderCore(BaseModel):
    """Universal "how is this binary blob shaped" fields. Every ENVI
    cube carries these; nothing here is sensor-specific."""

    model_config = ConfigDict(extra="forbid")

    samples: int = Field(..., gt=0, description="Pixels per line (across-track).")
    lines: int = Field(..., gt=0, description="Number of lines (along-track).")
    bands: int = Field(..., gt=0, description="Number of spectral bands.")
    header_offset: int = Field(
        default=0, ge=0, description="Bytes to skip at the start of the .bin."
    )
    data_type: int = Field(
        ...,
        description=(
            "ENVI data-type code. 4 = float32, 2 = int16, 1 = uint8, "
            "12 = uint16. AVIRIS-NG uses 4."
        ),
    )
    interleave: Literal["bsq", "bil", "bip"] = Field(
        ...,
        description=(
            "Band layout in the .bin. AVIRIS-NG uses BIL "
            "(band-interleaved-by-line: lines-major, then bands, then samples)."
        ),
    )
    byte_order: Literal[0, 1] = Field(
        default=0, description="0 = little-endian (Intel), 1 = big-endian."
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Free-text description string from the header. JPL sometimes "
            "leaves stale unit strings in here (e.g. claims 'radiance' on a "
            "reflectance product). Don't trust this for unit determination; "
            "the builder samples actual pixel values to disambiguate."
        ),
    )
    wavelength_units: Optional[Literal["Nanometers", "Micrometers"]] = Field(
        default=None,
        description="Wavelength axis units. AVIRIS-NG uses Nanometers.",
    )
    data_ignore_value: Optional[float] = Field(
        default=None,
        description=(
            "Sentinel value marking out-of-swath / invalid pixels. "
            "AVIRIS-NG ships -9999."
        ),
    )


class ENVISpectralAxis(BaseModel):
    """Per-band spectral axis. Parallel lists, all of length `bands`."""

    model_config = ConfigDict(extra="forbid")

    wavelengths: List[float] = Field(
        ...,
        description="Central wavelength per band, in `header.wavelength_units`.",
    )
    fwhm: Optional[List[float]] = Field(
        default=None,
        description="Full-width-half-maximum per band, same units.",
    )
    bbl: Optional[List[int]] = Field(
        default=None,
        description=(
            "Bad-band list. 1 = good band, 0 = band flagged unusable "
            "(typically inside atmospheric water absorption regions)."
        ),
    )

    @model_validator(mode="after")
    def _check_lengths(self) -> "ENVISpectralAxis":
        n = len(self.wavelengths)
        if self.fwhm is not None and len(self.fwhm) != n:
            raise ValueError(
                f"fwhm length {len(self.fwhm)} != wavelengths length {n}"
            )
        if self.bbl is not None and len(self.bbl) != n:
            raise ValueError(
                f"bbl length {len(self.bbl)} != wavelengths length {n}"
            )
        return self


class ENVIMapInfo(BaseModel):
    """Parsed `map info = { ... }` string. AVIRIS-NG flights are
    typically rotated to track the aircraft heading; the `rotation`
    field is the one piece of map metadata that affects how the cube
    projects to lat/lon downstream. PRISMA/EnMAP scenes are usually
    near-axis-aligned so they ignore rotation; AVIRIS-NG cannot."""

    model_config = ConfigDict(extra="forbid")

    projection: str = Field(..., description="e.g. 'UTM'.")
    reference_pixel_x: float = Field(
        ..., description="X pixel coordinate of the reference (usually 1)."
    )
    reference_pixel_y: float = Field(
        ..., description="Y pixel coordinate of the reference (usually 1)."
    )
    reference_easting: float = Field(
        ..., description="Easting (meters) at the reference pixel."
    )
    reference_northing: float = Field(
        ..., description="Northing (meters) at the reference pixel."
    )
    pixel_size_x: float = Field(..., gt=0, description="Pixel size in X (meters).")
    pixel_size_y: float = Field(..., gt=0, description="Pixel size in Y (meters).")
    utm_zone: Optional[int] = Field(default=None, description="UTM zone number.")
    utm_hemisphere: Optional[Literal["North", "South"]] = Field(default=None)
    datum: Optional[str] = Field(default=None, description="e.g. 'WGS-84'.")
    units: Literal["Meters", "Feet"] = Field(default="Meters")
    rotation_degrees: float = Field(
        default=0.0,
        description=(
            "Scene rotation about the reference pixel. AVIRIS-NG flight "
            "lines typically carry a non-zero value (e.g. 90.0); the "
            "vendable schema treats this as an approximation note for "
            "now and defers exact lat/lon projection to a future Action."
        ),
    )


class ENVIInstrumentCal(BaseModel):
    """Documentary / audit-trail fields. JPL stamps the calibration
    file paths it used into every header — the builder doesn't consume
    these but preserves them so we can trace which calibration
    generation produced a given cube."""

    model_config = ConfigDict(extra="forbid")

    crosstrack_scatter_file: Optional[str] = None
    spectral_scatter_file: Optional[str] = None
    flat_field_file: Optional[str] = None
    wavelength_file: Optional[str] = None
    rcc_file: Optional[str] = None
    bad_pixel_map: Optional[str] = None
    correction_factors: Optional[List[float]] = Field(
        default=None,
        description=(
            "Per-band radiometric correction factors. Already applied to "
            "the bytes in the .bin by JPL pre-distribution."
        ),
    )
    smoothing_factors: Optional[List[float]] = Field(default=None)
    radiance_version: Optional[str] = Field(
        default=None,
        description=(
            "JPL processing version tag (e.g. 'v2.0'). Note: present even "
            "on reflectance products that have inherited the radiance "
            "header template. Not a reliable unit indicator."
        ),
    )


class ENVIAuxiliaryProducts(BaseModel):
    """Companion files in the same scene folder. AVIRIS-NG distributes
    a 3-band water-vapor retrieval (`_h2o_`) alongside the primary
    cube. We record paths so a future Action can pick them up; the
    vendable schema doesn't widen to carry them in v1."""

    model_config = ConfigDict(extra="forbid")

    h2o_cube_path: Optional[str] = None
    h2o_header_path: Optional[str] = None
    h2o_band_names: Optional[List[str]] = Field(
        default=None,
        description=(
            "Names of the 3 h2o bands, e.g. ['Column water vapor', "
            "'Liquid absorption path', 'Ice absorption path']."
        ),
    )


class ENVIMetadata(BaseModel):
    """Top-level container. Mirrors EnmapMetadata's role for the EnMAP
    XML sidecar — the typed parsing result the FileHelper hands to the
    DatasetBuilder."""

    model_config = ConfigDict(extra="forbid")

    core: ENVIHeaderCore
    spectral: ENVISpectralAxis
    map_info: Optional[ENVIMapInfo] = None
    calibration: ENVIInstrumentCal = Field(default_factory=ENVIInstrumentCal)
    auxiliary: ENVIAuxiliaryProducts = Field(default_factory=ENVIAuxiliaryProducts)

    primary_cube_path: str = Field(
        ..., description="Absolute path to the primary .bin file."
    )
    primary_header_path: str = Field(
        ..., description="Absolute path to the primary .hdr file."
    )

    # Scene-level fields the helper derives from the filename (the
    # header itself doesn't carry these on AVIRIS-NG).
    scene_id: Optional[str] = Field(
        default=None,
        description=(
            "Scene identifier parsed from the filename "
            "(e.g. 'ang20151219t081738' → flight date+time)."
        ),
    )
    acquisition_at: Optional[datetime] = Field(
        default=None,
        description="UTC flight acquisition time parsed from the scene id.",
    )
