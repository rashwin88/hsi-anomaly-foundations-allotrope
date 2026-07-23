"""
Pydantic models for SatVu HotSat-1 L2 Visual scene metadata.

The vendor ships each scene as a folder with a STAC-compliant
``metadata.json`` sidecar plus several GeoTIFF rasters. We parse the
subset of the STAC item we actually consume; the bulk of the STAC
extensions (links, derived_from, software versions) are kept around as
opaque ``extra_properties`` so consumers downstream don't lose
provenance.

This mirrors the role of ``envi_metadata.py`` (AVIRIS-NG) and
``enmap_metadata.py`` (EnMAP):
    Ring 2 = "what does the raw metadata for THIS file format look like
              once parsed?"

The UDM (Usable Data Mask) is a separate raster, not metadata, but
its **bit semantics** are stable per the vendor's user guide and live
here as an enum so the helper / builder both have a single source of
truth for them.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntFlag
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class HotSatUDMFlags(IntFlag):
    """Bit flags inside the per-pixel UDM raster (uint8).

    Source: SatVu Imagery User Guide v1.0 (Dec 2024), Table 4. Values
    are additive — a pixel that is both ``BAD_PIXEL`` and ``NO_DATA``
    holds ``9`` (= 1 + 8).

    A pixel is "valid" iff the UDM byte is ``0`` — no flags set.
    """

    BAD_PIXEL = 1   # unreliable response; interpolated if possible
    CLOUD = 2       # surface fully obscured by cloud
    SATURATED = 4   # detector exceeded its maximum
    NO_DATA = 8     # off-swath or uninterpolable bad pixel


class HotSatRasterBandStats(BaseModel):
    """Per-band statistics that SatVu publishes inside the
    ``raster:bands`` array of each data asset. We treat them as
    optional everywhere because they only need to exist for the data
    assets (visual / experimental_stacked), not the previews / UDM."""

    model_config = ConfigDict(extra="forbid")

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    valid_percent: Optional[float] = None


class HotSatRasterBand(BaseModel):
    """One band's metadata as it appears under ``raster:bands``. For
    HotSat L2 Visual products there is exactly one band per data
    asset (it's a thermal product, not a multispectral one)."""

    model_config = ConfigDict(extra="forbid")

    bits_per_sample: Optional[int] = Field(
        default=None, description="Effective dynamic range. SatVu ships 14 bits in a uint16 container."
    )
    data_type: Optional[str] = Field(
        default=None, description="STAC raster:bands data_type, e.g. 'uint16'."
    )
    nodata: Optional[float] = Field(
        default=None, description="Sentinel value indicating off-swath / fill. SatVu uses 0."
    )
    offset: Optional[float] = Field(
        default=0.0,
        description=(
            "STAC raster:bands offset. SatVu's L2 Visual product ships "
            "offset=0 / scale=1 — i.e. there is no embedded DN-to-"
            "radiance / DN-to-Kelvin transform. Values stay as DN."
        ),
    )
    scale: Optional[float] = Field(default=1.0)
    sampling: Optional[str] = Field(default=None)
    statistics: Optional[HotSatRasterBandStats] = None


class HotSatAsset(BaseModel):
    """One asset (raster, preview, JSON) inside the scene bundle. We
    keep the absolute filesystem path next to the STAC ``href`` so the
    helper can hand both to downstream code without repeated joins."""

    model_config = ConfigDict(extra="forbid")

    href: str = Field(..., description="Relative href as it appears in metadata.json.")
    absolute_path: str = Field(..., description="Resolved absolute path on disk.")
    roles: List[str] = Field(default_factory=list)
    type: Optional[str] = None
    raster_bands: Optional[List[HotSatRasterBand]] = Field(
        default=None,
        description="STAC raster:bands — only present on data assets.",
    )


class HotSatProjection(BaseModel):
    """Projection / pixel-grid info derived from the ``proj:*``
    namespace of the STAC item. HotSat L2 Visual scenes are
    orthorectified to UTM at acquisition time, so this fully describes
    the raster grid."""

    model_config = ConfigDict(extra="forbid")

    epsg: int = Field(..., description="UTM EPSG code, e.g. 32639.")
    shape: Tuple[int, int] = Field(
        ..., description="Raster shape (rows, cols) per the STAC item."
    )
    transform: List[float] = Field(
        ...,
        description=(
            "9-element affine transform from pixel coords to projected "
            "coords (the STAC ``proj:transform`` list)."
        ),
    )
    proj_bbox: Tuple[float, float, float, float] = Field(
        ..., description="(min_e, min_n, max_e, max_n) in projection coords."
    )


class HotSatViewGeometry(BaseModel):
    """Acquisition geometry — view + sun angles. Not load-bearing for
    onboarding but downstream analyses (e.g. shadow-aware anomaly
    scoring) may want it, so we keep it typed."""

    model_config = ConfigDict(extra="forbid")

    view_azimuth: Optional[float] = None
    view_off_nadir: Optional[float] = None
    sun_azimuth: Optional[float] = None
    sun_elevation: Optional[float] = None


class HotSatMetadata(BaseModel):
    """Top-level parsed-metadata container. The helper hands this to
    the dataset builder; the builder reads pixels from the rasters
    referenced through ``primary_asset`` / ``udm_asset``."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(..., description="STAC item ``id``.")
    collection: str = Field(..., description="STAC ``collection`` (e.g. 'visual').")
    acquisition_at: Optional[datetime] = Field(
        default=None, description="Parsed from STAC ``properties.datetime``."
    )
    platform: Optional[str] = None
    gsd_meters: Optional[float] = None
    cloud_cover_pct: Optional[float] = Field(
        default=None, description="From STAC ``properties.eo:cloud_cover``."
    )

    bbox_wgs84: Tuple[float, float, float, float] = Field(
        ..., description="(min_lon, min_lat, max_lon, max_lat) — top-level STAC bbox."
    )
    geometry: Dict[str, Any] = Field(
        ..., description="Top-level STAC ``geometry`` (WGS84 polygon)."
    )

    projection: HotSatProjection
    view: HotSatViewGeometry = Field(default_factory=HotSatViewGeometry)

    # Assets we recognise. Primary asset is the one whose pixels are
    # consumed as the cube; UDM is the quality raster. The vendor
    # currently ships both a single-frame ortho AND a stacked (median
    # of 25 frames) variant — we expose both as separate assets and
    # let the builder pick its preferred one.
    primary_visual_asset: HotSatAsset = Field(
        ..., description="Single-frame ortho ``visual`` asset."
    )
    stacked_visual_asset: Optional[HotSatAsset] = Field(
        default=None,
        description=(
            "25-frame median-stacked ``experimental_stacked`` asset. "
            "Lower noise than the single-frame ortho per the user "
            "guide; the builder prefers it when present."
        ),
    )
    udm_asset: Optional[HotSatAsset] = Field(
        default=None,
        description=(
            "Usable Data Mask raster. uint8 byte-per-pixel with "
            "HotSatUDMFlags semantics."
        ),
    )

    processing_software: Dict[str, str] = Field(
        default_factory=dict,
        description="STAC ``processing:software`` dict (image-processing version etc.).",
    )

    # Anything in properties we didn't model explicitly — preserved
    # verbatim so we don't silently drop provenance fields. The helper
    # populates this from the parsed JSON minus our recognised keys.
    extra_properties: Dict[str, Any] = Field(default_factory=dict)
