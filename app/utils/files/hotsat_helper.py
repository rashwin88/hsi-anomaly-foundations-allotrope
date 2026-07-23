"""
File helper for SatVu HotSat-1 L2 Visual scenes.

A HotSat scene is a flat folder shipped by SatVu containing roughly:

    metadata.json                                  # STAC item
    <id>_visual_30_hotsat1_ortho.tiff              # single-frame ortho
    <id>_visual_30_hotsat1_stacked.tiff            # 25-frame stacked
    <id>_visual_30_hotsat1_udm.tiff                # usable-data-mask
    <id>_visual_30_hotsat1_thumbnail.png           # preview
    <id>_visual_30_hotsat1_overview.png            # preview
    <id>_visual_30_hotsat1_stackedoverview.png     # preview

This helper does ONLY parsing. It:

  - resolves the on-disk path of each recognised asset
  - parses metadata.json into a typed HotSatMetadata object
  - does NOT open the GeoTIFFs (the dataset builder reads pixels)
  - does NOT decide DN-vs-Kelvin (the L2 Visual product is DN by spec;
    the builder stamps the units field on the vendable)

Mirrors the role of ENVIHelper / EnmapHelper. After this helper runs
we have a typed metadata blob and a set of absolute paths; the
builder takes it from there.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.abstract_classes.file_helper import FileHelper
from app.models.file_processing.hotsat_metadata import (
    HotSatAsset,
    HotSatMetadata,
    HotSatProjection,
    HotSatRasterBand,
    HotSatRasterBandStats,
    HotSatViewGeometry,
)
from app.models.file_processing.sources import FileSourceConfig
from app.models.hyperspectral_concepts.file_components import HotSatFileComponents
from app.models.hyperspectral_concepts.references import ReferenceDefinition


logger = logging.getLogger("HotSatHelper")
logger.setLevel(logging.INFO)


# Filename suffixes for the standard HotSat L2 Visual distribution.
# We accept the scene folder; we glob inside it for these. The vendor
# keeps the scene id as the filename prefix, but we deliberately do
# not bake that prefix into the matching — the suffix alone is the
# stable contract.
_PRIMARY_VISUAL_SUFFIX = "_ortho.tiff"
_STACKED_VISUAL_SUFFIX = "_stacked.tiff"
_UDM_SUFFIX = "_udm.tiff"


# Property keys we lift out of the STAC item explicitly. Anything
# else lands in HotSatMetadata.extra_properties so we don't silently
# drop provenance fields.
_RECOGNISED_PROPERTIES = {
    "datetime",
    "platform",
    "gsd",
    "eo:cloud_cover",
    "proj:bbox",
    "proj:epsg",
    "proj:shape",
    "proj:transform",
    "proj:geometry",
    "view:azimuth",
    "view:off_nadir",
    "view:sun_azimuth",
    "view:sun_elevation",
    "processing:software",
    # Date provenance fields kept but ignored:
    "created",
    "created_at",
}


class HotSatHelper(FileHelper[HotSatMetadata]):
    """
    File helper for SatVu HotSat-1 L2 Visual scene folders.

    ``FileSourceConfig.source_path`` is the path to the scene folder.
    The template is a small dict mapping ``HotSatFileComponents`` to
    ``ReferenceDefinition`` carrying the filename suffix as its
    ``property_name`` — see ``HotSatHelper.default_template()``.
    """

    def __init__(
        self,
        file_source_config: FileSourceConfig,
        template: Optional[Dict[HotSatFileComponents, ReferenceDefinition]] = None,
    ):
        super().__init__(
            file_source_config=file_source_config,
            template=template or HotSatHelper.default_template(),
        )
        self.scene_folder: str = file_source_config.source_path
        if not os.path.isdir(self.scene_folder):
            raise FileNotFoundError(
                f"HotSat scene folder not found: {self.scene_folder}"
            )

        metadata_path = os.path.join(self.scene_folder, "metadata.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"HotSat scene folder missing metadata.json: {metadata_path}"
            )
        self._metadata_path: str = metadata_path
        self._file_metadata: HotSatMetadata = self._construct_metadata_structure()

    # ------------------------------------------------------------------
    # FileHelper contract
    # ------------------------------------------------------------------

    @property
    def file_metadata(self) -> HotSatMetadata:
        return self._file_metadata

    @property
    def template(self) -> Dict[HotSatFileComponents, ReferenceDefinition]:
        return self._template

    def extract_specific_bands(self, *args, **kwargs):  # noqa: ANN201
        """Not used by the HotSat builder — the builder opens the
        ortho / stacked / UDM GeoTIFFs via rasterio directly. Defined
        for parity with the abstract FileHelper contract."""
        raise NotImplementedError(
            "HotSat band extraction is performed by HotSatDatasetBuilder "
            "via rasterio; see the builder for the read path."
        )

    # ------------------------------------------------------------------
    # Public accessors — convenient absolute paths for the builder
    # ------------------------------------------------------------------

    @property
    def primary_visual_path(self) -> str:
        """Absolute path to the single-frame ortho GeoTIFF."""
        return self._file_metadata.primary_visual_asset.absolute_path

    @property
    def stacked_visual_path(self) -> Optional[str]:
        """Absolute path to the 25-frame stacked GeoTIFF, if shipped."""
        a = self._file_metadata.stacked_visual_asset
        return a.absolute_path if a is not None else None

    @property
    def udm_path(self) -> Optional[str]:
        """Absolute path to the UDM raster, if shipped."""
        a = self._file_metadata.udm_asset
        return a.absolute_path if a is not None else None

    @staticmethod
    def default_template() -> Dict[HotSatFileComponents, ReferenceDefinition]:
        """Standard SatVu L2 Visual asset suffixes.

        The helper does the actual resolution by walking the assets
        block of metadata.json; this template exists for API parity
        with the other helpers (and as a documented place to look up
        the canonical suffix-per-component contract).
        """
        from app.models.hyperspectral_concepts.references import ReferenceType

        suffixes = {
            HotSatFileComponents.METADATA_JSON: "metadata.json",
            HotSatFileComponents.PRIMARY_VISUAL: _PRIMARY_VISUAL_SUFFIX,
            HotSatFileComponents.STACKED_VISUAL: _STACKED_VISUAL_SUFFIX,
            HotSatFileComponents.UDM: _UDM_SUFFIX,
        }
        return {
            comp: ReferenceDefinition(
                description=f"HotSat-1 {comp.value} filename suffix",
                reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
                property_name=suffix,
            )
            for comp, suffix in suffixes.items()
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _construct_metadata_structure(self) -> HotSatMetadata:
        with open(self._metadata_path, "r") as f:
            raw = json.load(f)

        scene_id = raw.get("id")
        if not scene_id:
            raise ValueError(
                f"metadata.json missing 'id' field at {self._metadata_path!r}"
            )
        collection = raw.get("collection", "visual")
        bbox = raw.get("bbox")
        geometry = raw.get("geometry") or {}
        if not bbox or len(bbox) != 4:
            raise ValueError(
                "metadata.json is missing a valid 4-element bbox at "
                f"{self._metadata_path!r}"
            )

        props: Dict[str, Any] = raw.get("properties") or {}
        acq = _parse_iso_datetime(props.get("datetime"))
        projection = self._parse_projection(props)
        view = HotSatViewGeometry(
            view_azimuth=_opt_float(props.get("view:azimuth")),
            view_off_nadir=_opt_float(props.get("view:off_nadir")),
            sun_azimuth=_opt_float(props.get("view:sun_azimuth")),
            sun_elevation=_opt_float(props.get("view:sun_elevation")),
        )

        assets_raw: Dict[str, Any] = raw.get("assets") or {}
        assets_parsed: Dict[str, HotSatAsset] = {
            name: self._parse_asset(name, a)
            for name, a in assets_raw.items()
            if isinstance(a, dict) and a.get("href")
        }

        primary = self._pick_asset(
            assets_parsed,
            preferred_keys=("visual",),
            fallback_suffix=_PRIMARY_VISUAL_SUFFIX,
            required=True,
            role="primary_visual",
        )
        stacked = self._pick_asset(
            assets_parsed,
            preferred_keys=("experimental_stacked", "stacked"),
            fallback_suffix=_STACKED_VISUAL_SUFFIX,
            required=False,
            role="stacked_visual",
        )
        udm = self._pick_asset(
            assets_parsed,
            preferred_keys=("udm",),
            fallback_suffix=_UDM_SUFFIX,
            required=False,
            role="udm",
        )

        extra = {k: v for k, v in props.items() if k not in _RECOGNISED_PROPERTIES}
        processing = props.get("processing:software") or {}
        if not isinstance(processing, dict):
            # Vendor sometimes ships a string; keep it under the key
            # but coerce to dict so the typed field stays well-formed.
            processing = {"raw": str(processing)}

        return HotSatMetadata(
            scene_id=scene_id,
            collection=collection,
            acquisition_at=acq,
            platform=props.get("platform"),
            gsd_meters=_opt_float(props.get("gsd")),
            cloud_cover_pct=_opt_float(props.get("eo:cloud_cover")),
            bbox_wgs84=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            geometry=geometry,
            projection=projection,
            view=view,
            primary_visual_asset=primary,
            stacked_visual_asset=stacked,
            udm_asset=udm,
            processing_software={str(k): str(v) for k, v in processing.items()},
            extra_properties=extra,
        )

    def _parse_projection(self, props: Dict[str, Any]) -> HotSatProjection:
        epsg = props.get("proj:epsg")
        shape = props.get("proj:shape")
        transform = props.get("proj:transform")
        proj_bbox = props.get("proj:bbox")
        if not isinstance(epsg, int):
            raise ValueError("metadata.json missing proj:epsg or non-integer")
        if not (isinstance(shape, list) and len(shape) == 2):
            raise ValueError(
                "metadata.json missing or malformed proj:shape (expected 2-element list)"
            )
        if not (isinstance(transform, list) and len(transform) in (6, 9)):
            raise ValueError(
                "metadata.json missing or malformed proj:transform "
                "(expected 6- or 9-element affine list)"
            )
        if not (isinstance(proj_bbox, list) and len(proj_bbox) == 4):
            raise ValueError("metadata.json missing or malformed proj:bbox")
        return HotSatProjection(
            epsg=int(epsg),
            shape=(int(shape[0]), int(shape[1])),
            transform=[float(x) for x in transform],
            proj_bbox=(
                float(proj_bbox[0]),
                float(proj_bbox[1]),
                float(proj_bbox[2]),
                float(proj_bbox[3]),
            ),
        )

    def _parse_asset(self, name: str, raw: Dict[str, Any]) -> HotSatAsset:
        href = str(raw.get("href"))
        abs_path = os.path.normpath(os.path.join(self.scene_folder, href))
        raster_bands = None
        rb = raw.get("raster:bands")
        if isinstance(rb, list) and rb:
            raster_bands = [self._parse_raster_band(b) for b in rb]
        return HotSatAsset(
            href=href,
            absolute_path=abs_path,
            roles=list(raw.get("roles") or []),
            type=raw.get("type"),
            raster_bands=raster_bands,
        )

    def _parse_raster_band(self, raw: Dict[str, Any]) -> HotSatRasterBand:
        stats_raw = raw.get("statistics") or None
        stats = (
            HotSatRasterBandStats(
                minimum=_opt_float(stats_raw.get("minimum")),
                maximum=_opt_float(stats_raw.get("maximum")),
                mean=_opt_float(stats_raw.get("mean")),
                stddev=_opt_float(stats_raw.get("stddev")),
                valid_percent=_opt_float(stats_raw.get("valid_percent")),
            )
            if isinstance(stats_raw, dict)
            else None
        )
        return HotSatRasterBand(
            bits_per_sample=_opt_int(raw.get("bits_per_sample")),
            data_type=raw.get("data_type"),
            nodata=_opt_float(raw.get("nodata")),
            offset=_opt_float(raw.get("offset")) if raw.get("offset") is not None else 0.0,
            scale=_opt_float(raw.get("scale")) if raw.get("scale") is not None else 1.0,
            sampling=raw.get("sampling"),
            statistics=stats,
        )

    def _pick_asset(
        self,
        assets: Dict[str, HotSatAsset],
        *,
        preferred_keys: tuple,
        fallback_suffix: str,
        required: bool,
        role: str,
    ) -> Optional[HotSatAsset]:
        """Pick an asset by STAC key first, then fall back to suffix
        matching against the absolute path. Returns ``None`` and logs a
        warning if not required and not found; raises if required."""
        for key in preferred_keys:
            if key in assets and os.path.isfile(assets[key].absolute_path):
                return assets[key]
        # Suffix fallback — handy if SatVu rotates asset keys between
        # processing-software versions.
        for asset in assets.values():
            if asset.absolute_path.lower().endswith(fallback_suffix.lower()) and os.path.isfile(asset.absolute_path):
                return asset
        if required:
            raise FileNotFoundError(
                f"HotSat scene folder {self.scene_folder!r} is missing the "
                f"required {role!r} asset (keys tried: {preferred_keys!r}, "
                f"suffix tried: {fallback_suffix!r})."
            )
        logger.warning(
            "HotSat scene %s: optional %s asset not found "
            "(keys tried %s, suffix %s).",
            self.scene_folder, role, preferred_keys, fallback_suffix,
        )
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # STAC permits trailing 'Z' for UTC; Python <3.11's fromisoformat
    # doesn't accept it. Normalise.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
