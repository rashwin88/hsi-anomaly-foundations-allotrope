"""scene_onboard worker handler.

Uses the existing `app.utils.dataset_builder.*` classes — each one is the
integrated entry point for its sensor: it constructs the file helper,
builds a STAC item (bbox + datetime + properties), extracts band
information, and exposes `vend_dataset()` for the normalised cube. We
read all the metadata we need off that one object plus its file_helper.

Flow per job:
    1. Read staging dir under allotrope_data/staging/<job_id>/.
    2. Validate sensor shape (loose) and locate the "source path" the
       dataset builders expect:
           prisma   → first .he5 file
           landsat9 → first .tif file
           enmap    → first directory containing METADATA.XML
    3. Filename-derived sensor_scene_id → pre-flight duplicate check
       (UNIQUE(sensor_type, sensor_scene_id) is the real backstop).
    4. Mint scene_id, plan final raw_path = scenes/<scene_id>/raw/.
    5. shutil.move(staging → raw_path).
    6. Instantiate the per-sensor DatasetBuilder pointing at the final
       raw_path; pull metadata + vend_dataset() for the thumbnail.
    7. Save thumbnail to allotrope_artifacts/scenes/<scene_id>/thumbnail.png.
    8. INSERT the Scene row (single transaction with the runner's
       mark_complete).

Failure path:
    Any raised exception bubbles to the runner; job marked failed. We
    leave the moved files in place (they're under scenes/<scene_id>/raw/)
    so an operator can inspect them.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .visualizations import render_all
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.aviris_ng_dataset_builder import AvirisNGDatasetBuilder
from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
from app.utils.dataset_builder.hotsat_dataset_builder import HotSatDatasetBuilder
from app.utils.dataset_builder.landsat_dataset_builder import LandsatDataBuilder
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.utils.stac.stac_utils.file_name_parsers import FileNameParser

from allotrope.config import settings
from allotrope.models import Job, Scene

logger = logging.getLogger("allotrope.worker.scene_onboard")

# Loose per-sensor shape checks. Real validation happens when the builder
# tries to read the file; this just fails fast on obvious mismatches.
_SENSOR_FILE_HINTS = {
    "prisma": (".he5",),
    "landsat9": (".tif", ".tiff"),
    "enmap": (".tif", ".tiff", ".xml"),
    "aviris_ng": (".bin", ".hdr"),
    "hotsat1": (".tif", ".tiff", ".json"),
}

# --- Source path resolution ------------------------------------------


def _files_in(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _resolve_source_path(sensor_type: str, raw_root: Path) -> Path:
    """Thin alias to the shared resolver.

    The implementation lives in `allotrope.sensors.source_path` so the
    action_run handler can share it when re-vending a Scene from a
    `band_filter_apply` recipe. Single source of truth across onboarding
    and post-onboard recipes.
    """
    from allotrope.sensors.source_path import resolve_source_path

    return resolve_source_path(sensor_type, raw_root)


def _validate_for_sensor(sensor_type: str, files: list[Path]) -> None:
    hints = _SENSOR_FILE_HINTS.get(sensor_type)
    if hints is None:
        raise ValueError(f"unknown sensor_type {sensor_type!r}")
    if not any(p.suffix.lower() in hints for p in files):
        exts = sorted({p.suffix.lower() for p in files})
        raise ValueError(
            f"no files matching {hints} for sensor_type={sensor_type!r}; "
            f"got extensions {exts}"
        )


# --- Filename-only sensor_scene_id (for the duplicate pre-check) -----


def _filename_for_id(sensor_type: str, files: list[Path]) -> str:
    """Pick the file/folder name the StacCreator/builder will identify on.

    Used for the pre-flight duplicate check before we move files. The
    DatasetBuilder will compute this same id internally during onboarding;
    we read it from `stac_item.id` afterwards and assert they match.
    """
    if sensor_type == "prisma":
        for p in files:
            if p.suffix.lower() == ".he5":
                return p.stem
    elif sensor_type == "landsat9":
        for p in files:
            if p.suffix.lower() in (".tif", ".tiff"):
                return p.stem
    elif sensor_type == "enmap":
        for p in files:
            if p.name.upper().endswith("-METADATA.XML"):
                return p.parent.name
    elif sensor_type == "aviris_ng":
        # Scene id is the parent folder name (ang<YYYYMMDD>t<HHMMSS>).
        for p in files:
            name = p.name.lower()
            if name.endswith(".bin") and "_corr_" in name:
                return p.parent.name
    elif sensor_type == "hotsat1":
        # HotSat scene id is the stable prefix of every raster
        # filename, also written into ``metadata.json`` as ``id``.
        # We derive it from the ``_ortho.tiff`` / ``_stacked.tiff``
        # / ``_udm.tiff`` filename rather than opening JSON here so
        # the duplicate pre-check stays a pure stat-only pass.
        for p in files:
            name = p.name
            for suffix in ("_ortho.tiff", "_stacked.tiff", "_udm.tiff"):
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    if base:
                        return base
    raise ValueError(
        f"could not derive filename id for sensor_type={sensor_type!r}"
    )


# --- DatasetBuilder dispatch -----------------------------------------


def _build(sensor_type: str, source_path: Path):
    """Instantiate the right DatasetBuilder for this sensor."""
    cfg = FileSourceConfig(source_path=str(source_path))
    if sensor_type == "prisma":
        return PrismaDatasetBuilder(file_source_configuration=cfg)
    if sensor_type == "landsat9":
        return LandsatDataBuilder(file_source_configuration=cfg)
    if sensor_type == "enmap":
        return EnmapDatasetBuilder(file_source_configuration=cfg)
    if sensor_type == "aviris_ng":
        return AvirisNGDatasetBuilder(file_source_configuration=cfg)
    if sensor_type == "hotsat1":
        return HotSatDatasetBuilder(file_source_configuration=cfg)
    raise ValueError(f"unknown sensor_type {sensor_type!r}")


# Thumbnail + composite rendering moved to allotrope_worker.visualizations.


# --- Metadata extraction from builder --------------------------------


def _band_count_from_builder(sensor_type: str, builder) -> int:
    """Pull band count off the right shape for each sensor.

    PRISMA: builder.band_information is keyed by SpectralFamily; sum
            band counts across families.
    Landsat9: thermal scenes are single-band by convention.
    EnMAP: builder.band_information has bands per family; sum.
    """
    if sensor_type in ("landsat9", "hotsat1"):
        # Single-band thermal scenes by convention.
        return 1
    bi = builder.band_information
    if not bi:
        return 0
    # HyperpectralBandInformation.bands_by_index is the canonical
    # per-family map (band_index → HyperSpectralBand). Sum across
    # families (PRISMA = VNIR + SWIR; EnMAP = whatever the parser
    # produces).
    return sum(len(info.bands_by_index) for info in bi.values())


def _cloud_cover_pct(sensor_type: str, builder) -> float | None:
    """Sensor-specific cloud cover extraction. Best-effort."""
    helper = builder.file_helper
    metadata = getattr(helper, "file_metadata", None)
    if metadata is None:
        return None

    # `0.0` is a legitimate value (cloud-free scene) — keep it, only
    # return None when the field is genuinely missing.
    if sensor_type == "enmap":
        qf = getattr(metadata, "quality_flags", None)
        v = getattr(qf, "cloud_cover", None) if qf is not None else None
        return float(v) if v is not None else None

    if sensor_type == "prisma":
        rm = getattr(metadata, "root_metadata", None)
        if rm is not None:
            attrs = getattr(rm, "file_attributes", None) or {}
            v = attrs.get("Cloudy_pixels_percentage")
            if v is not None:
                return float(v)

    if sensor_type == "hotsat1":
        # HotSat metadata.json carries STAC eo:cloud_cover directly.
        v = getattr(metadata, "cloud_cover_pct", None)
        return float(v) if v is not None else None

    return None


def _native_projection(sensor_type: str, builder) -> str | None:
    """Best-effort CRS string."""
    helper = builder.file_helper
    metadata = getattr(helper, "file_metadata", None)
    if metadata is None:
        return None
    if sensor_type == "prisma":
        rm = getattr(metadata, "root_metadata", None)
        if rm is not None:
            attrs = getattr(rm, "file_attributes", None) or {}
            epsg = attrs.get("Epsg_Code")
            if epsg is not None:
                return f"EPSG:{int(epsg)}"
    if sensor_type == "aviris_ng":
        mi = getattr(metadata, "map_info", None)
        if mi is not None and mi.utm_zone is not None:
            # WGS-84 UTM EPSG: 326XX (North) / 327XX (South).
            hem = (mi.utm_hemisphere or "North").lower()
            base = 32600 if hem.startswith("n") else 32700
            return f"EPSG:{base + int(mi.utm_zone)}"
        return None
    if sensor_type == "hotsat1":
        # HotSat metadata.json STAC item carries proj:epsg directly.
        proj = getattr(metadata, "projection", None)
        if proj is not None and proj.epsg:
            return f"EPSG:{int(proj.epsg)}"
        return None
    # Landsat9 / EnMAP: rasterio has crs on the underlying dataset, but
    # accessing it via the helper is sensor-specific. Skip for v1 — we
    # have bbox in WGS84 which is what the api/UI need.
    return None


# --- Handler ---------------------------------------------------------


def handle_scene_onboard(session: Session, job: Job) -> tuple[str, uuid.UUID]:
    """Worker handler for `scene_onboard` jobs."""
    payload = job.payload or {}
    sensor_type = payload.get("sensor_type")
    staging_rel = payload.get("staging_dir")
    uploaded_by_user_id = payload.get("uploaded_by_user_id")
    if not sensor_type or not staging_rel:
        raise ValueError("payload missing sensor_type or staging_dir")

    data_root = Path(settings.data_dir)
    artifacts_root = Path(settings.artifacts_dir)
    staging_root = data_root / staging_rel
    if not staging_root.is_dir():
        raise FileNotFoundError(f"staging dir not found: {staging_root}")

    files = _files_in(staging_root)
    if not files:
        raise ValueError(f"staging dir is empty: {staging_root}")

    # 1. Loose validation -------------------------------------------------
    _validate_for_sensor(sensor_type, files)

    # 2. Pre-flight duplicate check on the filename-derived id -----------
    rel_files = [p.relative_to(staging_root) for p in files]
    candidate_id = _filename_for_id(sensor_type, rel_files)
    existing = session.scalars(
        select(Scene.id)
        .where(Scene.sensor_type == sensor_type)
        .where(Scene.sensor_scene_id == candidate_id)
    ).first()
    if existing is not None:
        raise ValueError(
            f"scene already exists: sensor_type={sensor_type!r} "
            f"sensor_scene_id={candidate_id!r} (existing scene_{existing})"
        )

    # 3. Mint scene_id + plan final raw_path -----------------------------
    scene_id = uuid.uuid4()
    final_raw_rel = f"scenes/{scene_id}/raw"
    final_raw_abs = data_root / final_raw_rel
    final_raw_abs.parent.mkdir(parents=True, exist_ok=True)
    if final_raw_abs.exists():
        raise FileExistsError(f"raw_path already exists: {final_raw_abs}")

    # 4. Move staging → final --------------------------------------------
    shutil.move(str(staging_root), str(final_raw_abs))

    # 5. Instantiate DatasetBuilder on the final path --------------------
    source_path = _resolve_source_path(sensor_type, final_raw_abs)
    builder = _build(sensor_type, source_path)
    item = builder.stac_item

    # Sanity check — the builder's id should match our pre-flight id.
    if item.id != candidate_id:
        logger.warning(
            "builder id %r differs from pre-flight id %r (sensor=%s)",
            item.id, candidate_id, sensor_type,
        )

    sensor_scene_id = item.id
    bbox = item.bbox or [0.0, 0.0, 0.0, 0.0]
    acq = item.datetime
    if acq is not None and acq.tzinfo is None:
        acq = acq.replace(tzinfo=timezone.utc)

    props = item.properties or {}
    processing_level = props.get("processing:level") or props.get("processing_level")
    product_type = props.get("product_type")

    band_count = _band_count_from_builder(sensor_type, builder)
    cloud_pct = _cloud_cover_pct(sensor_type, builder)
    native_projection = _native_projection(sensor_type, builder)

    # 6. Vend cube + persist + render visualizations --------------------
    # vend_dataset() runs the full normalisation pipeline (DN→reflectance,
    # validity masks, band ordering). Expensive but onboarding is one-time.
    #
    # Persistence first, then visualizations: if anything fails during
    # render we still have the vendable.pkl on disk and an operator can
    # re-run the visualization step out-of-band. Inverse order would
    # mean a vendable-less scene with rendered PNGs, which lies about
    # what's available for Step 12+ action handlers.
    #
    # Persistence rationale: see project memory `project_vendable_lifecycle.md`.
    vendable_rel = f"scenes/{scene_id}/vendable/vendable.pkl"
    thumbnail_rel: str | None = None
    try:
        vendable_abs = data_root / vendable_rel
        vendable_abs.parent.mkdir(parents=True, exist_ok=True)
        # AVIRIS-NG materialises memmap-backed BSQ cube + validity to
        # disk. Park them next to the pickle so all scene-derived
        # vendable artefacts live in one place.
        if sensor_type == "aviris_ng":
            vendable = builder.vend_dataset(
                validity_cube_dir=vendable_abs.parent,
            )
        else:
            vendable = builder.vend_dataset()
        with vendable_abs.open("wb") as f:
            pickle.dump(vendable, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 6b. Build splib07 cache for this sensor's band grid (best-effort).
        # Idempotent — same grid hashes to the same cache filename, so the
        # second PRISMA / EnMAP / AVIRIS-NG scene onboarded is a no-op
        # file-existence check. Failures are non-fatal: spectral matching
        # is an optional downstream feature, and the action surfaces a
        # clear error if the cache is missing at submit time.
        splib_diag: dict = {"status": "skipped:non_hyperspectral"}
        if sensor_type in ("prisma", "enmap", "aviris_ng"):
            try:
                from app.spectral_match import build_cache_for_vendable
                from allotrope.action_types.spectral_library_match import (
                    SpectralLibraryMatchConfig as _SLMConfig,
                )

                # Build the cache with the same chapters / min_coverage the
                # spectral_library_match action defaults to, so the action's
                # cache-key lookup hits this file (the chapters set is part
                # of the cache hash). Pulled from the action's Pydantic
                # field defaults — single source of truth. (Read from
                # model_fields rather than instantiating; the config has
                # other required fields that aren't relevant here.)
                _fields = _SLMConfig.model_fields
                _chapters = _fields["chapters"].default_factory()
                _min_cov = _fields["min_coverage"].default
                splib_diag = build_cache_for_vendable(
                    vendable=vendable,
                    sensor_id=sensor_type,
                    chapters=tuple(_chapters),
                    min_coverage=_min_cov,
                )
                logger.info(
                    "splib07 cache for scene %s: %s", scene_id, splib_diag,
                )
            except Exception as exc:    # noqa: BLE001 — best-effort
                logger.exception(
                    "splib07 cache build failed for scene %s", scene_id,
                )
                splib_diag = {"status": f"failed:{type(exc).__name__}:{exc}"}

        try:
            written = render_all(
                vendable=vendable,
                sensor_type=sensor_type,
                artifacts_root=artifacts_root,
                scene_id_str=str(scene_id),
            )
            # color.png is the canonical thumbnail. If it failed to write
            # we leave thumbnail_path NULL — Library table falls back to
            # the dashed placeholder.
            thumbnail_rel = written.get("color")
        except Exception:
            logger.exception(
                "visualizations render failed for scene %s (vendable saved)", scene_id
            )
            thumbnail_rel = None
    except Exception:
        logger.exception("vend_dataset failed for scene %s", scene_id)
        raise

    # 7. INSERT Scene -----------------------------------------------------
    created_by: uuid.UUID | None
    try:
        created_by = uuid.UUID(uploaded_by_user_id) if uploaded_by_user_id else None
    except (TypeError, ValueError):
        created_by = None

    scene = Scene(
        id=scene_id,
        sensor_type=sensor_type,
        sensor_scene_id=sensor_scene_id,
        name=sensor_scene_id,
        acquisition_at=acq,
        processing_level=processing_level,
        product_type=product_type,
        bbox_min_lon=float(bbox[0]),
        bbox_min_lat=float(bbox[1]),
        bbox_max_lon=float(bbox[2]),
        bbox_max_lat=float(bbox[3]),
        native_projection=native_projection,
        band_count=band_count,
        cloud_cover_pct=cloud_pct,
        valid_pixel_pct=None,  # derivable from validity_cube; deferred
        has_annotations=False,
        extra_metadata={
            "platform": props.get("platform"),
            "stac_properties": dict(props),
            "onboarded_at": datetime.now(timezone.utc).isoformat(),
            "splib_cache": splib_diag,
        },
        raw_path=final_raw_rel,
        vendable_path=vendable_rel,
        thumbnail_path=thumbnail_rel,
        created_by_user_id=created_by,
    )
    session.add(scene)
    session.flush()  # surface UNIQUE violations here, not at commit time

    logger.info(
        "scene_onboard ok: scene=%s sensor=%s id=%s bands=%d cloud=%s",
        scene_id, sensor_type, sensor_scene_id, band_count, cloud_pct,
    )
    return ("scene", scene_id)
