"""
Getting results out of the system: the export bundle and the match probe.

Split out of actions.py. Both endpoints turn a finished analysis into something
consumable outside Allotrope:

    GET  /actions/{id}/spectral_library_match/at_pixel   what material is here?
    POST /actions/{id}/export                            the submission bundle

The export is the product's actual deliverable - a zip of GeoTIFF, Shapefile and
CSV naming every candidate anomaly with its coordinates and, for hyperspectral,
its matched material.

Two things worth knowing before touching this:

  - It runs SYNCHRONOUSLY in the api process, unlike project export, which is a
    queued job. That is why the "lightweight" api image carries geopandas,
    fiona, shapely and rasterio. The heavy imports are kept inside the handler.

  - Vendables hold no spatial reference, so every GeoTIFF an Action wrote has an
    identity transform. CRS and affine are recovered here by re-reading the raw
    scene file through app/georef/. A scene whose georeferencing cannot be
    resolved yields 422 crs_missing rather than a bundle that would be
    disqualified downstream for having no projection.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, Project, Scene
from ._action_common import action_or_404, output_for_action
from .deps import current_user_claims

logger = logging.getLogger("allotrope.api.action_export")

# Same /actions prefix as actions_router, so mounted paths are unchanged.
action_export_router = APIRouter(prefix="/actions", tags=["actions"])

# --- GET /actions/{id}/spectral_library_match/at_pixel --------------
#
# Lightweight probe endpoint used by the spectral_library_match viewer.
# Returns the top-K matches at one (row, col) by filtering the action's
# matches.parquet â€” keeps the frontend free of a parquet reader and
# bounds the response to a handful of rows.


class SpectralMatchAtPixelRow(BaseModel):
    rank: int
    library_ix: int
    material_id: str
    name: str
    chapter: str
    asd_subtype: str | None
    angle_deg: float
    n_bands_used: int


class SpectralMatchAtPixelResponse(BaseModel):
    row: int
    col: int
    matches: list[SpectralMatchAtPixelRow]


@action_export_router.get(
    "/{action_id}/spectral_library_match/at_pixel",
    response_model=SpectralMatchAtPixelResponse,
    summary="Top-K splib07 matches at one pixel of a spectral_library_match Action",
)
def spectral_library_match_at_pixel(
    action_id: str,
    row: int = Query(..., ge=0),
    col: int = Query(..., ge=0),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> SpectralMatchAtPixelResponse:
    """Filter ``matches.parquet`` by (row, col) and return its rows sorted by rank."""
    action = action_or_404(action_id, db)
    if action.type != "spectral_library_match":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="wrong_action_type",
        )
    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )
    parquet_path = (
        Path(settings.artifacts_dir) / output.artifact_path / "matches.parquet"
    )
    if not parquet_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="matches_parquet_missing",
        )

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pylist()
    hits = [r for r in df if r["row"] == row and r["col"] == col]
    hits.sort(key=lambda r: r["rank"])
    return SpectralMatchAtPixelResponse(
        row=row,
        col=col,
        matches=[
            SpectralMatchAtPixelRow(
                rank=r["rank"],
                library_ix=r["library_ix"],
                material_id=r["material_id"],
                name=r["name"],
                chapter=r["chapter"],
                asd_subtype=r.get("asd_subtype"),
                angle_deg=r["angle_deg"],
                n_bands_used=r["n_bands_used"],
            )
            for r in hits
        ],
    )


# --- POST /actions/{id}/export -------------------------------------
#
# Builds a submission-ready zip from the action's outputs and streams it
# back. Two flavours dispatched on action.type:
#   - spectral_library_match â†’ hyper bundle (GeoTIFF + SHP + JSON + CSV)
#   - anomaly_detection_prep â†’ thermal bundle (only when committed)
#
# Submission rules (2026-05-14):
#   * GeoTIFF must have a valid CRS â†’ 422 if missing, no silent identity-fallback.
#   * Filenames/folders must literally contain `hyper` / `thermal`.
#   * Shapefile sidecar set must be complete.
# All handled inside the bundle builders in app/spectral_match/export.py
# and app/anomaly_detection/export.py.


@action_export_router.post(
    "/{action_id}/export",
    summary="Build and stream a submission-ready bundle for an action's outputs",
    response_class=StreamingResponse,
)
def export_action(
    action_id: str,
    confidence_deg: float = Query(
        15.0,
        ge=0.0,
        le=90.0,
        description=(
            "SAM angle (degrees) below which a hyperspectral match is "
            "flagged as `confident`. Ignored for thermal exports."
        ),
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    import io

    from app.spectral_match.export import (
        ExportSpec, MissingCRSError, build_hyper_bundle,
    )
    from app.anomaly_detection.export import (
        ThermalExportSpec, build_thermal_bundle,
    )
    from app.georef import GeorefUnavailable, resolve_scene_georef

    action = action_or_404(action_id, db)
    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="output_not_ready",
        )
    project = db.get(Project, action.project_id)
    if project is None or project.scene_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_bound",
        )
    scene = db.get(Scene, project.scene_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_found",
        )

    artifact_dir = Path(settings.artifacts_dir) / output.artifact_path
    if not artifact_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="artifact_dir_missing",
        )

    action_wire = f"action_{action.id}"
    scene_wire = f"scene_{scene.id}"
    software_version = "allotrope/0.x"   # bumped at release time

    # Resolve scene-level georef from the raw scene file. Required by the
    # submission rules (GeoTIFF with valid CRS); the action's own TIFFs
    # were written with identity transform because the vendable doesn't
    # carry spatial reference (today). One read per export.
    scene_raw_dir = Path(settings.data_dir) / scene.raw_path
    if not scene_raw_dir.is_dir():
        # raw_path may point at a single file; normalise to its parent.
        scene_raw_dir = scene_raw_dir.parent if scene_raw_dir.suffix else scene_raw_dir
    try:
        # Resolve georef using the action's match_map.tif spatial shape
        # for hyper, or the anomaly_mask.tif for thermal.
        import rasterio
        if action.type == "spectral_library_match":
            ref_path = artifact_dir / "match_map.tif"
        else:
            ref_path = artifact_dir / "anomaly_mask.tif"
        if not ref_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"missing_input:{ref_path.name}",
            )
        with rasterio.open(ref_path) as src:
            target_shape = (src.height, src.width)
        transform, crs = resolve_scene_georef(
            scene_dir=scene_raw_dir,
            sensor_type=scene.sensor_type,
            target_shape=target_shape,
        )
    except GeorefUnavailable as exc:
        logger.warning("export refused: georef unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="crs_missing",
        ) from exc

    try:
        if action.type == "spectral_library_match":
            spec = ExportSpec(
                action_id=action_wire,
                scene_id=scene_wire,
                sensor_type=scene.sensor_type,
                artifact_dir=artifact_dir,
                splib_version=None,   # filled in below from summary if available
                software_version=software_version,
                confidence_threshold_deg=confidence_deg,
                override_transform=transform,
                override_crs=crs,
            )
            # If the summary captured the splib version, surface it in the manifest.
            try:
                summary = output.summary or {}
                splib_path = summary.get("splib_cache_path")
                if splib_path:
                    # Cache filename is splib07_<key>.npz; meta-sidecar JSON
                    # has the version string. Cheap lookup; failure tolerable.
                    from pathlib import Path as _P
                    sidecar = _P(splib_path).with_suffix(".json")
                    if sidecar.is_file():
                        import json as _json
                        meta = _json.loads(sidecar.read_text())
                        spec = ExportSpec(
                            action_id=spec.action_id,
                            scene_id=spec.scene_id,
                            sensor_type=spec.sensor_type,
                            artifact_dir=spec.artifact_dir,
                            splib_version=meta.get("splib07_version"),
                            software_version=spec.software_version,
                            confidence_threshold_deg=spec.confidence_threshold_deg,
                            override_transform=spec.override_transform,
                            override_crs=spec.override_crs,
                        )
            except Exception:    # noqa: BLE001 â€” best-effort manifest enrichment
                pass

            zip_bytes, zip_filename = build_hyper_bundle(spec)

        elif action.type == "anomaly_detection_prep":
            # Only exportable AFTER commit â€” the binary anomaly_mask.tif lands
            # at commit time. Refuse if not present.
            if not (artifact_dir / "anomaly_mask.tif").is_file():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="prep_not_committed",
                )
            spec_t = ThermalExportSpec(
                action_id=action_wire,
                scene_id=scene_wire,
                sensor_type=scene.sensor_type,
                artifact_dir=artifact_dir,
                software_version=software_version,
                override_transform=transform,
                override_crs=crs,
            )
            zip_bytes, zip_filename = build_thermal_bundle(spec_t)

        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="export_not_supported_for_action_type",
            )

    except MissingCRSError as exc:
        logger.warning("export refused: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="crs_missing",
        ) from exc
    except FileNotFoundError as exc:
        logger.warning("export missing input: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"missing_input:{exc}",
        ) from exc

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
            "Content-Length": str(len(zip_bytes)),
        },
    )

