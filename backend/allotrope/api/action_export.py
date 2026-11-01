"""
Getting results out of the system: the export bundle and the match probe.

Split out of actions.py. Both endpoints turn a finished analysis into something
consumable outside Allotrope:

    GET  /actions/{id}/spectral_library_match/at_pixel       what material is here?
    GET  /actions/{id}/spectral_library_match/object_coords  where is each namer object?
    POST /actions/{id}/export[?subset=manmade|natural]       the submission bundle
    POST /actions/{id}/export?classes=...                    the class-driven pack (action_class_pack.py)

The export is the product's actual deliverable - a zip of GeoTIFF, Shapefile and
CSV naming every candidate anomaly with its coordinates and, for hyperspectral,
its matched material. The bundle layout follows what the action wrote (SAM only,
family, or the spectral namer's schema /3 with its level-1 GeoTIFF); this endpoint
only resolves the scene georef that every raster in it is written with. ``subset`` cuts a
namer bundle down to the man-made (or natural) objects' pixels, with a boolean 0/1 GeoTIFF:
the shape a submission takes.

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

import dataclasses
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, Project, Scene
from ..version import __version__
from ._action_common import action_or_404, output_for_action
from .action_class_pack import class_pack_response, parse_tokens
from .deps import current_user_claims

logger = logging.getLogger("allotrope.api.action_export")

# Same /actions prefix as actions_router, so mounted paths are unchanged.
action_export_router = APIRouter(prefix="/actions", tags=["actions"])

# --- GET /actions/{id}/spectral_library_match/at_pixel --------------
#
# Lightweight probe endpoint used by the spectral_library_match viewer.
# Returns the top-K matches at one (row, col) by filtering the action's
# matches.parquet — keeps the frontend free of a parquet reader and
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
#   - spectral_library_match → hyper bundle (GeoTIFF + SHP + JSON + CSV)
#   - anomaly_detection_prep → thermal bundle (only when committed)
#
# Submission rules (2026-05-14):
#   * GeoTIFF must have a valid CRS → 422 if missing, no silent identity-fallback.
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
            "flagged as `confident`. Only for SAM-only actions (bundle "
            "schema /1); family-matched actions carry `named` instead. "
            "Ignored for thermal exports."
        ),
    ),
    subset: Literal["all", "manmade", "natural"] = Query(
        "all",
        description=(
            "spectral_library_match with namer results only: keep the pixels of "
            "objects the spectral namer called man-made (or natural), with a boolean "
            "0/1 GeoTIFF of them in place of the coded rasters. `all` is the full bundle."
        ),
    ),
    classes: str | None = Query(
        None,
        description=(
            "spectral_library_match only: the class-driven pack - comma list of natural, manmade, "
            "or man-made material class keys (namer/classes.py). Replaces subset."
        ),
    ),
    confident_only: bool = Query(False, description="With classes: drop tentative man-made objects."),
    manmade_p: float | None = Query(
        None, ge=0.0, le=1.0,
        description="With classes: the man-made cut on an object's mean P(man-made); default the action's 0.5.",
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    import io

    from app.spectral_match.export import (
        ExportSpec, MissingCRSError, SubsetUnavailableError, build_hyper_bundle,
    )
    from app.anomaly_detection.export import (
        ThermalExportSpec, build_thermal_bundle,
    )

    action, output, scene, artifact_dir = _export_context(action_id, db)
    software_version = f"allotrope/{__version__}"
    if classes is None and (confident_only or manmade_p is not None):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="confident_only and manmade_p apply to a class export; give classes= too")
    if classes is not None:
        if subset != "all":
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="give classes= or subset=, not both")
        parse_tokens(classes)                       # 422 on a bad token before the georef read
        transform, crs = _scene_georef(action, scene, artifact_dir)
        return class_pack_response(action, scene, artifact_dir, db, transform=transform, crs=crs,
                                   classes=classes, confident_only=confident_only, manmade_p=manmade_p,
                                   software_version=software_version)
    check_subset(action.type, artifact_dir, subset)

    action_wire = f"action_{action.id}"
    scene_wire = f"scene_{scene.id}"

    transform, crs = _scene_georef(action, scene, artifact_dir)

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
                subset=subset,
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
                        spec = dataclasses.replace(
                            spec, splib_version=meta.get("splib07_version"))
            except Exception:    # noqa: BLE001 — best-effort manifest enrichment
                pass

            zip_bytes, zip_filename = build_hyper_bundle(spec)

        elif action.type == "anomaly_detection_prep":
            # Only exportable AFTER commit — the binary anomaly_mask.tif lands
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
    except SubsetUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc),
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



# --- GET /actions/{id}/spectral_library_match/object_coords --------
#
# Centroid lat / lon of every spectral-namer object, so the match viewer can show where an
# object is. The browser has no projection library and the action's rasters carry an identity
# transform, so the conversion runs here through the same georef path the export uses.


class NamerObjectCoordsResponse(BaseModel):
    crs: str
    coords: dict[str, list[float]]          # obj_id -> [lat, lon], EPSG:4326


@action_export_router.get(
    "/{action_id}/spectral_library_match/object_coords",
    response_model=NamerObjectCoordsResponse,
    summary="Centroid lat / lon of each spectral-namer object of a spectral_library_match Action",
)
def spectral_namer_object_coords(
    action_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> NamerObjectCoordsResponse:
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.warp import transform as warp

    action, _output, scene, artifact_dir = _export_context(action_id, db)
    objects_path = artifact_dir / "namer_objects.parquet"
    if action.type != "spectral_library_match" or not objects_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="namer_objects_missing",
        )
    transform, crs = _scene_georef(action, scene, artifact_dir)
    objs = pq.read_table(objects_path, columns=["obj_id", "row", "col"]).to_pydict()
    if not objs["obj_id"]:
        return NamerObjectCoordsResponse(crs=str(crs), coords={})
    xs, ys = zip(*(rasterio.transform.xy(transform, float(r), float(c), offset="center")
                   for r, c in zip(objs["row"], objs["col"])))
    lons, lats = warp(crs, "EPSG:4326", list(xs), list(ys))
    return NamerObjectCoordsResponse(
        crs=str(crs),
        coords={str(int(i)): [round(la, 6), round(lo, 6)]
                for i, la, lo in zip(objs["obj_id"], lats, lons)},
    )


# --- shared by the export and the coords endpoint -------------------


def _export_context(action_id: str, db: Session):
    """(action, output, scene, artifact_dir) of a finished action, or the 404 that says why."""
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
    return action, output, scene, artifact_dir


def _scene_georef(action: Action, scene: Scene, artifact_dir: Path):
    """(transform, crs) of the scene, re-read from the raw file (the action's TIFFs carry an
    identity transform). The target shape is the action's match_map.tif (hyper) or
    anomaly_mask.tif (thermal). 422 crs_missing when the scene has no georeferencing."""
    import rasterio

    from app.georef import GeorefUnavailable, resolve_scene_georef

    scene_raw_dir = Path(settings.data_dir) / scene.raw_path
    if not scene_raw_dir.is_dir():
        # raw_path may point at a single file; normalise to its parent.
        scene_raw_dir = scene_raw_dir.parent if scene_raw_dir.suffix else scene_raw_dir
    ref_name = "match_map.tif" if action.type == "spectral_library_match" else "anomaly_mask.tif"
    ref_path = artifact_dir / ref_name
    if not ref_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"missing_input:{ref_path.name}",
        )
    try:
        with rasterio.open(ref_path) as src:
            target_shape = (src.height, src.width)
        return resolve_scene_georef(
            scene_dir=scene_raw_dir,
            sensor_type=scene.sensor_type,
            target_shape=target_shape,
        )
    except GeorefUnavailable as exc:
        logger.warning("georef unavailable: %s", exc)
        # The detail stays the code the UI matches on; the reason (e.g. generic_hsi's
        # "this file has no map coordinates") rides in a header for API callers.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="crs_missing",
            headers={"X-Allotrope-Reason": str(exc).encode("ascii", "replace").decode()[:300]},
        ) from exc


# The files a man-made / natural subset is cut from (written by _spectral_namer_run).
_NAMER_EXPORT_FILES = ("namer_pixels.parquet", "namer_objects.parquet", "namer_level1_map.tif")


def check_subset(action_type: str, artifact_dir: Path, subset: str) -> None:
    """422 unless ``subset`` can be honoured: "all" always; man-made / natural only on a
    spectral_library_match whose spectral namer ran (its namer_* files exist)."""
    if subset == "all":
        return
    if action_type != "spectral_library_match":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"subset={subset} applies to spectral_library_match exports only",
        )
    if not all((artifact_dir / f).is_file() for f in _NAMER_EXPORT_FILES):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(f"subset={subset} needs spectral-namer results, and this action has "
                    "none (it ran without the namer, or the namer failed); export it "
                    "without a subset"),
        )
