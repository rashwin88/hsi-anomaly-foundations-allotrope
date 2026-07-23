"""Annotation endpoints (Step 9 — annotation attach).

Routes:

    POST   /scenes/{scene_id}/annotations
        Multipart upload (file + name + optional description). Stages
        the file to `allotrope_data/staging/<job_id>/` and enqueues an
        `annotation_attach` worker job. Returns 202 + the job_id so the
        client can poll progress.

    GET    /scenes/{scene_id}/annotations
        List Annotation rows attached to the scene. Convention-based
        `has_overlay` flag: true when the worker wrote an overlay PNG.

    GET    /scenes/{scene_id}/annotations/{ann_id}/overlay
        Stream the pre-rendered RGBA overlay PNG. Frontend layers this
        over the active composite via absolute positioning.

    DELETE /scenes/{scene_id}/annotations/{ann_id}
        Synchronous delete (per abstractions-spec § 5.3): removes the
        row, files on disk, and overlay PNG. Recomputes
        `scenes.has_annotations` in the same transaction (CC-12).

Sequence diagram: final design/diagrams/annotation-attach.drawio (Step 9e)
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Annotation, Job, Scene
from .deps import current_user_claims
from .wireformat import parse_prefixed_id, to_prefixed

# The annotation type registry is shared between api and worker. The
# heavy worker deps (rasterio / scipy / Pillow) are imported lazily
# inside the per-type module's render functions so the api process
# only pays for numpy at import time.
from ..annotation_types import (
    get_spec,
    infer_kind_from_filename,
    public_catalog,
    supported_kinds,
)

logger = logging.getLogger("allotrope.api.annotations")

router = APIRouter(prefix="/scenes", tags=["annotations"])

# Companion router for the global annotation-type catalogue. Lives at
# `/annotation-types` so the frontend can populate the type picker
# from the same source of truth the api uses for validation. Mounted
# alongside the scene-scoped router in main.py. Route handler defined
# below the schemas so the response_model forward ref resolves cleanly.
catalog_router = APIRouter(tags=["annotations"])

# Accept the same chunk size + size cap as scene onboarding.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


# --- Schemas ----------------------------------------------------------


class AnnotationPublic(BaseModel):
    id: str                              # annotation_<uuid>
    scene_id: str                        # scene_<uuid>
    type: str
    name: str
    description: str | None
    file_path: str                       # relative to allotrope_data
    metadata: dict[str, Any]
    has_overlay: bool                    # true when overlay.png exists in artifacts
    created_at: datetime
    created_by_user_id: str | None       # user_<uuid> when set

    @classmethod
    def from_orm_annotation(cls, a: Annotation, has_overlay: bool) -> "AnnotationPublic":
        return cls(
            id=f"annotation_{a.id}",
            scene_id=f"scene_{a.scene_id}",
            type=a.type,
            name=a.name,
            description=a.description,
            file_path=a.file_path,
            metadata=a.extra_metadata or {},
            has_overlay=has_overlay,
            created_at=a.created_at,
            created_by_user_id=to_prefixed("user", a.created_by_user_id),
        )


class AnnotationsList(BaseModel):
    items: list[AnnotationPublic]


class AttachAccepted(BaseModel):
    """Mirror of OnboardAccepted — async job kicked off, client polls."""

    job_id: str                          # job_<uuid>
    annotation_name: str


class AnnotationTypeCatalogItem(BaseModel):
    kind: str
    label: str
    accepted_extensions: list[str]


class AnnotationTypeCatalog(BaseModel):
    items: list[AnnotationTypeCatalogItem]


@catalog_router.get(
    "/annotation-types",
    response_model=AnnotationTypeCatalog,
    summary="List supported annotation types (registry catalogue)",
)
def list_annotation_types(
    _claims: Claims = Depends(current_user_claims),
) -> AnnotationTypeCatalog:
    return AnnotationTypeCatalog(
        items=[AnnotationTypeCatalogItem(**item) for item in public_catalog()],
    )


# --- Helpers ----------------------------------------------------------


def _scene_or_404(scene_id_wire: str, db: Session) -> Scene:
    raw_id = parse_prefixed_id("scene", scene_id_wire)
    scene = db.get(Scene, raw_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_found"
        )
    return scene


def _annotation_or_404(scene: Scene, ann_id_wire: str, db: Session) -> Annotation:
    raw_id = parse_prefixed_id("annotation", ann_id_wire)
    ann = db.get(Annotation, raw_id)
    if ann is None or ann.scene_id != scene.id:
        # Treat "annotation under a different scene" as 404 rather than
        # 403 — don't leak which ids exist.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="annotation_not_found"
        )
    return ann


def _overlay_path(scene_id: uuid.UUID, ann_id: uuid.UUID) -> Path:
    """Convention path for the worker-rendered overlay PNG."""
    return (
        Path(settings.artifacts_dir)
        / "scenes"
        / str(scene_id)
        / "annotations"
        / str(ann_id)
        / "overlay.png"
    )


def _confined(root: Path, candidate: Path) -> Path:
    """Defence-in-depth: confine `candidate` to `root`."""
    full = candidate.resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="not_found"
        ) from exc
    return full


# --- POST /scenes/{id}/annotations ------------------------------------


@router.post(
    "/{scene_id}/annotations",
    response_model=AttachAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Attach an annotation file (queues an annotation_attach job)",
)
def attach_annotation(
    scene_id: str,
    name: str = Form(..., min_length=1, max_length=200),
    annotation_type: str | None = Form(
        default=None,
        description=(
            "Registry KIND. Optional — if omitted, inferred from the "
            "uploaded filename's extension. Fall back to providing it "
            "explicitly when the extension is shared by multiple types."
        ),
    ),
    description: str | None = Form(default=None, max_length=1000),
    file: UploadFile = File(..., description="The annotation file (e.g. gt.tif)"),
    claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> AttachAccepted:
    """Stage the upload, INSERT a `jobs` row, return 202.

    Heavy work (file copy, overlay render) happens in the worker handler
    so the api stays snappy. Same shape as scene onboard's flow.

    Type resolution:
      - If `annotation_type` is provided, validate it against the registry.
      - Otherwise, infer from the filename's extension via the registry's
        `infer_kind_from_filename`. Returns 422 `ambiguous_or_unknown_type`
        when the extension matches no registered type, or matches multiple.
    """
    scene = _scene_or_404(scene_id, db)

    if file.filename is None or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="empty_filename",
        )

    # Resolve the type — provided > inferred from filename.
    resolved_type: str
    if annotation_type:
        if annotation_type not in supported_kinds():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="unsupported_annotation_type",
            )
        resolved_type = annotation_type
    else:
        inferred = infer_kind_from_filename(file.filename)
        if inferred is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="ambiguous_or_unknown_type",
            )
        resolved_type = inferred

    # Per-type cheap reject (extension / shape sanity). Worker re-checks.
    spec = get_spec(resolved_type)
    try:
        spec.validate_upload(file.filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    job_id = uuid.uuid4()
    staging_root = Path(settings.data_dir) / "staging" / str(job_id)
    try:
        staging_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        logger.exception("staging dir create failed: %s", staging_root)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="staging_unavailable",
        ) from exc

    # Strip directory components from the multipart filename — we don't
    # want to allow nested dirs in the staging area for annotations.
    safe_name = Path(file.filename).name
    if not safe_name or "\x00" in safe_name:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_filename",
        )

    target = staging_root / safe_name
    total_bytes = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = file.file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                total_bytes += len(chunk)
    except Exception:
        logger.exception("annotation upload failed")
        shutil.rmtree(staging_root, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="upload_failed",
        )

    payload: dict[str, Any] = {
        "scene_id": f"scene_{scene.id}",
        "annotation_type": resolved_type,
        "name": name,
        "description": description,
        "filename": safe_name,
        "staging_dir": f"staging/{job_id}",
        "uploaded_by_user_id": claims["sub"],
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    job = Job(
        id=job_id,
        type="annotation_attach",
        status="queued",
        payload=payload,
    )
    db.add(job)
    db.commit()

    logger.info(
        "queued annotation_attach job=%s scene=%s file=%s bytes=%d",
        job.id,
        scene.id,
        safe_name,
        total_bytes,
    )
    return AttachAccepted(
        job_id=f"job_{job.id}",
        annotation_name=name,
    )


# --- GET /scenes/{id}/annotations -------------------------------------


@router.get(
    "/{scene_id}/annotations",
    response_model=AnnotationsList,
    status_code=status.HTTP_200_OK,
    summary="List annotations attached to a scene",
)
def list_annotations(
    scene_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> AnnotationsList:
    scene = _scene_or_404(scene_id, db)
    rows = db.scalars(
        select(Annotation)
        .where(Annotation.scene_id == scene.id)
        .order_by(Annotation.created_at)
    ).all()
    items = [
        AnnotationPublic.from_orm_annotation(
            a, has_overlay=_overlay_path(scene.id, a.id).is_file()
        )
        for a in rows
    ]
    return AnnotationsList(items=items)


# --- GET /scenes/{id}/annotations/{ann_id}/overlay --------------------


@router.get(
    "/{scene_id}/annotations/{ann_id}/overlay",
    status_code=status.HTTP_200_OK,
    summary="Stream the RGBA overlay PNG (optionally re-rendered at a custom dot radius)",
)
def get_annotation_overlay(
    scene_id: str,
    ann_id: str,
    radius: int | None = Query(
        default=None,
        description=(
            "Override the default dot radius (output pixels). When "
            "absent, the worker-rendered PNG from attach time is "
            "served. When present, the api re-renders on demand from "
            "the source mask file. Bounds depend on the type's spec."
        ),
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    scene = _scene_or_404(scene_id, db)
    ann = _annotation_or_404(scene, ann_id, db)

    artifacts_root = Path(settings.artifacts_dir).resolve()

    # Default-radius path: serve the cached PNG the worker wrote at
    # attach time. Cache-Control: immutable so the browser keeps it.
    if radius is None:
        full = _confined(artifacts_root, _overlay_path(scene.id, ann.id))
        if not full.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="overlay_unavailable"
            )
        return FileResponse(
            path=str(full),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # Custom-radius path: re-render on demand from the source mask.
    # Per-radius output is content-addressed by (scene, ann, radius);
    # tag the response so browsers cache each radius separately.
    spec = get_spec(ann.type)
    source_file = (Path(settings.data_dir) / ann.file_path).resolve()
    if not source_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="source_unavailable"
        )

    # Non-default radii are a tuning step, not a permanent artifact —
    # write to a tmp file (the spec's render_overlay signature takes
    # a Path), read it back, then unlink. Browser-side cache via
    # Cache-Control + the radius query param keeps repeated requests
    # off the api.
    tmp = artifacts_root / "scenes" / str(scene.id) / "annotations" / str(ann.id) / f"_overlay_r{int(radius)}.png"
    try:
        spec.render_overlay(source_file, tmp, radius=int(radius))
        data = tmp.read_bytes()
    finally:
        # Best-effort cleanup so we don't accumulate per-radius files
        # on disk. The browser cache handles repeated requests.
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    return Response(
        content=data,
        media_type="image/png",
        headers={
            # Custom-radius path: 1-day cache, NOT immutable. The
            # default-radius PNG above is immutable (1-year) because
            # the worker pre-rendered it at attach time. Custom radii
            # are re-rendered on the fly, so we keep the door open to
            # changing render semantics later — bumping the radius
            # query param invalidates browser-side; a backend tweak
            # gets picked up after 1 day or a hard reload.
            "Cache-Control": "public, max-age=86400",
        },
    )


# --- DELETE /scenes/{id}/annotations/{ann_id} -------------------------


@router.delete(
    "/{scene_id}/annotations/{ann_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an annotation (synchronous)",
    response_class=Response,
)
def delete_annotation(
    scene_id: str,
    ann_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """Synchronous delete per abstractions-spec § 5.3.

    Order: row first (so a partial filesystem cleanup doesn't strand a
    DB row referencing missing files), then files. We then recompute
    `scenes.has_annotations` based on the remaining rows for this scene,
    inside the same transaction.
    """
    scene = _scene_or_404(scene_id, db)
    ann = _annotation_or_404(scene, ann_id, db)

    raw_dir = Path(settings.data_dir) / "scenes" / str(scene.id) / "annotations" / str(ann.id)
    artifacts_dir = (
        Path(settings.artifacts_dir) / "scenes" / str(scene.id) / "annotations" / str(ann.id)
    )

    db.delete(ann)
    db.flush()

    # Recompute denormalized has_annotations flag (CC-12).
    remaining = db.scalar(
        select(Annotation.id).where(Annotation.scene_id == scene.id).limit(1)
    )
    scene.has_annotations = remaining is not None

    db.commit()

    # Files removed AFTER the DB commit succeeds — if rmtree fails the
    # row is gone and an operator can sweep up the orphan dir later.
    # Reverse of that order would risk DB rollback after deleted files.
    for d in (raw_dir, artifacts_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    logger.info(
        "deleted annotation %s for scene %s (has_annotations=%s)",
        ann.id, scene.id, scene.has_annotations,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
