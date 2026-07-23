"""Project Visualization endpoints (Step 15, scoped 2026-05-11).

Routes:
    POST   /projects/{project_id}/visualizations         create from upload
    GET    /projects/{project_id}/visualizations         list (newest first)
    GET    /visualizations/{viz_id}                      detail
    GET    /visualizations/{viz_id}/image                stream the saved PNG
    PATCH  /visualizations/{viz_id}                      rename / re-describe
    DELETE /visualizations/{viz_id}                      synchronous delete

Save model: any viewer that wants to pin its state flattens the current
frame to a PNG and POSTs it as `image` alongside the JSON metadata
(`source`, `name`, optional description, `view_state` blob).

Storage layout:
    allotrope_artifacts/projects/<project_id>/visualizations/<viz_id>/image.png

`source_kind = 'scene'`        → `source.scene_id` required (wire id)
`source_kind = 'action_output'`→ `source.action_id` required; api resolves
                                 to the unique ActionOutput.

Sequence diagram: final design/diagrams/visualization-save.drawio
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
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
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import ActionOutput, Project, Scene, Visualization
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.project_visualizations")

# Project-nested router (create + list).
project_visualizations_router = APIRouter(
    prefix="/projects/{project_id}/visualizations",
    tags=["visualizations"],
)
# Flat router for detail/patch/delete/image — viz_<uuid> is globally unique.
visualizations_router = APIRouter(prefix="/visualizations", tags=["visualizations"])

# 25 MB cap — saved PNGs are flattened viewer frames, not raw scenes.
_MAX_IMAGE_BYTES = 25 * 1024 * 1024


# --- Wire schemas ----------------------------------------------------


class VisualizationSource(BaseModel):
    """Polymorphic source spec on create.

    Exactly one of `scene_id` or `action_id` is required. `action_id` is
    resolved server-side to the unique ActionOutput for that Action.
    """

    kind: str = Field(..., description="'scene' or 'action_output'")
    scene_id: str | None = Field(None, description="wire id scene_<uuid>")
    action_id: str | None = Field(None, description="wire id action_<uuid>")


class VisualizationCreateMeta(BaseModel):
    """JSON metadata POSTed alongside the image upload."""

    source: VisualizationSource
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    view_state: dict[str, Any] = Field(default_factory=dict)


class VisualizationPatch(BaseModel):
    """PATCH body. Only name / description are mutable per spec § 5.9."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)


class VisualizationPublic(BaseModel):
    id: str                            # viz_<uuid>
    project_id: str                    # project_<uuid>
    source_kind: str
    source_scene_id: str | None        # scene_<uuid>
    source_action_output_id: str | None  # output_<uuid>
    name: str
    description: str | None
    artifact_path: str
    image_url: str                     # /visualizations/<viz_id>/image
    view_state: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm_viz(cls, v: Visualization) -> "VisualizationPublic":
        viz_wire = f"viz_{v.id}"
        return cls(
            id=viz_wire,
            project_id=f"project_{v.project_id}",
            source_kind=v.source_kind,
            source_scene_id=(
                f"scene_{v.source_scene_id}" if v.source_scene_id else None
            ),
            source_action_output_id=(
                f"output_{v.source_action_output_id}"
                if v.source_action_output_id
                else None
            ),
            name=v.name,
            description=v.description,
            artifact_path=v.artifact_path,
            image_url=f"/visualizations/{viz_wire}/image",
            view_state=v.view_state or {},
            created_at=v.created_at,
            updated_at=v.updated_at,
        )


class VisualizationList(BaseModel):
    items: list[VisualizationPublic]
    total: int
    limit: int
    offset: int


# --- Helpers ---------------------------------------------------------


def _project_or_404(project_id_wire: str, db: Session) -> Project:
    raw = parse_prefixed_id("project", project_id_wire)
    project = db.get(Project, raw)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project_not_found"
        )
    return project


def _viz_or_404(viz_id_wire: str, db: Session) -> Visualization:
    raw = parse_prefixed_id("viz", viz_id_wire)
    viz = db.get(Visualization, raw)
    if viz is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="visualization_not_found"
        )
    return viz


def _resolve_source(
    source: VisualizationSource, project: Project, db: Session
) -> tuple[uuid.UUID | None, uuid.UUID | None]:
    """Return (scene_id, action_output_id) — exactly one non-None.

    Enforces that the referenced scene/action lives inside this Project's
    investigation scope (same Project for actions; matching scene_id for
    Scene-rooted visualizations).
    """
    if source.kind == "scene":
        if not source.scene_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="scene_id_required",
            )
        scene_uuid = parse_prefixed_id("scene", source.scene_id)
        # The Project is rooted on one Scene; only that scene is valid.
        if scene_uuid != project.scene_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="scene_not_in_project",
            )
        scene = db.get(Scene, scene_uuid)
        if scene is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_found"
            )
        return (scene_uuid, None)

    if source.kind == "action_output":
        if not source.action_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="action_id_required",
            )
        action_uuid = parse_prefixed_id("action", source.action_id)
        # The ActionOutput table has UNIQUE(action_id), so .scalar() is safe.
        output = db.scalar(
            select(ActionOutput).where(ActionOutput.action_id == action_uuid)
        )
        if output is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="action_output_not_found",
            )
        # Cheap project-scope check: the output's path starts with the
        # project's directory. Saves a JOIN to actions.
        expected_prefix = f"projects/{project.id}/actions/"
        if not output.artifact_path.startswith(expected_prefix):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="action_not_in_project",
            )
        return (None, output.id)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="bad_source_kind"
    )


# --- POST /projects/{id}/visualizations ------------------------------


@project_visualizations_router.post(
    "",
    response_model=VisualizationPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Save a flattened viewer frame as a Visualization",
)
def create_visualization(
    project_id: str,
    meta: str = Form(
        ...,
        description="JSON-encoded VisualizationCreateMeta",
    ),
    image: UploadFile = File(
        ...,
        description="Flattened viewer frame as PNG (≤ 25 MB)",
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> VisualizationPublic:
    # Parse + validate metadata.
    try:
        meta_obj = VisualizationCreateMeta.model_validate(json.loads(meta))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_metadata",
        ) from exc

    project = _project_or_404(project_id, db)

    if image.content_type not in {"image/png", "image/jpeg"}:
        # PNG is the contract; jpeg accepted as a small mercy. Reject the
        # rest so we don't catalog arbitrary file types as visualizations.
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="unsupported_image_type",
        )

    scene_id, action_output_id = _resolve_source(meta_obj.source, project, db)

    viz_id = uuid.uuid4()
    rel_dir = Path("projects") / str(project.id) / "visualizations" / str(viz_id)
    abs_dir = Path(settings.artifacts_dir) / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    # Pick filename by content type so the browser serves the right MIME
    # back. Extension drives FileResponse media_type below.
    ext = "png" if image.content_type == "image/png" else "jpg"
    filename = f"image.{ext}"
    image_abs = abs_dir / filename

    written = 0
    try:
        with image_abs.open("wb") as f:
            while True:
                chunk = image.file.read(1 << 16)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="image_too_large",
                    )
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(abs_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(abs_dir, ignore_errors=True)
        logger.exception("visualization image write failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="image_write_failed",
        ) from exc

    viz = Visualization(
        id=viz_id,
        project_id=project.id,
        source_kind=meta_obj.source.kind,
        source_scene_id=scene_id,
        source_action_output_id=action_output_id,
        name=meta_obj.name,
        description=meta_obj.description,
        artifact_path=str(rel_dir / filename),
        view_state=meta_obj.view_state,
    )
    db.add(viz)
    try:
        db.commit()
    except Exception:
        db.rollback()
        shutil.rmtree(abs_dir, ignore_errors=True)
        raise
    db.refresh(viz)
    logger.info(
        "visualization created id=%s project=%s source=%s",
        viz.id,
        project.id,
        viz.source_kind,
    )
    return VisualizationPublic.from_orm_viz(viz)


# --- GET /projects/{id}/visualizations -------------------------------


@project_visualizations_router.get(
    "",
    response_model=VisualizationList,
    status_code=status.HTTP_200_OK,
    summary="List Visualizations in a Project (newest first)",
)
def list_visualizations(
    project_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> VisualizationList:
    project = _project_or_404(project_id, db)

    total = (
        db.scalar(
            select(func.count(Visualization.id)).where(
                Visualization.project_id == project.id
            )
        )
        or 0
    )
    rows = db.scalars(
        select(Visualization)
        .where(Visualization.project_id == project.id)
        .order_by(Visualization.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return VisualizationList(
        items=[VisualizationPublic.from_orm_viz(v) for v in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# --- GET /visualizations/{viz_id} ------------------------------------


@visualizations_router.get(
    "/{viz_id}",
    response_model=VisualizationPublic,
    status_code=status.HTTP_200_OK,
    summary="Get one Visualization by id",
)
def get_visualization(
    viz_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> VisualizationPublic:
    return VisualizationPublic.from_orm_viz(_viz_or_404(viz_id, db))


# --- GET /visualizations/{viz_id}/image ------------------------------


@visualizations_router.get(
    "/{viz_id}/image",
    status_code=status.HTTP_200_OK,
    summary="Stream the saved visualization image",
    response_class=FileResponse,
)
def get_visualization_image(
    viz_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    viz = _viz_or_404(viz_id, db)
    abs_path = Path(settings.artifacts_dir) / viz.artifact_path
    if not abs_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="image_missing"
        )
    media_type = (
        "image/jpeg" if abs_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    )
    return FileResponse(
        path=str(abs_path),
        media_type=media_type,
        # Visualization bytes are immutable post-create; cache forever.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --- PATCH /visualizations/{viz_id} ----------------------------------


@visualizations_router.patch(
    "/{viz_id}",
    response_model=VisualizationPublic,
    status_code=status.HTTP_200_OK,
    summary="Rename / re-describe a Visualization",
)
def patch_visualization(
    viz_id: str,
    body: VisualizationPatch,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> VisualizationPublic:
    viz = _viz_or_404(viz_id, db)
    touched = False
    if body.name is not None:
        viz.name = body.name
        touched = True
    if body.description is not None:
        viz.description = body.description
        touched = True
    if touched:
        db.commit()
        db.refresh(viz)
    return VisualizationPublic.from_orm_viz(viz)


# --- DELETE /visualizations/{viz_id} ---------------------------------


@visualizations_router.delete(
    "/{viz_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a Visualization (DB row + on-disk directory)",
    response_class=Response,
)
def delete_visualization(
    viz_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    viz = _viz_or_404(viz_id, db)
    # Compute the directory before we wipe the row — the path column is
    # the source of truth.
    abs_image = Path(settings.artifacts_dir) / viz.artifact_path
    abs_dir = abs_image.parent
    db.delete(viz)
    db.commit()
    # Best-effort cleanup; missing dir is fine.
    if abs_dir.exists():
        shutil.rmtree(abs_dir, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
