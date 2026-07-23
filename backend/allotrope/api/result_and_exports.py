"""Result + Export endpoints (Step 17).

Result is a **computed view** — no table. We assemble it on the fly from
existing rows so it always reflects current state.

Export is **async-via-queue** (Option B in the spec): the api enqueues
a `project_export` job and returns the job id; the worker assembles the
bundle on disk and INSERTs the Export row only after success. Download
streams the produced zip back to the user.

Routes:
    GET /projects/{project_id}/result          computed snapshot
    POST /projects/{project_id}/exports        enqueue project_export job
    GET /projects/{project_id}/exports         list completed exports
    GET /exports/{export_id}                   detail (DB row)
    GET /exports/{export_id}/download          stream bundle bytes

Sequence diagram: final design/diagrams/project-export.drawio
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import (
    Action,
    ActionOutput,
    Annotation,
    Export,
    Job,
    Note,
    Project,
    Scene,
    Visualization,
)
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.result_and_exports")

project_result_router = APIRouter(
    prefix="/projects/{project_id}", tags=["result"]
)
project_exports_router = APIRouter(
    prefix="/projects/{project_id}/exports", tags=["exports"]
)
exports_router = APIRouter(prefix="/exports", tags=["exports"])


# --- Wire schemas ----------------------------------------------------


class ResultActionLine(BaseModel):
    id: str
    type: str
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    failure_reason: str | None
    output_id: str | None
    summary: dict[str, Any] | None  # action_output.summary; None if no output yet


class ResultProjectMeta(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: datetime
    scene_id: str
    scene_name: str
    scene_sensor_type: str


class ResultPublic(BaseModel):
    project: ResultProjectMeta
    actions: list[ResultActionLine]
    visualization_count: int
    note_count: int
    annotation_count: int
    last_action_completed_at: datetime | None
    generated_at: datetime


class ExportPublic(BaseModel):
    id: str
    project_id: str
    bundle_path: str
    download_url: str
    snapshot_at: datetime
    size_bytes: int
    format: str
    created_at: datetime

    @classmethod
    def from_orm(cls, e: Export) -> "ExportPublic":
        wire = f"export_{e.id}"
        return cls(
            id=wire,
            project_id=f"project_{e.project_id}",
            bundle_path=e.bundle_path,
            download_url=f"/exports/{wire}/download",
            snapshot_at=e.snapshot_at,
            size_bytes=e.size_bytes,
            format=e.format,
            created_at=e.created_at,
        )


class ExportList(BaseModel):
    items: list[ExportPublic]


class ExportAccepted(BaseModel):
    """Returned from POST /projects/{id}/exports — the job id, not the
    Export row (which doesn't exist yet)."""

    job_id: str
    project_id: str


# --- Helpers ---------------------------------------------------------


def _project_or_404(project_id_wire: str, db: Session) -> Project:
    raw = parse_prefixed_id("project", project_id_wire)
    project = db.get(Project, raw)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project_not_found"
        )
    return project


def _export_or_404(export_id_wire: str, db: Session) -> Export:
    raw = parse_prefixed_id("export", export_id_wire)
    e = db.get(Export, raw)
    if e is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="export_not_found"
        )
    return e


# --- GET /projects/{id}/result ---------------------------------------


@project_result_router.get(
    "/result",
    response_model=ResultPublic,
    status_code=status.HTTP_200_OK,
    summary="Computed Result snapshot for a Project",
)
def get_project_result(
    project_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ResultPublic:
    project = _project_or_404(project_id, db)
    scene = db.get(Scene, project.scene_id)
    if scene is None:
        # FK invariant should prevent this; defend cleanly.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="scene_missing_for_project",
        )

    # All actions for this project, with their outputs left-joined so we
    # can render the "Result is the up-to-date roll-up of completed
    # work" line in a single round-trip.
    rows = list(
        db.execute(
            select(Action, ActionOutput)
            .join(ActionOutput, ActionOutput.action_id == Action.id, isouter=True)
            .where(Action.project_id == project.id)
            .order_by(Action.created_at.asc())
        ).all()
    )

    actions: list[ResultActionLine] = []
    for a, o in rows:
        actions.append(
            ResultActionLine(
                id=f"action_{a.id}",
                type=a.type,
                status=a.status,
                started_at=a.started_at,
                completed_at=a.completed_at,
                failure_reason=a.failure_reason,
                output_id=(f"output_{o.id}" if o is not None else None),
                summary=(o.summary if o is not None else None),
            )
        )

    viz_count = (
        db.scalar(
            select(func.count(Visualization.id)).where(
                Visualization.project_id == project.id
            )
        )
        or 0
    )
    note_count = (
        db.scalar(
            select(func.count(Note.id)).where(Note.project_id == project.id)
        )
        or 0
    )
    annotation_count = (
        db.scalar(
            select(func.count(Annotation.id)).where(
                Annotation.scene_id == project.scene_id
            )
        )
        or 0
    )
    last_action_completed_at = db.scalar(
        select(func.max(Action.completed_at)).where(
            Action.project_id == project.id, Action.status == "complete"
        )
    )

    return ResultPublic(
        project=ResultProjectMeta(
            id=f"project_{project.id}",
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            scene_id=f"scene_{scene.id}",
            scene_name=scene.name,
            scene_sensor_type=scene.sensor_type,
        ),
        actions=actions,
        visualization_count=int(viz_count),
        note_count=int(note_count),
        annotation_count=int(annotation_count),
        last_action_completed_at=last_action_completed_at,
        generated_at=datetime.utcnow(),
    )


# --- POST /projects/{id}/exports -------------------------------------


@project_exports_router.post(
    "",
    response_model=ExportAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a project_export job (returns job id, not Export)",
)
def create_export_job(
    project_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ExportAccepted:
    project = _project_or_404(project_id, db)
    # Reject if another project_export is already queued or running for
    # this project — keeps the bundle directory race-free without a hard
    # lock.
    existing = db.scalar(
        select(Job).where(
            Job.type == "project_export",
            Job.project_id == project.id,
            Job.status.in_(("queued", "running")),
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="export_already_running"
        )

    job = Job(
        id=uuid.uuid4(),
        type="project_export",
        status="queued",
        project_id=project.id,
        target_kind="export",  # target_id populated by the worker
        target_id=None,
        payload={"project_id": f"project_{project.id}"},
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    logger.info(
        "queued project_export project=%s job=%s", project.id, job.id
    )
    return ExportAccepted(
        job_id=f"job_{job.id}", project_id=f"project_{project.id}"
    )


# --- GET /projects/{id}/exports --------------------------------------


@project_exports_router.get(
    "",
    response_model=ExportList,
    status_code=status.HTTP_200_OK,
    summary="List completed Exports for a Project (newest first)",
)
def list_project_exports(
    project_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ExportList:
    project = _project_or_404(project_id, db)
    rows = list(
        db.scalars(
            select(Export)
            .where(Export.project_id == project.id)
            .order_by(Export.created_at.desc())
        ).all()
    )
    return ExportList(items=[ExportPublic.from_orm(e) for e in rows])


# --- GET /exports/{id} -----------------------------------------------


@exports_router.get(
    "/{export_id}",
    response_model=ExportPublic,
    status_code=status.HTTP_200_OK,
    summary="Get one Export by id",
)
def get_export(
    export_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ExportPublic:
    return ExportPublic.from_orm(_export_or_404(export_id, db))


# --- GET /exports/{id}/download --------------------------------------


@exports_router.get(
    "/{export_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Stream the bundle file",
    response_class=FileResponse,
)
def download_export(
    export_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    e = _export_or_404(export_id, db)
    abs_path = Path(settings.artifacts_dir) / e.bundle_path
    if not abs_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="bundle_missing"
        )
    media_type = (
        "application/zip"
        if e.format == "zip"
        else "application/octet-stream"
    )
    filename = abs_path.name
    return FileResponse(
        path=str(abs_path),
        media_type=media_type,
        filename=filename,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
