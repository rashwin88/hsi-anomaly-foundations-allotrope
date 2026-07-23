"""Project endpoints (Steps 10 + 11).

Routes:
    POST   /projects                — create a Project bound to a Scene (Step 10)
    GET    /projects                — list Projects (paginated, optional scene filter)
    GET    /projects/{project_id}   — read a single Project + child counts (Step 11 workspace shell)
    DELETE /projects/{project_id}   — synchronous delete (cascades to children)

Project is the structural pivot per abstractions-spec § 5.4 — workspace
bound to one Scene, owner is the user who created it.

Authorization (v1): any authenticated user can create / list / read /
delete projects. Per-user scoping arrives with multi-user RBAC (deferred
per spec § 11). The single-machine demo treats Library + Projects as
shared.

Sequence diagrams:
    final design/diagrams/projects-create.drawio
    final design/diagrams/projects-list.drawio
    final design/diagrams/projects-delete.drawio
    final design/diagrams/project-workspace-load.drawio   (Step 11 GET detail)
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Project, Scene
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.projects")

router = APIRouter(prefix="/projects", tags=["projects"])


# --- Schemas ----------------------------------------------------------


class ProjectPublic(BaseModel):
    """Wire shape for a Project row.

    `scene_name` and `scene_sensor_type` are denormalised onto the
    response so the Projects landing page can render the table without
    a second fetch per row. Both come from a JOIN at query time.
    """

    id: str                         # project_<uuid>
    user_id: str                    # user_<uuid>
    scene_id: str                   # scene_<uuid>
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    # Scene metadata for the list view — saves a per-row fetch.
    scene_name: str
    scene_sensor_type: str

    @classmethod
    def from_orm_project(cls, p: Project, scene: Scene) -> "ProjectPublic":
        return cls(
            id=f"project_{p.id}",
            user_id=f"user_{p.user_id}",
            scene_id=f"scene_{p.scene_id}",
            name=p.name,
            description=p.description,
            created_at=p.created_at,
            updated_at=p.updated_at,
            scene_name=scene.name,
            scene_sensor_type=scene.sensor_type,
        )


class ProjectsPage(BaseModel):
    """Paginated list of projects."""

    items: list[ProjectPublic]
    total: int
    limit: int
    offset: int


class ProjectDetail(ProjectPublic):
    """Wire shape for the workspace detail endpoint.

    Adds child counts so the workspace shell can show "0 actions",
    "0 visualizations", "0 notes" without firing extra requests. All
    counts hard-code to 0 in Step 11 (action / visualization / note
    tables don't exist yet — they land in Step 12+); each field is
    swapped to a real `SELECT count(*)` once its table is added.
    """

    action_count: int = 0
    visualization_count: int = 0
    note_count: int = 0


class CreateProjectPayload(BaseModel):
    """Request body for POST /projects."""

    scene_id: str = Field(
        ...,
        description="Wire-format scene id, e.g. `scene_<uuid>`. Immutable on the row.",
    )
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


# --- Helpers ----------------------------------------------------------


def _project_or_404(project_id_wire: str, db: Session) -> Project:
    raw_id = parse_prefixed_id("project", project_id_wire)
    project = db.get(Project, raw_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project_not_found"
        )
    return project


# --- POST /projects ---------------------------------------------------


@router.post(
    "",
    response_model=ProjectPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Project bound to a Scene",
)
def create_project(
    payload: CreateProjectPayload,
    claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ProjectPublic:
    """Create a Project for the authenticated user, bound to a Scene.

    Returns 201 with the freshly-created row's wire shape. The Scene
    must exist (404 scene_not_found otherwise). Scene + user are
    immutable on the row per spec § 5.4.
    """
    # Resolve the scene first so a bad scene_id fails before INSERT.
    scene_uuid = parse_prefixed_id("scene", payload.scene_id)
    scene = db.get(Scene, scene_uuid)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_found"
        )

    user_uuid = uuid.UUID(claims["sub"])

    project = Project(
        user_id=user_uuid,
        scene_id=scene.id,
        name=payload.name,
        description=payload.description,
    )
    db.add(project)
    try:
        db.commit()
    except IntegrityError as exc:
        # Both FKs are RESTRICT — the only real way to hit IntegrityError
        # here is a TOCTOU race where the Scene is deleted between our
        # db.get and the commit. Surface it cleanly rather than 500-ing.
        db.rollback()
        logger.warning("create_project IntegrityError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="scene_not_found"
        ) from exc

    db.refresh(project)
    logger.info(
        "created project %s on scene %s (user=%s)",
        project.id, scene.id, user_uuid,
    )
    return ProjectPublic.from_orm_project(project, scene)


# --- GET /projects ----------------------------------------------------


@router.get(
    "",
    response_model=ProjectsPage,
    status_code=status.HTTP_200_OK,
    summary="List Projects (paginated)",
)
def list_projects(
    limit: int = Query(50, ge=1, le=500, description="Max items per page (1-500)"),
    offset: int = Query(0, ge=0, description="Offset into the result set"),
    scene_id: str | None = Query(
        None,
        description="Optional filter: wire-format scene id (e.g. `scene_<uuid>`).",
    ),
    user_id: str | None = Query(
        None,
        description="Optional filter: wire-format user id (e.g. `user_<uuid>`).",
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ProjectsPage:
    """Return a page of Projects newest-first.

    Joins `scenes` for the denormalised `scene_name` / `scene_sensor_type`
    so the list view doesn't fan out to N+1 queries client-side.
    """
    sid = parse_prefixed_id("scene", scene_id) if scene_id is not None else None
    uid = parse_prefixed_id("user", user_id) if user_id is not None else None

    # Build the filter clause once and reuse for both count + page.
    filters = []
    if sid is not None:
        filters.append(Project.scene_id == sid)
    if uid is not None:
        filters.append(Project.user_id == uid)

    total = db.scalar(
        select(func.count(Project.id)).where(*filters)
    ) or 0

    page_stmt = (
        select(Project, Scene)
        .join(Scene, Scene.id == Project.scene_id)
        .where(*filters)
        .order_by(Project.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = db.execute(page_stmt).all()

    items = [ProjectPublic.from_orm_project(p, s) for p, s in rows]
    return ProjectsPage(items=items, total=total, limit=limit, offset=offset)


# --- GET /projects/{id} ----------------------------------------------


@router.get(
    "/{project_id}",
    response_model=ProjectDetail,
    status_code=status.HTTP_200_OK,
    summary="Get one Project by id (workspace detail)",
)
def get_project(
    project_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ProjectDetail:
    """Read a Project by wire-format id, bundled with Scene metadata
    and child entity counts.

    Powers the Project workspace shell (Step 11). The api stitches the
    Project + Scene into a single response so the client can render the
    workspace header in one round-trip; counts are stubbed at 0 for
    Step 11 and swapped to live counts as each child table lands.
    """
    project = _project_or_404(project_id, db)
    scene = db.get(Scene, project.scene_id)
    if scene is None:
        # RESTRICT FK should make this impossible — but defend cleanly.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="scene_missing_for_project",
        )
    base = ProjectPublic.from_orm_project(project, scene)
    return ProjectDetail(
        **base.model_dump(),
        # action_count / visualization_count / note_count default to 0
        # until Step 12+ adds the source tables. Swap to live `SELECT
        # count(*)` queries here when each table lands.
    )


# --- DELETE /projects/{id} -------------------------------------------


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a Project (synchronous, cascades to children)",
    response_class=Response,
)
def delete_project(
    project_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """Delete a Project and CASCADE through its children.

    Per the cascade table in abstractions-spec § 6:
    - Project → Action: CASCADE
    - Project → Visualization: CASCADE
    - Project → Note: CASCADE
    - Project → NoteReference (via ref_project_id): CASCADE
    - Project → Export: CASCADE
    - Project → Job (when project_id IS NOT NULL): CASCADE
    All of those are wired by the *child* table's migration FK; the
    DELETE here just trips them.

    None of those tables exist yet (Step 12+) so for v1 the cascade is
    a no-op below the projects row itself.

    Synchronous (per CC-15 — Project just-exists + delete; no states).
    """
    project = _project_or_404(project_id, db)
    project_uuid = project.id
    db.delete(project)
    db.commit()

    # Best-effort filesystem cleanup. The DB cascade has already wiped
    # the rows referencing this project; without rmtree the on-disk
    # artifact tree (action outputs, visualizations, exports) would
    # linger and the Monitoring page's disk usage wouldn't drop.
    artifacts_root = (
        Path(settings.artifacts_dir) / "projects" / str(project_uuid)
    )
    if artifacts_root.exists():
        try:
            shutil.rmtree(artifacts_root)
        except OSError as exc:
            logger.warning(
                "project %s delete: rmtree(%s) failed: %s",
                project_uuid,
                artifacts_root,
                exc,
            )

    logger.info("deleted project %s", project_uuid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


