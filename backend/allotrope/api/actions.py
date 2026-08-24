"""Action endpoints (Step 12c).

Routes:
    POST   /projects/{project_id}/actions  — submit an Action + enqueue its `action_run` Job
    GET    /projects/{project_id}/actions  — list Actions for a project (paginated)
    GET    /actions/{action_id}            — detail (includes ActionOutput when complete)
    GET    /action-types                   — public catalog (drives picker + Action card)

Submit flow (one DB transaction):
    1. Auth + parse project_id.
    2. Load Project + Scene (Scene needed for sensor_type).
    3. Look up the action type module from the registry (404 on unknown type).
    4. Run module.validate_config(raw_cfg, sensor_type) — Pydantic 422 on shape errors.
    5. Cross-field semantic checks the type module can't do alone:
         - configuration.input_scene_id must equal project.scene_id
         - configuration.input_band_filter_output_id (when present) must
           reference a `complete` ActionOutput from the same project.
    6. Optional: validate action_template_id exists.
    7. INSERT Action(status='queued') + INSERT Job(type='action_run',
       target_kind='action', target_id=action.id, project_id, payload).
    8. Return ActionPublic.

Worker keeps actions.status and jobs.status in lockstep at transaction
boundaries (Step 12d). The api never mutates lifecycle state directly —
it only writes the queued row and the paired job.

Sequence diagrams:
    final design/diagrams/action-submit.drawio
    final design/diagrams/action-list.drawio
    final design/diagrams/action-detail.drawio
    final design/diagrams/action-types-catalog.drawio
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import action_types as action_types_registry
from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, ActionOutput, ActionTemplate, Job, Project, Scene
from ._action_common import (
    ActionOutputPublic,
    action_or_404,
    output_for_action,
    output_to_wire,
)
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.actions")

# Two routers — one for project-nested (submit + list), one flat for
# action_<uuid> detail and the action-types catalog. main.py mounts both.
project_actions_router = APIRouter(
    prefix="/projects/{project_id}/actions",
    tags=["actions"],
)
actions_router = APIRouter(prefix="/actions", tags=["actions"])
action_types_router = APIRouter(prefix="/action-types", tags=["actions"])


# --- Schemas ----------------------------------------------------------


class CreateActionPayload(BaseModel):
    """Request body for POST /projects/{project_id}/actions."""

    type: str = Field(
        ...,
        description="Action type slug — must be a key in the action_types registry.",
        min_length=1,
        max_length=100,
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific configuration body. Validated by the type's module.",
    )
    action_template_id: str | None = Field(
        default=None,
        description="Optional template id (action_template_<uuid>) the configuration was seeded from.",
    )


class ActionPublic(BaseModel):
    """Wire shape for an Action row."""

    id: str                          # action_<uuid>
    project_id: str                  # project_<uuid>
    action_template_id: str | None   # action_template_<uuid>
    type: str
    configuration: dict[str, Any]
    status: str
    failure_reason: str | None
    started_at: datetime | None
    completed_at: datetime | None
    cancellation_requested: bool
    created_at: datetime

    @classmethod
    def from_orm_action(cls, a: Action) -> "ActionPublic":
        return cls(
            id=f"action_{a.id}",
            project_id=f"project_{a.project_id}",
            action_template_id=(
                f"action_template_{a.action_template_id}"
                if a.action_template_id is not None
                else None
            ),
            type=a.type,
            configuration=a.configuration,
            status=a.status,
            failure_reason=a.failure_reason,
            started_at=a.started_at,
            completed_at=a.completed_at,
            cancellation_requested=a.cancellation_requested,
            created_at=a.created_at,
        )


class ActionDetail(ActionPublic):
    """ActionPublic with the embedded ActionOutput, if one exists."""

    output: ActionOutputPublic | None = None


class ActionsPage(BaseModel):
    """Paginated list of Actions."""

    items: list[ActionPublic]
    total: int
    limit: int
    offset: int


class ActionTypeCatalog(BaseModel):
    """Wire shape for GET /action-types."""

    items: list[dict[str, Any]]


# --- Helpers ---------------------------------------------------------


def _project_and_scene_or_404(
    project_id_wire: str, db: Session
) -> tuple[Project, Scene]:
    project_uuid = parse_prefixed_id("project", project_id_wire)
    project = db.get(Project, project_uuid)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project_not_found"
        )
    scene = db.get(Scene, project.scene_id)
    if scene is None:
        # RESTRICT FK should make this impossible in practice.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="scene_missing_for_project",
        )
    return project, scene



# --- POST /projects/{id}/actions -----------------------------------


@project_actions_router.post(
    "",
    response_model=ActionPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Submit an Action and enqueue its action_run job",
)
def create_action(
    project_id: str,
    payload: CreateActionPayload,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionPublic:
    project, scene = _project_and_scene_or_404(project_id, db)

    # 1. Resolve the action type module.
    try:
        spec = action_types_registry.get_spec(payload.type)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown_action_type:{payload.type}",
        ) from e

    # 2. Type-side shape validation (Pydantic).
    try:
        validated = spec.validate_config(payload.configuration, scene.sensor_type)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        ) from e
    except ValueError as e:
        # Sensor-mismatch and similar cross-field shape errors.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    # 3. Cross-field semantic checks (input refs must be self-consistent
    #    with the bound Project + Scene). The type module owns shape;
    #    the api owns membership.

    # 3a. input_scene_id is the project's bound Scene — always.
    #     If the type's config schema requires it (band_filter_apply),
    #     the client-provided value must match. If absent
    #     (scene_segmentation, by design), the server fills it in so
    #     downstream consumers always read it from the configuration
    #     JSONB regardless of which type produced the row.
    project_scene_wire = f"scene_{project.scene_id}"
    cfg_scene_id = validated.get("input_scene_id")
    if cfg_scene_id is None:
        validated["input_scene_id"] = project_scene_wire
    elif parse_prefixed_id("scene", cfg_scene_id) != project.scene_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="input_scene_id must equal the project's bound scene",
        )

    # 3b. Every output reference inside `validated` must point at a
    #     `complete` ActionOutput whose Action belongs to this Project,
    #     and (when the field name implies a specific upstream type)
    #     must point at that producing type.
    _OUTPUT_REF_RULES: dict[str, str | None] = {
        # field name → required producing action type (None = any type)
        "input_band_filter_output_id": "band_filter_apply",
        "input_scene_segmentation_output_id": "scene_segmentation",
        "input_cloud_mask_output_id": "cloud_mask",
        # Future input refs add a line here and a corresponding META input spec.
    }
    for field_name, required_type in _OUTPUT_REF_RULES.items():
        referenced_output_id = validated.get(field_name)
        if referenced_output_id is None:
            continue
        output_uuid = parse_prefixed_id("output", referenced_output_id)
        ref_output = db.get(ActionOutput, output_uuid)
        if ref_output is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{field_name}_not_found",
            )
        ref_action = db.get(Action, ref_output.action_id)
        if (
            ref_action is None
            or ref_action.project_id != project.id
            or ref_action.status != "complete"
            or (required_type is not None and ref_action.type != required_type)
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{field_name}_not_usable",
            )

    # 3c. input_annotation_id (when present) must be a raster annotation
    #     attached to the project's bound Scene.
    cfg_annotation = validated.get("input_annotation_id")
    if cfg_annotation is not None:
        from ..models import Annotation

        annotation_uuid = parse_prefixed_id("annotation", cfg_annotation)
        ref_annotation = db.get(Annotation, annotation_uuid)
        if ref_annotation is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="input_annotation_id_not_found",
            )
        if ref_annotation.scene_id != project.scene_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="input_annotation_id_wrong_scene",
            )

    # 4. Optional: action_template_id existence. SET NULL semantics on
    #    delete mean we don't enforce membership beyond "row exists at
    #    submit time" — the configuration is the canonical record.
    template_uuid: uuid.UUID | None = None
    if payload.action_template_id is not None:
        template_uuid = parse_prefixed_id("action_template", payload.action_template_id)
        template = db.get(ActionTemplate, template_uuid)
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="action_template_not_found",
            )
        if template.type != payload.type:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="action_template_type_mismatch",
            )

    # 5. INSERT Action + paired action_run Job in one transaction.
    action_id = uuid.uuid4()
    action = Action(
        id=action_id,
        project_id=project.id,
        action_template_id=template_uuid,
        type=payload.type,
        configuration=validated,
        status="queued",
    )
    db.add(action)

    job = Job(
        id=uuid.uuid4(),
        type="action_run",
        status="queued",
        project_id=project.id,
        target_kind="action",
        target_id=action_id,
        payload={
            "action_id": f"action_{action_id}",
            "project_id": f"project_{project.id}",
            "scene_id": f"scene_{project.scene_id}",
            "action_type": payload.type,
        },
    )
    db.add(job)

    db.commit()
    db.refresh(action)

    logger.info(
        "queued action=%s type=%s project=%s job=%s",
        action.id,
        action.type,
        project.id,
        job.id,
    )

    return ActionPublic.from_orm_action(action)


# --- GET /projects/{id}/actions ------------------------------------


@project_actions_router.get(
    "",
    response_model=ActionsPage,
    summary="List Actions for a Project (paginated)",
)
def list_project_actions(
    project_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status_filter: str | None = Query(
        None,
        alias="status",
        description="Optional: queued | running | complete | failed | cancelled",
    ),
    type_filter: str | None = Query(
        None,
        alias="type",
        description="Optional: action type slug",
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionsPage:
    project_uuid = parse_prefixed_id("project", project_id)
    project = db.get(Project, project_uuid)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project_not_found"
        )

    base = select(Action).where(Action.project_id == project_uuid)
    if status_filter is not None:
        base = base.where(Action.status == status_filter)
    if type_filter is not None:
        base = base.where(Action.type == type_filter)

    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0

    rows = db.scalars(
        base.order_by(Action.created_at.desc()).limit(limit).offset(offset)
    ).all()

    return ActionsPage(
        items=[ActionPublic.from_orm_action(a) for a in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# --- GET /actions/{id} ---------------------------------------------


@actions_router.get(
    "/{action_id}",
    response_model=ActionDetail,
    summary="Get one Action by id (includes ActionOutput when complete)",
)
def get_action(
    action_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionDetail:
    action = action_or_404(action_id, db)
    base = ActionPublic.from_orm_action(action)
    output = output_for_action(action.id, db)
    return ActionDetail(
        **base.model_dump(),
        output=output_to_wire(output) if output is not None else None,
    )


# --- DELETE /actions/{id} ------------------------------------------


@actions_router.delete(
    "/{action_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an Action (cancel-while-running blocked with 409)",
    response_class=Response,
)
def delete_action(
    action_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """Synchronous Action delete.

    Spec Â§ 5.5 originally deferred individual Action delete to Project-
    delete-only; we lift that here. Guardrails:

    - status='running' is rejected with 409 — the worker still owns the
      row's lifecycle. Use the cancellation flag and wait for the
      terminal transition before deleting.
    - ActionOutput CASCADEs via its FK; Visualizations sourced from the
      ActionOutput CASCADE via theirs.
    - Job rows (target_kind='action', target_id=action.id) carry a SOFT
      ref — they stay as audit history with a dangling target_id, which
      the Jobs UI already tolerates.

    After the row deletes, we rmtree the action's artifact directory so
    we don't litter `projects/<pid>/actions/<aid>/` with stale outputs.
    """
    raw_id = parse_prefixed_id("action", action_id)
    action = db.get(Action, raw_id)
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="action_not_found"
        )

    if action.status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="action_running",
        )

    project_id = action.project_id
    action_uuid = action.id

    # Capture the on-disk directories of Visualizations that will CASCADE
    # via visualizations.source_action_output_id → action_outputs.id →
    # actions.id. Without this the DB rows go but the per-viz dirs
    # under projects/<pid>/visualizations/<vid>/ linger forever and disk
    # usage doesn't drop.
    from ..models import ActionOutput as _AO, Visualization as _Viz  # local

    output_ids = list(
        db.scalars(
            select(_AO.id).where(_AO.action_id == action_uuid)
        ).all()
    )
    orphan_viz_paths: list[Path] = []
    if output_ids:
        for v_path in db.scalars(
            select(_Viz.artifact_path).where(
                _Viz.source_action_output_id.in_(output_ids)
            )
        ).all():
            # artifact_path points at the image file; parent is the dir.
            orphan_viz_paths.append(
                (Path(settings.artifacts_dir) / v_path).parent
            )

    db.delete(action)
    db.commit()

    artifact_dir = (
        Path(settings.artifacts_dir)
        / "projects"
        / str(project_id)
        / "actions"
        / str(action_uuid)
    )
    to_remove = [artifact_dir, *orphan_viz_paths]
    for d in to_remove:
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError as exc:
                logger.warning(
                    "action %s delete: rmtree(%s) failed: %s",
                    action_uuid,
                    d,
                    exc,
                )

    logger.info(
        "action deleted id=%s project=%s viz_dirs=%d",
        action_uuid,
        project_id,
        len(orphan_viz_paths),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- GET /action-types ---------------------------------------------


@action_types_router.get(
    "",
    response_model=ActionTypeCatalog,
    summary="Public catalog of action types (label / description / inputs / outputs / sensor compatibility / defaults)",
)
def get_action_types(
    _claims: Claims = Depends(current_user_claims),
) -> ActionTypeCatalog:
    """Drives the Action picker dialog and the Action card on the workspace.

    Single source of truth: each `action_types/<kind>.py` module's META
    payload. Authentication required to keep parity with annotation_types.
    """
    return ActionTypeCatalog(items=action_types_registry.public_catalog())
