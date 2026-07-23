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
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import action_types as action_types_registry
from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, ActionOutput, ActionTemplate, Job, Project, Scene
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
action_outputs_router = APIRouter(prefix="/action-outputs", tags=["actions"])
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


class ActionOutputPublic(BaseModel):
    """Wire shape for ActionOutput rows."""

    id: str                          # output_<uuid>
    action_id: str                   # action_<uuid>
    artifact_path: str
    summary: dict[str, Any]
    created_at: datetime


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


def _action_or_404(action_id_wire: str, db: Session) -> Action:
    action_uuid = parse_prefixed_id("action", action_id_wire)
    action = db.get(Action, action_uuid)
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="action_not_found"
        )
    return action


def _output_for_action(action_id: uuid.UUID, db: Session) -> ActionOutput | None:
    return db.scalar(
        select(ActionOutput).where(ActionOutput.action_id == action_id)
    )


def _output_to_wire(o: ActionOutput) -> ActionOutputPublic:
    return ActionOutputPublic(
        id=f"output_{o.id}",
        action_id=f"action_{o.action_id}",
        artifact_path=o.artifact_path,
        summary=o.summary,
        created_at=o.created_at,
    )


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
    action = _action_or_404(action_id, db)
    base = ActionPublic.from_orm_action(action)
    output = _output_for_action(action.id, db)
    return ActionDetail(
        **base.model_dump(),
        output=_output_to_wire(output) if output is not None else None,
    )


# --- GET /actions/{id}/files/{filename} ------------------------------


_FILE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".pkl": "application/octet-stream",
}


@actions_router.get(
    "/{action_id}/files/{filename}",
    status_code=status.HTTP_200_OK,
    summary="Stream an artifact file from the Action's output directory",
    response_class=FileResponse,
)
def get_action_file(
    action_id: str,
    filename: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream a file from `<artifacts>/<action_output.artifact_path>/<filename>`.

    Used by the Action card / Output viewer in the workspace to render
    `preview.png`, fetch `diagnostics.json`, and (later) ship raw rasters
    to client-side tooling.

    - 404 `action_not_found` for unknown action_id.
    - 404 `output_not_ready` when the action has no ActionOutput yet.
    - 404 `file_not_found` when the artifact dir exists but the named
      file is missing.
    - 422 `invalid_filename` for traversal attempts (`..`, slashes,
      empty names) — basename-only access is enforced.

    Path-traversal defence is two-fold: filename is rejected if it
    contains `/` or `\\` or `..` segments; the resolved absolute path
    is then asserted to live inside the artifacts root.
    """
    action = _action_or_404(action_id, db)

    # Filename must be a single basename — no traversal, no nesting.
    if not filename or "/" in filename or "\\" in filename or filename in ("..", "."):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_filename",
        )
    if ".." in filename.split("."):
        # paranoid catch — shouldn't trigger because of the slash check above
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_filename",
        )

    output = _output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    full = (artifacts_root / output.artifact_path / filename).resolve()
    # Defence in depth — confine to artifacts root.
    try:
        full.relative_to(artifacts_root)
    except ValueError:
        logger.warning(
            "action %s output filename=%r escapes artifacts root",
            action.id,
            filename,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )
    if not full.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )

    media = _FILE_MEDIA_TYPES.get(full.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(full),
        media_type=media,
        # Action artifacts are write-once + immutable per § 5.6 — cache aggressively.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --- GET /actions/{id}/output/{relpath:path} -------------------------
#
# Some action types (anomaly_scoring) write nested per-model artifact
# directories. The single-basename `/files/{filename}` route can't
# reach them. This sibling allows nested relpaths inside the same
# artifact root, with the same traversal defence.


@actions_router.get(
    "/{action_id}/output/{relpath:path}",
    status_code=status.HTTP_200_OK,
    summary="Stream a nested artifact file from the Action's output directory",
    response_class=FileResponse,
)
def get_action_output_file(
    action_id: str,
    relpath: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream a nested file under `<artifacts>/<action_output.artifact_path>/<relpath>`.

    Used by the anomaly_scoring viewer to fetch
    `models/<codename>/anomaly_score.png` etc. without flattening the
    on-disk layout.
    """
    action = _action_or_404(action_id, db)

    if not relpath or relpath.startswith("/") or "\\" in relpath:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_relpath",
        )
    parts = relpath.split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_relpath",
        )

    output = _output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    full = (artifacts_root / output.artifact_path / relpath).resolve()
    try:
        full.relative_to(artifacts_root)
    except ValueError:
        logger.warning(
            "action %s output relpath=%r escapes artifacts root",
            action.id,
            relpath,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )
    if not full.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )

    media = _FILE_MEDIA_TYPES.get(full.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(full),
        media_type=media,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --- GET /action-outputs/{id} ---------------------------------------
#
# Tiny lookup endpoint so the frontend can resolve an
# ``output_<uuid>`` to its producing ``action_<uuid>``. Used by the
# NewActionDialog when wiring an anomaly_detection_prep — once the user
# picks an upstream anomaly_scoring Output we need to fetch that
# Action's summary.json to discover which algorithms ran, so the
# dialog can render one weight input per algorithm.


@action_outputs_router.get(
    "/{output_id}",
    response_model=ActionOutputPublic,
    summary="Look up an ActionOutput by wire-format id",
)
def get_action_output(
    output_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionOutputPublic:
    """Resolve ``output_<uuid>`` → its ActionOutput row (incl. action_id)."""
    raw_id = parse_prefixed_id("output", output_id)
    output = db.get(ActionOutput, raw_id)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_found",
        )
    return _output_to_wire(output)


# --- POST /actions/{id}/anomaly_detection_preview -------------------
#
# Interactive Apply endpoint for prep actions sitting in
# ``needs_threshold``. The user moves a slider in the viewer, presses
# Apply, and this endpoint computes the binary anomaly mask + (if GT
# attached) precision/recall/F1 for *that specific* threshold choice.
# The mask + metrics are ephemeral — recomputed every call. See
# Roadmap step 14.5 for the design discussion.


class AnomalyDetectionPreviewRequest(BaseModel):
    """Body for the Apply round-trip."""

    threshold: float = Field(
        ...,
        description=(
            "Slider value. Interpreted as a percentile (0..100) when "
            "threshold_mode='percentile', or as a raw composite value "
            "in [0, 1] when threshold_mode='absolute'."
        ),
    )
    threshold_mode: str = Field(
        default="percentile",
        description="'percentile' (default) or 'absolute'.",
    )
    dilation_kernel: int = Field(
        default=0,
        ge=0,
        description=(
            "Odd integer (or 0 to disable). Side of the square structuring "
            "element used to dilate the binary anomaly mask before metrics."
        ),
    )


class AnomalyDetectionPreviewResponse(BaseModel):
    """Wire shape returned by the Apply endpoint.

    The mask PNG itself is served via a sibling endpoint that the
    frontend hits as an image URL — keeps JSON small and lets the
    browser cache-bust on the threshold tuple.
    """

    threshold_absolute: float
    threshold_percentile: float
    dilation_kernel: int
    n_anomalous: int
    n_kept: int
    metrics: dict | None
    # Frontend constructs a deterministic URL from these params to
    # request the rendered PNG separately.
    mask_url: str


@actions_router.post(
    "/{action_id}/anomaly_detection_preview",
    response_model=AnomalyDetectionPreviewResponse,
    summary="Apply a threshold to the composite score and return the binary mask + metrics",
)
def anomaly_detection_preview(
    action_id: str,
    body: AnomalyDetectionPreviewRequest,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> AnomalyDetectionPreviewResponse:
    """Interactive threshold preview for an ``anomaly_detection_prep`` action.

    The action must be in status ``needs_threshold`` and its output dir
    must contain ``composite_score.tif`` + ``summary.json``. Everything
    else (the mask, the metrics) is recomputed fresh on each call —
    nothing is persisted.

    Returns the rendered binary mask via a sibling URL (cached in this
    process's memory and served by ``GET .../anomaly_detection_preview_mask``).
    """
    from ._anomaly_detection_preview import (
        compute_preview,
        load_gt_mask_for_action,
    )

    action = _action_or_404(action_id, db)
    if action.type != "anomaly_detection_prep":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="action_type_mismatch",
        )
    if action.status not in ("needs_threshold", "complete"):
        # We tolerate `complete` so users who later committed a
        # threshold can still re-explore alternatives in the prep
        # viewer until the commit pathway exists.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"action_not_in_needs_threshold (status={action.status!r})",
        )

    output = _output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    output_dir = (artifacts_root / output.artifact_path).resolve()
    composite_path = output_dir / "composite_score.tif"
    if not composite_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="composite_score_not_found",
        )

    # Optional GT — drives metrics when present.
    gt_mask = None
    annotation_id = (action.configuration or {}).get("input_annotation_id")
    composite_shape = None
    if annotation_id:
        # Resolve scene_id via the action's project — same scene_dir
        # convention anomaly_scoring uses for its own GT loader.
        project = db.get(Project, action.project_id)
        scene_id_for_gt = str(project.scene_id) if project else None

        # The composite raster's shape is the GT lookup constraint.
        # Read the composite header cheaply to get it without pulling
        # the whole raster into RAM here (the cache in
        # _anomaly_detection_preview handles the full read).
        import rasterio
        with rasterio.open(composite_path) as src:
            composite_shape = src.shape
        if scene_id_for_gt is not None:
            data_root = Path(settings.data_dir).resolve()
            gt_mask = load_gt_mask_for_action(
                data_root=data_root,
                scene_id=scene_id_for_gt,
                annotation_id=annotation_id,
                expected_shape=composite_shape,
            )

    try:
        result = compute_preview(
            composite_path=composite_path,
            threshold=body.threshold,
            threshold_mode=body.threshold_mode,
            dilation_kernel=body.dilation_kernel,
            gt_mask=gt_mask,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Stash the rendered PNG in the per-request cache so the sibling
    # GET endpoint can serve it without recomputing.
    _stash_preview_mask(
        action_id=str(action.id),
        params=(body.threshold, body.threshold_mode, body.dilation_kernel),
        png_bytes=result.mask_png_bytes,
    )
    # Browser-relative URL — the frontend's nginx proxies /api/* to
    # this api process, so we include the /api prefix so an <img src>
    # in the SPA dereferences correctly.
    mask_url = (
        f"/api/actions/{action_id}/anomaly_detection_preview_mask"
        f"?t={body.threshold}&mode={body.threshold_mode}"
        f"&dk={body.dilation_kernel}"
    )
    return AnomalyDetectionPreviewResponse(
        threshold_absolute=result.threshold_absolute,
        threshold_percentile=result.threshold_percentile,
        dilation_kernel=result.dilation_kernel,
        n_anomalous=result.n_anomalous,
        n_kept=result.n_kept,
        metrics=result.metrics,
        mask_url=mask_url,
    )


# Tiny per-process cache: the POST renders the PNG, the GET serves it
# bytewise. Keyed by (action_id, threshold, mode, kernel). LRU-capped.
_PREVIEW_MASK_CACHE_MAX = 16
_preview_mask_cache: "dict[tuple, bytes]" = {}
_preview_mask_lru: "list[tuple]" = []


def _stash_preview_mask(*, action_id: str, params: tuple, png_bytes: bytes) -> None:
    key = (action_id, *params)
    _preview_mask_cache[key] = png_bytes
    if key in _preview_mask_lru:
        _preview_mask_lru.remove(key)
    _preview_mask_lru.append(key)
    while len(_preview_mask_lru) > _PREVIEW_MASK_CACHE_MAX:
        oldest = _preview_mask_lru.pop(0)
        _preview_mask_cache.pop(oldest, None)


@actions_router.get(
    "/{action_id}/anomaly_detection_preview_mask",
    summary="Stream the binary anomaly mask PNG matching the last Apply call's params",
)
def anomaly_detection_preview_mask(
    action_id: str,
    t: float = Query(...),
    mode: str = Query("percentile"),
    dk: int = Query(0),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """Serve a previously-rendered binary anomaly mask PNG.

    The POST endpoint computes the PNG and stashes it; this sibling
    GET serves the bytes. If the cache lost the entry (process
    restarted, or another Apply call evicted it), the client should
    re-POST to recompute.
    """
    _action = _action_or_404(action_id, db)
    key = (str(_action.id), t, mode, dk)
    png_bytes = _preview_mask_cache.get(key)
    if png_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="preview_mask_not_cached_repost_apply",
        )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# --- POST /actions/{id}/anomaly_detection_commit --------------------
#
# Lock in the user's chosen threshold + dilation on a prep action.
# Writes ``anomaly_mask.tif`` (binary geotiff) + ``metrics.json`` into
# the action's existing output directory, merges the chosen parameters
# into ``actions.configuration`` so they're a permanent property of
# the action row, marks the action_output's summary with
# ``committed: true`` so downstream pickers can filter, and flips the
# action's status from ``needs_threshold`` (or already-``complete`` on
# a re-commit) to ``complete``.
#
# Apply is unaffected — a committed prep still accepts further Apply
# calls. Users can re-explore and re-commit at any time; re-commit
# overwrites the canonical mask + metrics + configuration.


class AnomalyDetectionCommitRequest(BaseModel):
    threshold: float = Field(
        ...,
        description=(
            "Slider value to lock in. Same semantics as the preview "
            "endpoint: 'above pX' when threshold_mode='percentile', "
            "raw composite value when threshold_mode='absolute'."
        ),
    )
    threshold_mode: str = Field(default="percentile")
    dilation_kernel: int = Field(default=0, ge=0)


class AnomalyDetectionCommitResponse(BaseModel):
    """Wire shape returned after a successful commit."""

    threshold_absolute: float
    threshold_percentile: float
    dilation_kernel: int
    n_anomalous: int
    n_kept: int
    metrics: dict | None
    # Convenience pointer to the saved binary mask raster — frontend
    # can use this to surface a download link.
    mask_tif_path: str


@actions_router.post(
    "/{action_id}/anomaly_detection_commit",
    response_model=AnomalyDetectionCommitResponse,
    summary="Lock the chosen threshold on an anomaly_detection_prep action",
)
def anomaly_detection_commit(
    action_id: str,
    body: AnomalyDetectionCommitRequest,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> AnomalyDetectionCommitResponse:
    """Commit the user's chosen threshold on a prep action.

    Re-runs the exact preview math (so the saved mask matches what the
    viewer last showed) and writes:

      - ``<output_dir>/anomaly_mask.tif``  — uint8 binary geotiff
      - ``<output_dir>/metrics.json``      — threshold + dilation +
                                             P / R / F1 + TP/FP/FN
                                             (P/R/F1 only when GT
                                             attached)

    Then mutates the action row + its output row:

      - ``action.configuration.committed_threshold|mode|dilation`` set
      - ``action_output.summary.committed = true`` + the same params
      - ``action.status = "complete"``

    Re-commit is allowed — overwrites the prior mask + metrics + flags.
    """
    import json as _json

    import numpy as np
    import rasterio

    from ._anomaly_detection_preview import (
        compute_preview,
        load_gt_mask_for_action,
    )

    action = _action_or_404(action_id, db)
    if action.type != "anomaly_detection_prep":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="action_type_mismatch",
        )
    if action.status not in ("needs_threshold", "complete"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"action_not_committable (status={action.status!r})",
        )

    output = _output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    output_dir = (artifacts_root / output.artifact_path).resolve()
    composite_path = output_dir / "composite_score.tif"
    if not composite_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="composite_score_not_found",
        )

    # Optional GT — drives metrics in the commit payload the same way
    # Apply does.
    gt_mask = None
    annotation_id = (action.configuration or {}).get("input_annotation_id")
    if annotation_id:
        project = db.get(Project, action.project_id)
        scene_id_for_gt = str(project.scene_id) if project else None
        with rasterio.open(composite_path) as src:
            composite_shape = src.shape
        if scene_id_for_gt is not None:
            data_root = Path(settings.data_dir).resolve()
            gt_mask = load_gt_mask_for_action(
                data_root=data_root,
                scene_id=scene_id_for_gt,
                annotation_id=annotation_id,
                expected_shape=composite_shape,
            )

    try:
        result = compute_preview(
            composite_path=composite_path,
            threshold=body.threshold,
            threshold_mode=body.threshold_mode,
            dilation_kernel=body.dilation_kernel,
            gt_mask=gt_mask,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # --- Persist the binary mask raster -------------------------------
    # Recompute the mask at full resolution here (the preview's
    # rendered PNG was downsampled). Cheap on this same composite.
    with rasterio.open(composite_path) as src:
        composite = src.read(1).astype("float32", copy=False)
        profile = src.profile
    finite = np.isfinite(composite)
    if body.threshold_mode == "percentile":
        finite_vals = np.sort(composite[finite])
        if finite_vals.size == 0:
            absolute_threshold = float("inf")
        else:
            quantile_pct = max(0.0, min(100.0, float(body.threshold)))
            absolute_threshold = float(np.percentile(finite_vals, quantile_pct))
    else:
        absolute_threshold = float(body.threshold)

    raw_mask = (composite >= np.float32(absolute_threshold)) & finite
    if body.dilation_kernel and body.dilation_kernel > 1:
        from ._anomaly_detection_preview import _square_dilate
        raw_mask = _square_dilate(raw_mask, int(body.dilation_kernel)) & finite
    binary_mask = raw_mask.astype("uint8")

    profile.update(
        dtype="uint8",
        count=1,
        nodata=None,
        compress="deflate",
        predictor=2,
    )
    mask_tif_path_abs = output_dir / "anomaly_mask.tif"
    with rasterio.open(mask_tif_path_abs, "w", **profile) as dst:
        dst.write(binary_mask, 1)

    # --- Persist metrics.json -----------------------------------------
    metrics_payload: dict[str, Any] = {
        "threshold_absolute": result.threshold_absolute,
        "threshold_percentile": result.threshold_percentile,
        "threshold_mode": body.threshold_mode,
        "dilation_kernel": int(body.dilation_kernel),
        "n_anomalous": result.n_anomalous,
        "n_kept": result.n_kept,
        "has_gt": result.metrics is not None,
        "committed_at": datetime.utcnow().isoformat() + "Z",
    }
    if result.metrics:
        metrics_payload.update({
            "precision": result.metrics["precision"],
            "recall": result.metrics["recall"],
            "f1": result.metrics["f1"],
            "tp": result.metrics["tp"],
            "fp": result.metrics["fp"],
            "fn": result.metrics["fn"],
            "tn": result.metrics["tn"],
            "n_gt_positives": result.metrics["n_gt_positives"],
        })
    (output_dir / "metrics.json").write_text(_json.dumps(metrics_payload, indent=2))

    # --- Stamp the action + output rows -------------------------------
    cfg = dict(action.configuration or {})
    cfg["committed_threshold"] = float(body.threshold)
    cfg["committed_threshold_mode"] = body.threshold_mode
    cfg["committed_threshold_absolute"] = float(result.threshold_absolute)
    cfg["committed_dilation_kernel"] = int(body.dilation_kernel)
    action.configuration = cfg

    # Updating a JSONB column on a row that was loaded from another
    # transaction requires marking it modified explicitly when the
    # mutation is a dict-replace; SQLAlchemy's ORM picks up the
    # assignment above just fine, but the nested merge below needs a
    # flag_modified to make sure the change is written.
    from sqlalchemy.orm.attributes import flag_modified

    output_summary = dict(output.summary or {})
    output_summary["committed"] = True
    output_summary["committed_threshold"] = float(body.threshold)
    output_summary["committed_threshold_mode"] = body.threshold_mode
    output_summary["committed_threshold_absolute"] = float(result.threshold_absolute)
    output_summary["committed_dilation_kernel"] = int(body.dilation_kernel)
    if result.metrics:
        output_summary["committed_metrics"] = {
            "precision": result.metrics["precision"],
            "recall": result.metrics["recall"],
            "f1": result.metrics["f1"],
        }
    output.summary = output_summary
    flag_modified(output, "summary")

    action.status = "complete"
    action.completed_at = datetime.utcnow()
    db.commit()

    return AnomalyDetectionCommitResponse(
        threshold_absolute=float(result.threshold_absolute),
        threshold_percentile=float(result.threshold_percentile),
        dilation_kernel=int(body.dilation_kernel),
        n_anomalous=int(result.n_anomalous),
        n_kept=int(result.n_kept),
        metrics=result.metrics,
        mask_tif_path=f"/api/actions/{action_id}/files/anomaly_mask.tif",
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

    Spec § 5.5 originally deferred individual Action delete to Project-
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


@actions_router.get(
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
    action = _action_or_404(action_id, db)
    if action.type != "spectral_library_match":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="wrong_action_type",
        )
    output = _output_for_action(action.id, db)
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


@actions_router.post(
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

    action = _action_or_404(action_id, db)
    output = _output_for_action(action.id, db)
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
