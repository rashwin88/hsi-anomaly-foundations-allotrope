"""Action endpoints (Step 12c).

Routes:
    POST   /projects/{project_id}/actions  â€” submit an Action + enqueue its `action_run` Job
    GET    /projects/{project_id}/actions  â€” list Actions for a project (paginated)
    GET    /actions/{action_id}            â€” detail (includes ActionOutput when complete)
    GET    /action-types                   â€” public catalog (drives picker + Action card)

Submit flow (one DB transaction):
    1. Auth + parse project_id.
    2. Load Project + Scene (Scene needed for sensor_type).
    3. Look up the action type module from the registry (404 on unknown type).
    4. Run module.validate_config(raw_cfg, sensor_type) â€” Pydantic 422 on shape errors.
    5. Cross-field semantic checks the type module can't do alone:
         - configuration.input_scene_id must equal project.scene_id
         - configuration.input_band_filter_output_id (when present) must
           reference a `complete` ActionOutput from the same project.
    6. Optional: validate action_template_id exists.
    7. INSERT Action(status='queued') + INSERT Job(type='action_run',
       target_kind='action', target_id=action.id, project_id, payload).
    8. Return ActionPublic.

Worker keeps actions.status and jobs.status in lockstep at transaction
boundaries (Step 12d). The api never mutates lifecycle state directly â€”
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
from ._action_common import action_or_404, output_for_action
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.actions")

# Two routers â€” one for project-nested (submit + list), one flat for
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
        description="Action type slug â€” must be a key in the action_types registry.",
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

    # 3a. input_scene_id is the project's bound Scene â€” always.
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
        # field name â†’ required producing action type (None = any type)
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
    #    submit time" â€” the configuration is the canonical record.
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
      empty names) â€” basename-only access is enforced.

    Path-traversal defence is two-fold: filename is rejected if it
    contains `/` or `\\` or `..` segments; the resolved absolute path
    is then asserted to live inside the artifacts root.
    """
    action = action_or_404(action_id, db)

    # Filename must be a single basename â€” no traversal, no nesting.
    if not filename or "/" in filename or "\\" in filename or filename in ("..", "."):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_filename",
        )
    if ".." in filename.split("."):
        # paranoid catch â€” shouldn't trigger because of the slash check above
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_filename",
        )

    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    full = (artifacts_root / output.artifact_path / filename).resolve()
    # Defence in depth â€” confine to artifacts root.
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
        # Action artifacts are write-once + immutable per Â§ 5.6 â€” cache aggressively.
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
    action = action_or_404(action_id, db)

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

    output = output_for_action(action.id, db)
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
# NewActionDialog when wiring an anomaly_detection_prep â€” once the user
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
    """Resolve ``output_<uuid>`` â†’ its ActionOutput row (incl. action_id)."""
    raw_id = parse_prefixed_id("output", output_id)
    output = db.get(ActionOutput, raw_id)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_found",
        )
    return _output_to_wire(output)


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

    - status='running' is rejected with 409 â€” the worker still owns the
      row's lifecycle. Use the cancellation flag and wait for the
      terminal transition before deleting.
    - ActionOutput CASCADEs via its FK; Visualizations sourced from the
      ActionOutput CASCADE via theirs.
    - Job rows (target_kind='action', target_id=action.id) carry a SOFT
      ref â€” they stay as audit history with a dangling target_id, which
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
    # via visualizations.source_action_output_id â†’ action_outputs.id â†’
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
