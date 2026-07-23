"""Worker handler for `action_run` jobs (Step 12d).

Dispatches every Action through the `action_types` registry. The handler
itself owns:

  - Loading the Action row by `job.target_id`.
  - Lockstep status transitions: queued → running (committed early so
    the UI sees the transition) → complete | failed.
  - Building an `ActionRunContext` that carries paths, scene metadata,
    config, and an `on_step` heartbeat callback into the type module's
    `run` function.
  - Materialising the `action_outputs` row alongside `actions.status='complete'`
    in the runner's transaction.

Per-action-type math lives in `backend/allotrope/action_types/<kind>.py::run`.
Heavy imports (numpy, rasterio, app/) happen inside those functions, not
at module load time, so this dispatch module stays cheap to import.

Sequence diagram: final design/diagrams/action-worker-run.drawio
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from allotrope import action_types as action_types_registry
from allotrope.config import settings
from allotrope.models import Action, ActionOutput, Job, Project, Scene

logger = logging.getLogger("allotrope.worker.action_run")


# --- Context object passed into per-type run() functions --------------


@dataclass
class ActionRunContext:
    """Everything an action-type's `run` function needs from the worker.

    Constructed by `handle_action_run` after the Action row is loaded
    and the output directory is created. Type modules don't talk to the
    DB directly — they receive their inputs already-resolved here.
    """

    # Identity
    action_id: uuid.UUID
    project_id: uuid.UUID
    scene_id: uuid.UUID
    sensor_type: str

    # The validated configuration body from `actions.configuration`.
    configuration: dict[str, Any]

    # Filesystem locations (absolute paths inside the worker container).
    data_dir: Path                # /data
    artifacts_dir: Path           # /artifacts
    output_dir: Path              # absolute path to projects/<id>/actions/<id>/output/
    output_dir_rel: str           # same path, relative to artifacts_dir — what we persist

    # Convenience: resolved absolute path to the Scene's raw file.
    scene_raw_path: Path

    # Callback the type module calls between recipe steps. Triggers a
    # heartbeat (worker side will refresh jobs.last_heartbeat_at) and
    # logs the step name. No-op today; wired into Heartbeat in 12d/e.
    on_step: Callable[[str], None] = field(default=lambda _msg: None)

    # Lookup helper for `input_action_output_id` references — given a
    # wire-format output_<uuid>, returns the absolute artifacts directory
    # containing that Action's pickle / rasters / etc.
    resolve_action_output: Callable[[str], Path] = field(
        default=lambda _wire: Path()
    )


# --- Helpers -----------------------------------------------------------


def _output_dir_for_action(project_id: uuid.UUID, action_id: uuid.UUID) -> str:
    """Relative-to-artifacts path. Storage layout per abstractions-spec § 5.6."""
    return f"projects/{project_id}/actions/{action_id}/output"


def _make_resolve_action_output(
    session: Session, project_id: uuid.UUID, artifacts_root: Path
) -> Callable[[str], Path]:
    """Build the resolver `ActionRunContext.resolve_action_output`.

    Given a wire-format `output_<uuid>`, returns the absolute path to the
    referenced ActionOutput's artifacts directory. Verifies the output
    exists and belongs to the same project (defence in depth — the api
    already enforced this at submit time, but the worker shouldn't trust
    the configuration blindly).
    """

    def resolve(wire_id: str) -> Path:
        if not wire_id.startswith("output_"):
            raise ValueError(f"not a wire-format output id: {wire_id!r}")
        raw_uuid = uuid.UUID(wire_id[len("output_"):])
        output = session.get(ActionOutput, raw_uuid)
        if output is None:
            raise ValueError(f"action output not found: {wire_id}")
        ref_action = session.get(Action, output.action_id)
        if ref_action is None or ref_action.project_id != project_id:
            raise ValueError(
                f"action output {wire_id} does not belong to project {project_id}"
            )
        return artifacts_root / output.artifact_path

    return resolve


# --- Mark Action running (early commit) -------------------------------


def _mark_action_running(session: Session, action: Action) -> None:
    """Commit the queued → running transition so the UI sees it.

    The runner's `mark_complete` / `mark_failed` won't fire until the
    handler returns or raises — without this early commit, the actions
    row stays at status='queued' until the entire recipe finishes. That
    erases the "running" state from the UI's perspective.
    """
    action.status = "running"
    action.started_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(action)


# --- Main handler ------------------------------------------------------


def handle_action_run(
    session: Session, job: Job
) -> tuple[str | None, uuid.UUID | None]:
    """Dispatch an `action_run` job to the right action_types module.

    Contract with the runner (per `handlers.py`):
      - Returns (target_kind, target_id) on success. Runner commits.
      - Raises on failure. Runner calls mark_failed and rolls back.

    The Action row's lifecycle is mirrored alongside the Job's:
      - Pre-handler: status='queued'
      - Early commit by this handler: status='running' + started_at
      - On success: status='complete' + completed_at, persisted by the
        runner's mark_complete.
      - On exception: status='failed' + failure_reason, persisted in the
        except branch below before re-raising.
    """
    if job.target_id is None or job.target_kind != "action":
        raise RuntimeError(
            f"action_run job missing target_kind=action / target_id; "
            f"target_kind={job.target_kind!r} target_id={job.target_id!r}"
        )

    action = session.get(Action, job.target_id)
    if action is None:
        raise RuntimeError(
            f"action_run job target_id={job.target_id} does not match any actions.id"
        )

    project = session.get(Project, action.project_id)
    if project is None:
        raise RuntimeError(
            f"action {action.id} references missing project {action.project_id}"
        )

    scene = session.get(Scene, project.scene_id)
    if scene is None:
        raise RuntimeError(
            f"project {project.id} references missing scene {project.scene_id}"
        )

    # Resolve the action type module up front — fail early if the type
    # was renamed between submit and run.
    try:
        spec = action_types_registry.get_spec(action.type)
    except KeyError as e:
        raise RuntimeError(
            f"action {action.id} has unknown type {action.type!r}; "
            f"registry has {action_types_registry.supported_kinds()}"
        ) from e

    # Mark queued → running with an early commit.
    _mark_action_running(session, action)
    logger.info(
        "action started action=%s type=%s project=%s scene=%s",
        action.id,
        action.type,
        project.id,
        scene.id,
    )

    # Build paths.
    data_root = Path(settings.data_dir)
    artifacts_root = Path(settings.artifacts_dir)
    output_dir_rel = _output_dir_for_action(project.id, action.id)
    output_dir_abs = artifacts_root / output_dir_rel
    output_dir_abs.mkdir(parents=True, exist_ok=True)

    scene_raw_path = data_root / scene.raw_path

    ctx = ActionRunContext(
        action_id=action.id,
        project_id=project.id,
        scene_id=scene.id,
        sensor_type=scene.sensor_type,
        configuration=action.configuration,
        data_dir=data_root,
        artifacts_dir=artifacts_root,
        output_dir=output_dir_abs,
        output_dir_rel=output_dir_rel,
        scene_raw_path=scene_raw_path,
        on_step=lambda msg: logger.info(
            "action=%s step=%s", action.id, msg
        ),
        resolve_action_output=_make_resolve_action_output(
            session, project.id, artifacts_root
        ),
    )

    # Run the recipe. Type modules are responsible for raising on
    # internal failures; we wrap so a failure here marks the action
    # failed before the runner sees it.
    try:
        spec.run(ctx)
        summary = spec.summarize(ctx, output_dir_abs)
        spec.preview(ctx, output_dir_abs)
    except Exception as e:
        # Mirror the failure onto the actions row so the workspace UI
        # can render the failure_reason on the Action card. Commit
        # immediately — the runner's mark_failed handles the Job side
        # but doesn't know about the Action row.
        action.status = "failed"
        action.completed_at = datetime.now(timezone.utc)
        action.failure_reason = f"{e.__class__.__name__}: {e}"[:1000]
        session.commit()
        logger.exception("action failed action=%s", action.id)
        raise

    # Materialise the ActionOutput. Single transaction with the
    # Action's terminal-status transition; the runner's mark_complete
    # commits both alongside the Job's complete transition.
    output = ActionOutput(
        id=uuid.uuid4(),
        action_id=action.id,
        artifact_path=output_dir_rel,
        summary=summary,
    )
    session.add(output)
    # Action types that have an interactive phase after the worker
    # finishes (e.g. anomaly_detection_prep, which waits for the user
    # to dial in a threshold) declare a non-"complete" terminal status
    # via a module-level `TERMINAL_STATUS` attribute. Default is
    # "complete" — preserves behaviour for every existing type.
    terminal_status = getattr(spec, "TERMINAL_STATUS", "complete")
    action.status = terminal_status
    action.completed_at = datetime.now(timezone.utc)
    session.flush()  # surface UNIQUE(action_id) violations now, not at commit

    logger.info(
        "action complete action=%s output=%s artifact_path=%s",
        action.id,
        output.id,
        output_dir_rel,
    )

    # Runner's mark_complete uses these to populate jobs.target_kind /
    # target_id with the produced ActionOutput row.
    return ("action_output", output.id)
