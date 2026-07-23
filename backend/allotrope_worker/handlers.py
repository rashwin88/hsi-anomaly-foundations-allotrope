"""Per-type job handler dispatch.

A handler signature is:
    (session, job) -> (target_kind, target_id) on success
                    -> raise on failure

The runner owns transaction lifecycle: handlers may `session.add(...)` /
`session.flush()` / `session.execute(...)` freely but MUST NOT commit or
rollback. The runner's `mark_complete` / `mark_failed` issues the COMMIT
that publishes both the handler's writes and the lifecycle transition.

Real handlers land progressively:
    scene_onboard      → Step 7c (plumbing-first; metadata placeholder)
    annotation_attach  → Step 9
    action_run         → Step 12c
    project_export     → Step 17
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

from allotrope.models import Job

from .action_run import handle_action_run
from .annotation_attach import handle_annotation_attach
from .project_export import handle_project_export
from .scene_onboard import handle_scene_onboard

logger = logging.getLogger("allotrope.worker.handlers")

# (target_kind, target_id) or (None, None) if there's no produced entity.
HandlerResult = tuple[str | None, uuid.UUID | None]
Handler = Callable[[Session, Job], HandlerResult]


def _placeholder(kind: str) -> Handler:
    """Build a placeholder handler tagged with its job type.

    Used for the job types whose real implementation hasn't landed yet.
    """

    def run(_session: Session, job: Job) -> HandlerResult:
        logger.info(
            "[%s] placeholder handler running for job %s (payload keys: %s)",
            kind,
            job.id,
            sorted((job.payload or {}).keys()),
        )
        sleep_sec = float(os.environ.get("WORKER_PLACEHOLDER_SLEEP_SEC", "0.5"))
        time.sleep(sleep_sec)
        return (None, None)

    return run


HANDLERS: dict[str, Handler] = {
    "scene_onboard": handle_scene_onboard,
    "annotation_attach": handle_annotation_attach,
    "action_run": handle_action_run,
    "project_export": handle_project_export,
}


def supported_types() -> list[str]:
    return sorted(HANDLERS.keys())
