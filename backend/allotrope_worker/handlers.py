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

import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

from allotrope.models import Job

from .action_run import handle_action_run
from .annotation_attach import handle_annotation_attach
from .project_export import handle_project_export
from .scene_onboard import handle_scene_onboard

# (target_kind, target_id) or (None, None) if there's no produced entity.
HandlerResult = tuple[str | None, uuid.UUID | None]
Handler = Callable[[Session, Job], HandlerResult]


HANDLERS: dict[str, Handler] = {
    "scene_onboard": handle_scene_onboard,
    "annotation_attach": handle_annotation_attach,
    "action_run": handle_action_run,
    "project_export": handle_project_export,
}


def supported_types() -> list[str]:
    return sorted(HANDLERS.keys())
