"""
Lookup helpers shared by the Action endpoint modules.

`actions.py` grew past 1,300 lines and is being split by concern. These two
helpers are needed by more than one of the resulting modules, so they live here
rather than being imported across siblings or - worse - duplicated.

The bar for living here is being needed by more than one Action module -
nothing else. ActionOutputPublic qualifies: actions.py embeds it in ActionDetail
and action_files.py returns it directly. Response models used by exactly one
module stay with that module, so this never becomes the next dumping ground.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Action, ActionOutput
from .wireformat import parse_prefixed_id


class ActionOutputPublic(BaseModel):
    """Wire shape for ActionOutput rows."""

    id: str                          # output_<uuid>
    action_id: str                   # action_<uuid>
    artifact_path: str
    summary: dict[str, Any]
    created_at: datetime


def output_to_wire(o: ActionOutput) -> ActionOutputPublic:
    """ORM row -> wire shape, with ids given their `output_` / `action_` prefixes."""
    return ActionOutputPublic(
        id=f"output_{o.id}",
        action_id=f"action_{o.action_id}",
        artifact_path=o.artifact_path,
        summary=o.summary,
        created_at=o.created_at,
    )


def action_or_404(action_id_wire: str, db: Session) -> Action:
    """Resolve a prefixed `action_<uuid>` wire id to its row, or 404."""
    action_uuid = parse_prefixed_id("action", action_id_wire)
    action = db.get(Action, action_uuid)
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="action_not_found"
        )
    return action


def output_for_action(action_id: uuid.UUID, db: Session) -> ActionOutput | None:
    """
    The Action's single output row, if it has one.

    A `complete` Action has exactly one ActionOutput - enforced by the
    action_outputs_action_id_uniq constraint. `anomaly_detection_prep` is the
    documented exception: it writes an output and then parks at
    `needs_threshold` awaiting a human, so an output can exist before the
    action is complete.
    """
    return db.scalar(select(ActionOutput).where(ActionOutput.action_id == action_id))
