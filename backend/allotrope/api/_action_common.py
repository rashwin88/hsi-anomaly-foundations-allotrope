"""
Lookup helpers shared by the Action endpoint modules.

`actions.py` grew past 1,300 lines and is being split by concern. These two
helpers are needed by more than one of the resulting modules, so they live here
rather than being imported across siblings or - worse - duplicated.

Deliberately narrow: only helpers with no schema dependencies belong here. The
Pydantic response models and the helpers that build them stay with the endpoints
that own them, so this module never becomes the next dumping ground.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Action, ActionOutput
from .wireformat import parse_prefixed_id


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
