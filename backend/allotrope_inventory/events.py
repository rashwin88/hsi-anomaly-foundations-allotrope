"""Audit log writer. Every mutating route calls `record` inside its own
transaction, so an event exists if and only if the change committed."""

import uuid
from typing import Any

from sqlalchemy.orm import Session

from .models import Event


def record(
    db: Session, kind: str, actor: str,
    dataset_id: uuid.UUID | None = None, **payload: Any,
) -> None:
    """Stage an `events` row on `db`; the caller's commit writes it."""
    db.add(Event(kind=kind, actor=actor, dataset_id=dataset_id, payload=payload))
