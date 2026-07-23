"""ActionOutput entity (abstractions-spec § 5.6).

The artifact every completed Action produces. 1:1 with its Action when
status='complete'.

- Identity: action_id is UNIQUE — exactly one Output per Action.
- Lifecycle: created in the same worker transaction that marks the
  Action complete. Immutable after creation.
- Storage: artifact_path is relative to allotrope_artifacts and points
  at the Output's directory:
      projects/<project_id>/actions/<action_id>/output/
  Per-action-type schema variation lives in the directory layout +
  summary JSONB. Single unified table — no per-type subtables.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ActionOutput(Base):
    __tablename__ = "action_outputs"

    # action_id UNIQUE enforces the 1:1 with Action when complete. The
    # backing index also serves "lookup output by action" without needing
    # an additional declaration.
    __table_args__ = (
        UniqueConstraint("action_id", name="action_outputs_action_id_uniq"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # CASCADE: Action delete (which only happens via Project delete) wipes
    # the Output row. The disk artifacts under artifact_path are removed
    # by the Project delete handler in the same operation — DB row
    # removal is the trigger, not the cleanup itself.
    action_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("actions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Directory holding the Output's artifacts (preview.png, mask layers,
    # processed cube pickle, …). Layout is per-action-type; the summary
    # JSONB names the files inside.
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Small structured payload: per-step metrics, kept_pct, threshold sets,
    # diagnostics. Big arrays go on disk under artifact_path.
    summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<ActionOutput(id={self.id!s}, action_id={self.action_id!s}, "
            f"artifact_path={self.artifact_path!r})>"
        )
