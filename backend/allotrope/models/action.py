"""Action entity (abstractions-spec § 5.5).

A verb taken on a Scene within a Project. The unit of investigation work.

- Ownership: Project (no user_id — inherits via project.user_id).
- Lifecycle: queued → running → complete | failed | cancelled. Created via
  api with status='queued' AND a paired `action_run` Job enqueued in the
  same transaction. Worker keeps actions.status in sync with jobs.status.
- Not individually deletable in v1 — cleanup via Project delete only.
- Cancel-while-queued: direct row update.
- Cancel-while-running: cancellation_requested=true; worker checks at
  recipe-step boundaries.
- Inputs (input_scene_id / input_action_output_ids / input_annotation_ids)
  live inside `configuration` JSONB. API validates references at submit
  time; no FK enforcement (referential safety comes from the cascade
  structure — Actions only delete via Project).
- Invariant: status='complete' ⇔ exactly one ActionOutput row exists.
- No updated_at: lifecycle uses started_at / completed_at.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Action(Base):
    __tablename__ = "actions"

    # Indexes that earn their keep:
    # - actions_project_id_idx — backs "list this project's actions"
    #   (Project workspace Action list pane, Step 12g).
    # - actions_status_idx — backs Jobs UI / status filters.
    # - actions_template_id_idx — only nullable column we'll filter on,
    #   for "actions created from this template" lookups when users delete
    #   a template.
    __table_args__ = (
        Index("actions_project_id_idx", "project_id"),
        Index("actions_status_idx", "status"),
        Index("actions_template_id_idx", "action_template_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # CASCADE: Project delete blast-radius is the entire investigation.
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Template that seeded this Action's configuration. NOT NULL at insert
    # time (api enforces) but nullable in storage so SET NULL on template
    # delete doesn't blow up — the configuration JSONB is the canonical
    # record of what ran; the template_id is just an audit trail of which
    # template was used.
    action_template_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("action_templates.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Action type. Text — same rationale as jobs.type and templates.type:
    # adding a new action type is a code-only change.
    type: Mapped[str] = mapped_column(Text, nullable=False)

    # The full recipe + input refs. Per-type Pydantic schema lives in the
    # action_types/<type>/ module; api validates against it at submit.
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    # Lifecycle.
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="queued"
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Worker sets started_at on transition to running, completed_at on the
    # terminal transition (complete/failed/cancelled).
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Cooperative cancellation flag. Worker polls at recipe-step boundaries.
    cancellation_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<Action(id={self.id!s}, project_id={self.project_id!s}, "
            f"type={self.type!r}, status={self.status!r})>"
        )
