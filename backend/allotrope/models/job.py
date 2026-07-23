"""Job entity (abstractions-spec § 5.13).

The Postgres-backed work queue. Worker pulls via:

    SELECT … FROM jobs
     WHERE status = 'queued'
       AND (type = ANY(:supported_types) OR :supported_types IS NULL)
     ORDER BY created_at
     FOR UPDATE SKIP LOCKED
     LIMIT 1;

`SKIP LOCKED` lets multiple worker replicas claim disjoint rows without
blocking each other — a row claimed by replica A is invisible to replica B's
SELECT until A's transaction ends.

Conventions locked in the spec:
- `target_id` is a **soft reference** — no FK. Rows persist after the target
  is deleted (audit history). `target_kind` discriminates.
- `project_id` is FK + CASCADE, nullable: project-scoped types
  (`action_run`, `project_export`) populate it; library-scoped types
  (`scene_onboard`, `annotation_attach`) leave it NULL.
- No generic `updated_at` — lifecycle uses named timestamps
  (`started_at`, `last_heartbeat_at`, `completed_at`).
- `claimed_by` (string worker id) is added pre-emptively so 6c's reaper
  + the future Jobs UI can attribute stale jobs to a worker. Not in the spec
  field list but consistent with the spec's intent for the reaper.
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
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Job(Base):
    __tablename__ = "jobs"

    # Indexes that earn their keep:
    # - jobs_claim_idx — backs the worker's claim query.
    #   (status, type, created_at) lets the planner narrow to queued+supported
    #   then walk by FIFO order.
    # - jobs_status_idx — backs the Jobs UI's status-filtered list.
    # - jobs_project_id_idx — backs project-scoped filters in the Jobs UI.
    __table_args__ = (
        Index(
            "jobs_claim_idx",
            "status",
            "type",
            "created_at",
        ),
        Index("jobs_status_idx", "status"),
        Index("jobs_project_id_idx", "project_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # --- Type + status ---------------------------------------------------
    # Both stored as text (not Postgres ENUM) — easy to add new types without
    # ALTER TYPE migrations. Validated at the api/worker layer.
    type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'queued'"),
    )

    # --- Optional project scope (NULL for library-scoped jobs) -----------
    # FK ref to projects.id deferred until the projects table exists
    # (Step 10). For now, plain UUID column with a comment + helper index.
    # When the projects table lands, we'll add the FK in that migration.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # --- Inputs ----------------------------------------------------------
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    # --- Output (soft-ref polymorphism) ----------------------------------
    target_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # --- Failure / cancellation ------------------------------------------
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancellation_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )

    # --- Lifecycle timestamps -------------------------------------------
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # --- Worker attribution (for 6c reaper + Jobs UI) --------------------
    # Identity of the worker holding the row while status='running'. Cleared
    # when the worker completes or fails the job. The reaper uses the
    # heartbeat timestamp (not this column) to decide stale; this column is
    # for *attribution* once stale.
    claimed_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Job(id={self.id!s}, type={self.type!r}, "
            f"status={self.status!r})>"
        )
