"""create jobs table

Revision: 20260510_03
Down revision: 20260510_02
Created: 2026-05-10

Creates the `jobs` table per abstractions-spec § 5.13. The Postgres-backed
work queue.

Notes:
- Worker claim path uses `SELECT … FOR UPDATE SKIP LOCKED` against this
  table — see allotrope_worker/runner.py.
- `target_id` is a soft reference (no FK) — polymorphic per § 7.
- `project_id` is a plain UUID column today; FK constraint to projects.id
  + ON DELETE CASCADE is added by the migration that creates `projects`
  (Step 10). Indexed already so the eventual FK doesn't require an index
  rebuild.
- Indexes:
    jobs_claim_idx       (status, type, created_at) — backs the worker's
                          claim query (filter by status+type, FIFO order).
    jobs_status_idx      single-column status — backs the Jobs UI list.
    jobs_project_id_idx  WHERE project_id IS NOT NULL — partial index, since
                          half the jobs are library-scoped (NULL).
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260510_03"
down_revision: Union[str, None] = "20260510_02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        # Type + status (text — not ENUM, so adding new types stays a
        # code-only change).
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
        # Optional project scope. FK + CASCADE arrives with the projects
        # table migration in Step 10.
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Inputs.
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        # Soft-ref output (no FK).
        sa.Column("target_kind", sa.Text(), nullable=True),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Failure / cancellation.
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        # Lifecycle timestamps.
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "last_heartbeat_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Attribution for stale jobs (set by worker on claim, cleared on
        # completion). String form: "<hostname>/<random-suffix>".
        sa.Column("claimed_by", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Composite index for the worker's claim query.
    op.create_index(
        "jobs_claim_idx",
        "jobs",
        ["status", "type", "created_at"],
    )
    # Single-status filter for the Jobs UI.
    op.create_index("jobs_status_idx", "jobs", ["status"])
    # Project filter for the Jobs UI; partial index since only project-scoped
    # job types populate this column.
    op.create_index(
        "jobs_project_id_idx",
        "jobs",
        ["project_id"],
        postgresql_where=sa.text("project_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("jobs_project_id_idx", table_name="jobs")
    op.drop_index("jobs_status_idx", table_name="jobs")
    op.drop_index("jobs_claim_idx", table_name="jobs")
    op.drop_table("jobs")
