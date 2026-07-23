"""create action_templates + actions + action_outputs tables

Revision: 20260510_06
Down revision: 20260510_05
Created: 2026-05-10

Adds the three Action-related tables per abstractions-spec § 5.5 / 5.6 / 5.7,
in dependency order:

  1. action_templates  — system-shared recipes; FKs target it from actions
  2. actions           — work units inside a project (FK projects + templates)
  3. action_outputs    — 1:1 artifact per completed action (UNIQUE on action_id)

Job-side wiring: jobs already carry a soft polymorphic reference
(target_kind + target_id) for the worker's claim flow, so no hard
`jobs.action_id` FK is needed. action_run jobs use
target_kind='action' + target_id=<action_uuid>; on Project delete,
CASCADE wipes both Actions and Jobs (jobs.project_id FK already CASCADEs
per Step 10's migration), and any jobs referencing deleted actions via
target_id become harmless orphans.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260510_06"
down_revision: Union[str, None] = "20260510_05"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- action_templates ---------------------------------------------
    # System-shared. Deletion of a template fires SET NULL on
    # actions.action_template_id (preserves the audit trail of which
    # template was used). Bootstrap seeds one per (type × applicable
    # sensor); see the seed-action-templates CLI in Step 12e.
    op.create_table(
        "action_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "is_system",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("action_templates_type_idx", "action_templates", ["type"])

    # --- actions -------------------------------------------------------
    # CASCADE on projects: Project delete removes all its Actions.
    # SET NULL on action_templates: template delete preserves the
    # audit row but clears the reference. configuration jsonb is the
    # canonical record of what ran.
    op.create_table(
        "actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "action_template_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column(
            "configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="actions_project_fk",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["action_template_id"],
            ["action_templates.id"],
            name="actions_template_fk",
            ondelete="SET NULL",
        ),
    )
    op.create_index("actions_project_id_idx", "actions", ["project_id"])
    op.create_index("actions_status_idx", "actions", ["status"])
    op.create_index(
        "actions_template_id_idx",
        "actions",
        ["action_template_id"],
    )

    # --- action_outputs ------------------------------------------------
    # UNIQUE on action_id enforces the 1:1 invariant from § 5.6
    # ("status='complete' ⇔ exactly one ActionOutput row exists").
    # CASCADE on actions: Action delete (only happens via Project delete
    # in v1) cleans up the Output row. Disk artifacts under artifact_path
    # are removed by the Project delete handler in the same operation.
    op.create_table(
        "action_outputs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column(
            "summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("action_id", name="action_outputs_action_id_uniq"),
        sa.ForeignKeyConstraint(
            ["action_id"],
            ["actions.id"],
            name="action_outputs_action_fk",
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    op.drop_table("action_outputs")
    op.drop_index("actions_template_id_idx", table_name="actions")
    op.drop_index("actions_status_idx", table_name="actions")
    op.drop_index("actions_project_id_idx", table_name="actions")
    op.drop_table("actions")
    op.drop_index("action_templates_type_idx", table_name="action_templates")
    op.drop_table("action_templates")
