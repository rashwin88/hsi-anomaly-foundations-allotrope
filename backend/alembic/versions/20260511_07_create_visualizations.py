"""create visualizations table (Step 15, scoped — no template entity)

Revision: 20260511_07
Down revision: 20260510_06
Created: 2026-05-11

Adds the Visualization entity per abstractions-spec § 5.9, with the
VisualizationTemplate half deliberately omitted (scope cut 2026-05-11).
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260511_07"
down_revision: Union[str, None] = "20260510_06"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "visualizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column(
            "source_scene_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("scenes.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "source_action_output_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("action_outputs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column(
            "view_state",
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
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "(source_kind = 'scene' AND source_scene_id IS NOT NULL AND source_action_output_id IS NULL) OR "
            "(source_kind = 'action_output' AND source_action_output_id IS NOT NULL AND source_scene_id IS NULL)",
            name="visualizations_source_xor_chk",
        ),
    )
    op.create_index("visualizations_project_id_idx", "visualizations", ["project_id"])
    op.create_index(
        "visualizations_source_scene_id_idx", "visualizations", ["source_scene_id"]
    )
    op.create_index(
        "visualizations_source_action_output_id_idx",
        "visualizations",
        ["source_action_output_id"],
    )


def downgrade() -> None:
    op.drop_index("visualizations_source_action_output_id_idx", table_name="visualizations")
    op.drop_index("visualizations_source_scene_id_idx", table_name="visualizations")
    op.drop_index("visualizations_project_id_idx", table_name="visualizations")
    op.drop_table("visualizations")
