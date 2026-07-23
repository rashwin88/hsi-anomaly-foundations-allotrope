"""create notes + note_references tables (Step 16)

Revision: 20260511_08
Down revision: 20260511_07
Created: 2026-05-11
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260511_08"
down_revision: Union[str, None] = "20260511_07"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
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
    op.create_index("notes_project_id_idx", "notes", ["project_id"])

    op.create_table(
        "note_references",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "note_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("notes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ref_project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "ref_action_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("actions.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "ref_output_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("action_outputs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "ref_viz_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("visualizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "ref_scene_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("scenes.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "(CASE WHEN ref_project_id IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_action_id  IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_output_id  IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_viz_id     IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_scene_id   IS NOT NULL THEN 1 ELSE 0 END) = 1",
            name="note_references_one_target_chk",
        ),
    )
    op.create_index("note_references_note_id_idx", "note_references", ["note_id"])
    op.create_index(
        "note_references_ref_project_id_idx", "note_references", ["ref_project_id"]
    )
    op.create_index(
        "note_references_ref_action_id_idx", "note_references", ["ref_action_id"]
    )
    op.create_index(
        "note_references_ref_output_id_idx", "note_references", ["ref_output_id"]
    )
    op.create_index(
        "note_references_ref_viz_id_idx", "note_references", ["ref_viz_id"]
    )
    op.create_index(
        "note_references_ref_scene_id_idx", "note_references", ["ref_scene_id"]
    )


def downgrade() -> None:
    for ix in (
        "note_references_ref_scene_id_idx",
        "note_references_ref_viz_id_idx",
        "note_references_ref_output_id_idx",
        "note_references_ref_action_id_idx",
        "note_references_ref_project_id_idx",
        "note_references_note_id_idx",
    ):
        op.drop_index(ix, table_name="note_references")
    op.drop_table("note_references")
    op.drop_index("notes_project_id_idx", table_name="notes")
    op.drop_table("notes")
