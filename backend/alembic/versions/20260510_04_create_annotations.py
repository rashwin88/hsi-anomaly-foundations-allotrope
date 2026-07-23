"""create annotations table

Revision: 20260510_04
Down revision: 20260510_03
Created: 2026-05-10

Creates the `annotations` table per abstractions-spec § 5.3.

Notes:
- FK to scenes.id with ON DELETE CASCADE — Annotations are owned by their
  Scene and cleaned up when the Scene is deleted.
- FK to users.id with ON DELETE SET NULL — created_by is audit-only.
- v1: only `type='raster_mask'`. No CHECK constraint to keep migrations
  forward-compatible; type is validated at the api/worker layer.
- Annotation rows are write-once; no `updated_at`.
- `metadata` JSONB column is exposed in Python as `extra_metadata` to
  avoid SQLAlchemy's reserved name (same pattern as scenes).
- One index: annotations_scene_id_idx — for the Scene Detail page's
  "annotations for this scene" lookup (Step 8).

Companion: `scenes.has_annotations` is maintained at the app level in the
same transaction as the Annotation insert/delete (CC-12).
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260510_04"
down_revision: Union[str, None] = "20260510_03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "annotations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scene_id", postgresql.UUID(as_uuid=True), nullable=False),
        # v1: only 'raster_mask'. Validated in api/worker, not in DB.
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # Relative to allotrope_data:
        #   scenes/<scene_id>/annotations/<annotation_id>/<filename>
        sa.Column("file_path", sa.Text(), nullable=False),
        # JSONB extras — class label map for multi-class masks, provenance.
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        # Audit.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_by_user_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["scene_id"],
            ["scenes.id"],
            name="annotations_scene_fk",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="annotations_created_by_fk",
            ondelete="SET NULL",
        ),
    )

    op.create_index(
        "annotations_scene_id_idx",
        "annotations",
        ["scene_id"],
    )


def downgrade() -> None:
    op.drop_index("annotations_scene_id_idx", table_name="annotations")
    op.drop_table("annotations")
