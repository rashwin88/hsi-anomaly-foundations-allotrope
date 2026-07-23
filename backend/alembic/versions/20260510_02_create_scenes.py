"""create scenes table

Revision: 20260510_02
Down revision: 20260510_01
Created: 2026-05-10

Creates the `scenes` table per abstractions-spec § 5.2.

Notes:
- (sensor_type, sensor_scene_id) is UNIQUE — prevents re-onboarding the
  same physical capture.
- bbox columns store EPSG:4326 lat/lon as NUMERIC (no PostGIS in v1).
- `metadata` column holds sensor-specific extras as JSONB.
- Scene is write-once at the row level (no updated_at). In-flight
  onboarding lives in the jobs table per lifecycle option B.
- created_by_user_id is audit-only with SET NULL on user delete (CC-3 / CC-4).
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260510_02"
down_revision: Union[str, None] = "20260510_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scenes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        # Sensor identity
        sa.Column("sensor_type", sa.Text(), nullable=False),
        sa.Column("sensor_scene_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        # Harmonized metadata
        sa.Column("acquisition_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_level", sa.Text(), nullable=True),
        sa.Column("product_type", sa.Text(), nullable=True),
        # Bounding box (EPSG:4326)
        sa.Column("bbox_min_lon", sa.Numeric(), nullable=False),
        sa.Column("bbox_min_lat", sa.Numeric(), nullable=False),
        sa.Column("bbox_max_lon", sa.Numeric(), nullable=False),
        sa.Column("bbox_max_lat", sa.Numeric(), nullable=False),
        sa.Column("native_projection", sa.Text(), nullable=True),
        sa.Column("band_count", sa.Integer(), nullable=False),
        sa.Column("cloud_cover_pct", sa.Numeric(), nullable=True),
        sa.Column("valid_pixel_pct", sa.Numeric(), nullable=True),
        # Denormalized
        sa.Column(
            "has_annotations",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        # Sensor-specific extras
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        # File paths
        sa.Column("raw_path", sa.Text(), nullable=False),
        sa.Column("vendable_path", sa.Text(), nullable=False),
        sa.Column("thumbnail_path", sa.Text(), nullable=True),
        # Audit
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
        sa.UniqueConstraint(
            "sensor_type",
            "sensor_scene_id",
            name="scenes_sensor_scene_uq",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="scenes_created_by_fk",
            ondelete="SET NULL",
        ),
    )

    # Indexes that pay off for the Library page's typical queries:
    # - sort by recency (created_at DESC)
    # - filter by sensor type
    # - filter by has_annotations
    op.create_index("scenes_created_at_idx", "scenes", ["created_at"])
    op.create_index("scenes_sensor_type_idx", "scenes", ["sensor_type"])
    op.create_index("scenes_has_annotations_idx", "scenes", ["has_annotations"])


def downgrade() -> None:
    op.drop_index("scenes_has_annotations_idx", table_name="scenes")
    op.drop_index("scenes_sensor_type_idx", table_name="scenes")
    op.drop_index("scenes_created_at_idx", table_name="scenes")
    op.drop_table("scenes")
