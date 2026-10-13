"""create inventory tables

Revision: 20261013_01
Down revision: None
Created: 2026-10-13

Creates the whole raw-dataset inventory schema in one revision: sensors,
tags, datasets, dataset_files, dataset_tags, provenance, events.

Notes:
- PostGIS is enabled IF NOT EXISTS. On Supabase it already lives in the
  `extensions` schema (on the search_path), so this is a no-op there.
- (sensor_type, sensor_scene_id, processing_level) is UNIQUE — the same
  delivery cannot be catalogued twice; L1 and L2D are two rows.
- Statuses are TEXT, not ENUM — validated at the api layer.
- events.dataset_id is SET NULL on delete so the audit log survives.
- Seeds the five sensors the main app supports.
"""
from collections.abc import Sequence
from typing import Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20261013_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UUID = postgresql.UUID(as_uuid=True)
JSONB_EMPTY = sa.text("'{}'::jsonb")


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    sensors = op.create_table(
        "sensors",
        sa.Column("slug", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("family", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        _created_at(),
    )

    op.create_table(
        "tags",
        sa.Column("name", sa.Text(), primary_key=True),
        sa.Column("description", sa.Text(), nullable=True),
        _created_at(),
    )

    op.create_table(
        "datasets",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "sensor_type", sa.Text(),
            sa.ForeignKey("sensors.slug", ondelete="RESTRICT"), nullable=False,
        ),
        sa.Column("sensor_scene_id", sa.Text(), nullable=False),
        sa.Column("processing_level", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("acquisition_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "footprint",
            geoalchemy2.Geometry("GEOMETRY", srid=4326, spatial_index=False),
            nullable=True,
        ),
        sa.Column("s3_prefix", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=JSONB_EMPTY),
        _created_at(),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("finalized_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "sensor_type", "sensor_scene_id", "processing_level",
            name="datasets_sensor_scene_level_uq",
        ),
    )
    op.create_index("datasets_status_idx", "datasets", ["status"])
    op.create_index("datasets_acquisition_at_idx", "datasets", ["acquisition_at"])
    op.create_index(
        "datasets_footprint_idx", "datasets", ["footprint"], postgresql_using="gist"
    )

    op.create_table(
        "dataset_files",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "dataset_id", UUID,
            sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("upload_id", sa.Text(), nullable=True),
        sa.Column("s3_version_id", sa.Text(), nullable=True),
        _created_at(),
        sa.UniqueConstraint("dataset_id", "relative_path", name="dataset_files_path_uq"),
    )
    op.create_index("dataset_files_sha256_idx", "dataset_files", ["sha256"])

    op.create_table(
        "dataset_tags",
        sa.Column(
            "dataset_id", UUID,
            sa.ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True,
        ),
        sa.Column(
            "tag", sa.Text(),
            sa.ForeignKey("tags.name", ondelete="CASCADE", onupdate="CASCADE"),
            primary_key=True,
        ),
        sa.Column("created_by", sa.Text(), nullable=False),
        _created_at(),
    )
    op.create_index("dataset_tags_tag_idx", "dataset_tags", ["tag"])

    op.create_table(
        "provenance",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "dataset_id", UUID,
            sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column(
            "dataset_file_id", UUID,
            sa.ForeignKey("dataset_files.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("fetched_by", sa.Text(), nullable=False),
        sa.Column(
            "fetched_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("provenance_dataset_id_idx", "provenance", ["dataset_id"])

    op.create_table(
        "events",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "dataset_id", UUID,
            sa.ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=JSONB_EMPTY),
        _created_at(),
    )
    op.create_index("events_dataset_id_idx", "events", ["dataset_id"])
    op.create_index("events_created_at_idx", "events", ["created_at"])

    op.bulk_insert(
        sensors,
        [
            {"slug": "prisma", "name": "PRISMA", "family": "hyperspectral"},
            {"slug": "enmap", "name": "EnMAP", "family": "hyperspectral"},
            {"slug": "aviris_ng", "name": "AVIRIS-NG", "family": "hyperspectral"},
            {"slug": "landsat9", "name": "Landsat 9", "family": "thermal"},
            {"slug": "hotsat1", "name": "HotSat-1", "family": "thermal"},
        ],
    )


def downgrade() -> None:
    for table in (
        "events", "provenance", "dataset_tags", "dataset_files", "datasets", "tags", "sensors"
    ):
        op.drop_table(table)
    # PostGIS is left installed — other schemas may depend on it.
