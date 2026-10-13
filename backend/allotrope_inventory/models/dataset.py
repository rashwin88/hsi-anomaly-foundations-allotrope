"""Dataset — one scene delivery, exactly as the provider shipped it.

Conventions (mirroring the main app's Scene):
- `(sensor_type, sensor_scene_id, processing_level)` is unique — the same
  delivery cannot be catalogued twice. L1 and L2D of one capture are two rows.
- Status is text, not an ENUM: `pending` (registered, uploading) →
  `available` (finalised, checksums verified) or `failed`.
- Footprint is EPSG:4326, any geometry type (scenes can be multipolygons).
- Every row is reproducible from `<s3_prefix>/manifest.json`; the bucket is
  the record of truth and this table is its index.
"""

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import DateTime, ForeignKey, Index, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    __table_args__ = (
        UniqueConstraint(
            "sensor_type", "sensor_scene_id", "processing_level",
            name="datasets_sensor_scene_level_uq",
        ),
        Index("datasets_status_idx", "status"),
        Index("datasets_acquisition_at_idx", "acquisition_at"),
        Index("datasets_footprint_idx", "footprint", postgresql_using="gist"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # --- Identity ---------------------------------------------------------
    sensor_type: Mapped[str] = mapped_column(
        Text, ForeignKey("sensors.slug", ondelete="RESTRICT"), nullable=False
    )
    sensor_scene_id: Mapped[str] = mapped_column(Text, nullable=False)
    processing_level: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'")
    )

    # --- Where and when ---------------------------------------------------
    acquisition_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # GIST index declared in __table_args__ (geoalchemy2's own is off so the
    # name follows the <table>_<col>_idx convention).
    footprint: Mapped[Any | None] = mapped_column(
        Geometry("GEOMETRY", srid=4326, spatial_index=False), nullable=True
    )

    # Key prefix inside the inventory bucket, no trailing slash:
    # `<sensor_type>/<processing_level>/ds_<uuid>`.
    s3_prefix: Mapped[str] = mapped_column(Text, nullable=False)

    # --- Sensor-specific extras (cloud cover, orbit, tile id, …) ---------
    # Python attr is `extra_metadata`; SQL column is "metadata", as in Scene.
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    # --- Audit ------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Label of the API key that registered it (keys carry a name).
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    finalized_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Dataset {self.sensor_type}/{self.sensor_scene_id} {self.processing_level} {self.status}>"
