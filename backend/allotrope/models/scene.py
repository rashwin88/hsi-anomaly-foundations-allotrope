"""Scene entity (abstractions-spec § 5.2).

Field set matches the spec exactly.

Conventions locked there:
- bbox columns are always EPSG:4326 (no per-row CRS column).
- Timestamps are timestamptz in UTC (incoming naïve datetimes get normalized
  to UTC at the api layer before insert).
- Scene is write-once: no `updated_at`. Status of in-flight onboarding
  lives in the `jobs` table, NOT here (lifecycle option B).
- `(sensor_type, sensor_scene_id)` is unique — prevents re-onboarding the
  same physical capture twice.
- `created_by_user_id` is audit-only (no ownership). Library is shared.
  SET NULL on user delete so a deleted user doesn't take the scene with them.

The Python attribute for the JSONB `metadata` column is named
`extra_metadata` because `metadata` is reserved on SQLAlchemy's Base.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Scene(Base):
    __tablename__ = "scenes"

    __table_args__ = (
        UniqueConstraint(
            "sensor_type",
            "sensor_scene_id",
            name="scenes_sensor_scene_uq",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Sensor + sensor-provided id (pair is unique).
    sensor_type: Mapped[str] = mapped_column(Text, nullable=False)
    sensor_scene_id: Mapped[str] = mapped_column(Text, nullable=False)

    # Display name; default at the api layer to sensor_scene_id when not set.
    name: Mapped[str] = mapped_column(Text, nullable=False)

    # --- Harmonized metadata (from STAC parsing during onboarding) -------
    acquisition_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    processing_level: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_type: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Bounding box always in EPSG:4326 — convention locked in spec.
    bbox_min_lon: Mapped[Decimal] = mapped_column(Numeric(), nullable=False)
    bbox_min_lat: Mapped[Decimal] = mapped_column(Numeric(), nullable=False)
    bbox_max_lon: Mapped[Decimal] = mapped_column(Numeric(), nullable=False)
    bbox_max_lat: Mapped[Decimal] = mapped_column(Numeric(), nullable=False)

    native_projection: Mapped[str | None] = mapped_column(Text, nullable=True)
    band_count: Mapped[int] = mapped_column(Integer, nullable=False)

    cloud_cover_pct: Mapped[Decimal | None] = mapped_column(Numeric(), nullable=True)
    valid_pixel_pct: Mapped[Decimal | None] = mapped_column(Numeric(), nullable=True)

    # --- Denormalized for the "has ground truth" filter facet ------------
    has_annotations: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )

    # --- Sensor-specific extras (variable per sensor type) ---------------
    # Python attr is `extra_metadata` to avoid colliding with SA's reserved
    # `metadata` name; SQL column is "metadata" per spec.
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    # --- File references (relative paths within named volumes) -----------
    raw_path: Mapped[str] = mapped_column(Text, nullable=False)
    vendable_path: Mapped[str] = mapped_column(Text, nullable=False)
    thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Audit / lifecycle ----------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Scene(id={self.id!s}, sensor_type={self.sensor_type!r}, "
            f"sensor_scene_id={self.sensor_scene_id!r})>"
        )
