"""Sensor — one row per instrument the inventory knows about.

Keyed by slug (`prisma`, `enmap`, …) — the same strings the main app stores
in `scenes.sensor_type` — so new sensors are a row insert, not a code change.
"""

from datetime import datetime

from sqlalchemy import DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Sensor(Base):
    __tablename__ = "sensors"

    slug: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # `hyperspectral` | `thermal` | … — free text, validated at the api layer.
    family: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
