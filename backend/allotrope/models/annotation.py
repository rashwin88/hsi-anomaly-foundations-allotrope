"""Annotation entity (abstractions-spec § 5.3).

Optional artifact attached to a Scene. v1 supports `type='raster_mask'` only.

Conventions locked in the spec:
- Lifecycle: created via `scene_onboard` (bundled with the scene) or
  `annotation_attach` (post-hoc). **Option B** — row exists only after the
  job succeeds. Synchronous *delete* via api (Step 9 wires the route).
- Fully immutable. No `updated_at`.
- `scene_id` is FK with CASCADE — deleting a Scene removes its Annotations.
- `created_by_user_id` is SET NULL on user delete (audit-only, not ownership).
- `file_path` is relative to the `allotrope_data` volume mount.
- Annotation insert/delete and `scenes.has_annotations` are maintained in
  the same transaction at the api/worker layer (CC-12).

The Python attr for the JSONB `metadata` column is `extra_metadata` to
avoid SA's reserved `metadata` name — same pattern as Scene.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Annotation(Base):
    __tablename__ = "annotations"

    # Index pays off when the api needs to list annotations for a scene
    # (Scene Detail page, Step 8). Single-column on scene_id is enough —
    # we expect only a handful of annotations per scene.
    __table_args__ = (
        Index("annotations_scene_id_idx", "scene_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenes.id", ondelete="CASCADE"),
        nullable=False,
    )

    # v1: 'raster_mask'. Plain text (not Postgres ENUM) for the same reason
    # job.type uses text — adding new types stays a code-only change.
    type: Mapped[str] = mapped_column(Text, nullable=False)

    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relative to allotrope_data mount, e.g.
    #   scenes/<scene_id>/annotations/<annotation_id>/<filename>
    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Class label map for multi-class masks, source/provenance, etc.
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

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
            f"<Annotation(id={self.id!s}, scene_id={self.scene_id!s}, "
            f"type={self.type!r})>"
        )
