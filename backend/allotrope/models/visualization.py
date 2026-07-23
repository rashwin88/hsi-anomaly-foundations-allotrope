"""Visualization entity (abstractions-spec § 5.9, scoped 2026-05-11).

A curated, project-scoped persisted visual — what the user saves from any
viewer (Scene Detail, Action output) when they want to pin a specific
frame for later reference, link from a Note, or include in an Export.

Scope cut (2026-05-11): VisualizationTemplate is NOT in v1. Each Action
already ships its own viewer with sensible defaults; templates were the
weaker half of the original spec. Add later only if curation demand
surfaces a real need for reusable recipes.

- Ownership: Project (no separate user_id).
- Lifecycle: synchronous create via api; immutable source/artifact;
  mutable name/description; individually deletable.
- Storage: allotrope_artifacts/projects/<project_id>/visualizations/<id>/
- Polymorphic source: exactly one of source_scene_id or
  source_action_output_id is non-NULL; source_kind discriminates.
- view_state JSONB: free-form pin of viewer settings (which panel was
  active, panzoom transform, threshold sliders, overlay toggles, etc.).
  Per-viewer shape; api does no validation here — the producing viewer
  reads its own keys back at restore time.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Visualization(Base):
    __tablename__ = "visualizations"

    __table_args__ = (
        Index("visualizations_project_id_idx", "project_id"),
        Index("visualizations_source_scene_id_idx", "source_scene_id"),
        Index("visualizations_source_action_output_id_idx", "source_action_output_id"),
        CheckConstraint(
            "(source_kind = 'scene' AND source_scene_id IS NOT NULL AND source_action_output_id IS NULL) OR "
            "(source_kind = 'action_output' AND source_action_output_id IS NOT NULL AND source_scene_id IS NULL)",
            name="visualizations_source_xor_chk",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    source_kind: Mapped[str] = mapped_column(Text, nullable=False)

    # Exactly one of these is set; CHECK constraint above enforces.
    source_scene_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenes.id", ondelete="CASCADE"),
        nullable=True,
    )
    source_action_output_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("action_outputs.id", ondelete="CASCADE"),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relative to allotrope_artifacts. Directory holding the saved PNG +
    # any composed overlays the viewer flattened at save time.
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Viewer-specific pinned state — see module docstring.
    view_state: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<Visualization(id={self.id!s}, project_id={self.project_id!s}, "
            f"source_kind={self.source_kind!r}, name={self.name!r})>"
        )
