"""Note + NoteReference entities (abstractions-spec § 5.10 / 5.11).

Notes are markdown text scoped to a Project. NoteReferences are typed
pointers from a Note to a specific entity (Scene / Action / Output /
Visualization / Project) within the same Project — they're the wire
that lets a note cite its targets so cascades clean up automatically.

Lifecycle
- Notes: synchronous create / edit / delete via api. Mutable content +
  updated_at.
- NoteReferences: created and removed by the api in sync with the
  Note's content. Exactly one of the five ref_* columns is non-NULL
  (CHECK enforced). All FKs CASCADE so deleting any referenced entity
  drops the NoteReference row but leaves the Note intact.

The reference resolution itself (parsing @-mentions in markdown) is api
logic — the model is just the persisted edge.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Note(Base):
    __tablename__ = "notes"

    __table_args__ = (Index("notes_project_id_idx", "project_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<Note(id={self.id!s}, project_id={self.project_id!s})>"


class NoteReference(Base):
    __tablename__ = "note_references"

    __table_args__ = (
        Index("note_references_note_id_idx", "note_id"),
        Index("note_references_ref_project_id_idx", "ref_project_id"),
        Index("note_references_ref_action_id_idx", "ref_action_id"),
        Index("note_references_ref_output_id_idx", "ref_output_id"),
        Index("note_references_ref_viz_id_idx", "ref_viz_id"),
        Index("note_references_ref_scene_id_idx", "ref_scene_id"),
        CheckConstraint(
            "(CASE WHEN ref_project_id IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_action_id  IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_output_id  IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_viz_id     IS NOT NULL THEN 1 ELSE 0 END + "
            " CASE WHEN ref_scene_id   IS NOT NULL THEN 1 ELSE 0 END) = 1",
            name="note_references_one_target_chk",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    note_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("notes.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Polymorphic target — exactly one is non-NULL; CHECK enforces.
    ref_project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,
    )
    ref_action_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("actions.id", ondelete="CASCADE"),
        nullable=True,
    )
    ref_output_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("action_outputs.id", ondelete="CASCADE"),
        nullable=True,
    )
    ref_viz_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("visualizations.id", ondelete="CASCADE"),
        nullable=True,
    )
    ref_scene_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenes.id", ondelete="CASCADE"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<NoteReference(id={self.id!s}, note_id={self.note_id!s})>"
