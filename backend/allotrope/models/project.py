"""Project entity (abstractions-spec § 5.4).

A workspace bound to exactly one Scene — the structural pivot of the
application. Owned by a User. Ownership root for Actions, Visualizations,
Notes, and Exports (none of which exist yet — Step 12+).

Conventions locked in the spec:
- `user_id` and `scene_id` immutable. `name` and `description` mutable.
- User → Project: RESTRICT on user delete. Hard-delete a user with
  active projects is blocked at the DB layer.
- Scene → Project: RESTRICT on scene delete. Same rationale.
- Has `updated_at` (per CC-2a — Project is in the legitimately-mutable list).
- No states on the row — Project just exists + can be deleted (CC-15).
- Delete cascades down to Actions / Visualizations / Notes / Exports
  (those FKs are added by their own migrations as those entities land).
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Project(Base):
    __tablename__ = "projects"

    # Indexes that earn their keep:
    # - projects_user_id_idx — backs "list my projects" filter (Step 10c).
    # - projects_scene_id_idx — backs "list projects for this scene"
    #   (Scene Detail page surfaces existing projects on the same scene).
    __table_args__ = (
        Index("projects_user_id_idx", "user_id"),
        Index("projects_scene_id_idx", "scene_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )

    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenes.id", ondelete="RESTRICT"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

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
            f"<Project(id={self.id!s}, name={self.name!r}, "
            f"scene_id={self.scene_id!s})>"
        )
