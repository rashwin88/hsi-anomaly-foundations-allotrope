"""ActionTemplate entity (abstractions-spec § 5.7).

A reusable recipe for Action runs. System-shared (no `user_id`,
no `project_id`).

- Seeded at bootstrap: one default per (action type × applicable sensor),
  marked `is_system=True` and treated as read-only by the api.
- Users save successful Action configs as new templates → `is_system=False`,
  editable.
- Deletion fires SET NULL on `actions.action_template_id` so the audit
  trail (which template was used) survives template churn — see Action.

Mutability per CC-2a: `name`, `description`, `configuration` mutable for
user templates. `type` and `is_system` immutable. Has `updated_at`.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ActionTemplate(Base):
    __tablename__ = "action_templates"

    # Filter by type for "templates available for this Action type" picker.
    __table_args__ = (
        Index("action_templates_type_idx", "type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Action type the template applies to. Text (not enum) so adding new
    # action types stays a code-only change — same pattern as jobs.type.
    type: Mapped[str] = mapped_column(Text, nullable=False)

    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The recipe payload. Schema is governed by the action type's Pydantic
    # config model (validated at api submit time, not at DB write time).
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    # System-seeded vs user-saved. is_system=True templates are read-only
    # in the UI. Sensor-keyed defaults all ship as is_system=True.
    is_system: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
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
            f"<ActionTemplate(id={self.id!s}, type={self.type!r}, "
            f"name={self.name!r}, is_system={self.is_system})>"
        )
