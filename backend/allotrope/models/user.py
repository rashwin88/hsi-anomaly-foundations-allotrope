"""User entity (abstractions-spec § 5.1).

Field set matches the spec exactly, plus `is_admin` (added during Step 1
build for the admin/create-user flow — see abstractions.md addendum).
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class User(Base):
    __tablename__ = "users"

    # Identifier. uuid4 for now; switch to uuid7 when stdlib lands it
    # (CPython PEP for uuid7 is in flight; library shims exist but uuid4 is
    # fine for our scale).
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Natural keys. Both case-insensitive unique — the unique index is
    # functional (LOWER(col)), added in the Alembic migration in Step 1c.
    username: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False)

    # Argon2id-encoded password (algo + params + salt + hash in one string).
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional display name; falls back to `username` in the UI.
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Updated on each successful login.
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # v1 admin flag. The seeded user has is_admin=true; only admins can
    # create new users via the api.
    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="false",
    )

    # Timestamps. server_default=now() means Postgres sets them, not Python.
    # onupdate fires on ORM updates (password change, display_name change, etc.).
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
            f"<User(id={self.id!s}, username={self.username!r}, "
            f"is_admin={self.is_admin})>"
        )
