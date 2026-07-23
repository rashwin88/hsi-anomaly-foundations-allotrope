"""create users table

Revision: 20260510_01
Down revision: None (initial)
Created: 2026-05-10

Creates the `users` table per abstractions-spec § 5.1, including the v1
`is_admin` flag and the case-insensitive functional unique indexes on
username and email.
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260510_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "is_admin",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Functional unique indexes for case-insensitive uniqueness.
    # SQLAlchemy's Column-level `unique=True` only does a plain unique
    # constraint; for LOWER(col) we have to build the index explicitly.
    op.create_index(
        "users_username_lower_uq",
        "users",
        [sa.text("lower(username)")],
        unique=True,
    )
    op.create_index(
        "users_email_lower_uq",
        "users",
        [sa.text("lower(email)")],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("users_email_lower_uq", table_name="users")
    op.drop_index("users_username_lower_uq", table_name="users")
    op.drop_table("users")
