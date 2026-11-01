"""create worker_status table (GPU monitoring, class-driven flow addition E)

Revision: 20261101_10
Down revision: 20260511_09
Created: 2026-11-01
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261101_10"
down_revision: Union[str, None] = "20260511_09"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "worker_status",
        sa.Column("worker_id", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("gpu_devices", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("gpu_note", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("torch_device", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("worker_id"),
    )


def downgrade() -> None:
    op.drop_table("worker_status")
