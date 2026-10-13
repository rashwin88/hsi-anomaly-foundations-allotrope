"""dataset_files part sha256

Revision: 20261013_03
Down revision: 20261013_02
Created: 2026-10-13

Adds `dataset_files.part_sha256` — the base64 SHA-256 of each PART_SIZE
part, as declared at upload start. Needed to re-sign part URLs on resume and
to recompute the composite checksum S3 stores on completion (S3 cannot give
a whole-file SHA-256 for multipart objects; see allotrope_inventory/s3.py).
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20261013_03"
down_revision: Union[str, None] = "20261013_02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "dataset_files",
        sa.Column(
            "part_sha256", postgresql.JSONB(), nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("dataset_files", "part_sha256")
