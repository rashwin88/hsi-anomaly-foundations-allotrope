"""lock out supabase data api

Revision: 20261013_02
Down revision: 20261013_01
Created: 2026-10-13

Supabase publishes every `public` table through its auto-generated REST API
to the `anon` and `authenticated` roles, and the anon key is public by
design. This service never uses that API — the inventory api is the only
door — so RLS is switched on with no policies: those roles see nothing,
while the table owner (the role the inventory api connects as) bypasses
RLS and is unaffected. Harmless on a plain Postgres.
"""
from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20261013_02"
down_revision: Union[str, None] = "20261013_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = (
    "sensors", "tags", "datasets", "dataset_files", "dataset_tags", "provenance", "events",
)


def upgrade() -> None:
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
