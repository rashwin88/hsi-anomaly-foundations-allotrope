"""dataset kind

Revision: 20261013_04
Down revision: 20261013_03
Created: 2026-10-13

Adds `datasets.kind` so different shapes of holding never mix in a filter:

- `scene`      one acquisition as delivered (EnMAP/PRISMA/… product). Default.
- `archive`    a provider's packaged delivery kept as-is (SpectralEarth's
               9-part tar.gz) — usually retired once unpacked.
- `collection` many small items unpacked from an archive (SpectralEarth's
               ~500k 128x128 patches), each a dataset_files row.

Also relabels the SpectralEarth archive: it is not an L2A scene, so its
level becomes `SpectralEarth` (s3_prefix is stored, so the objects stay put).
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20261013_04"
down_revision: Union[str, None] = "20261013_03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("kind", sa.Text(), nullable=False, server_default=sa.text("'scene'")),
    )
    op.create_index("datasets_kind_idx", "datasets", ["kind"])
    op.execute(
        "UPDATE datasets SET kind = 'archive', processing_level = 'SpectralEarth' "
        "WHERE sensor_scene_id = 'SPECTRAL_EARTH'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE datasets SET processing_level = 'L2A' WHERE sensor_scene_id = 'SPECTRAL_EARTH'"
    )
    op.drop_index("datasets_kind_idx", table_name="datasets")
    op.drop_column("datasets", "kind")
