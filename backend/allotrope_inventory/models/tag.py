"""Tag + DatasetTag — free-form labels on datasets.

Tags live here, not as S3 object tags: S3 caps objects at 10 tags and cannot
query them. A tag is a plain name (`cloud_free`) or `key:value`
(`site:jharia`); the `tags` row exists so names can carry a description and
be listed without scanning every dataset.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Tag(Base):
    __tablename__ = "tags"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DatasetTag(Base):
    __tablename__ = "dataset_tags"

    # PK (dataset_id, tag) serves per-dataset lookups; this serves "by tag".
    __table_args__ = (Index("dataset_tags_tag_idx", "tag"),)

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True
    )
    tag: Mapped[str] = mapped_column(
        Text, ForeignKey("tags.name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
