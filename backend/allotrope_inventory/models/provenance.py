"""Provenance — where a dataset (or one of its files) was fetched from.

Raw data arrives from many places (Beta S3, Google Drive, local disks, URLs),
often more than one per delivery. One row per origin; `dataset_file_id` is
NULL when the origin covers the whole delivery.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Provenance(Base):
    __tablename__ = "provenance"

    __table_args__ = (Index("provenance_dataset_id_idx", "dataset_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_files.id", ondelete="CASCADE"), nullable=True
    )

    # Full origin URI: `s3://…`, `gdrive://<file id>`, `file:///…`, `https://…`.
    # The scheme doubles as the source kind; validated at the api layer.
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    fetched_by: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
