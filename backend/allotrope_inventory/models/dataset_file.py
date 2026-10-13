"""DatasetFile â€” one original file inside a dataset's `payload/`.

- `relative_path` is the provider's own name, relative to `payload/`, so a
  delivery's internal folder structure survives untouched.
- `sha256` is the client-declared hex digest. S3 verifies it during upload
  (x-amz-checksum-sha256), so a stored file never disagrees with its row.
  Indexed so a re-upload of the same bytes under another dataset is caught.
- `part_sha256` lists the base64 SHA-256 of each PART_SIZE part, in order —
  re-signed into URLs on resume, and checked against S3's composite
  checksum on completion.
- `upload_id` holds the open multipart upload; cleared on finalise.
"""

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DatasetFile(Base):
    __tablename__ = "dataset_files"

    __table_args__ = (
        UniqueConstraint("dataset_id", "relative_path", name="dataset_files_path_uq"),
        Index("dataset_files_sha256_idx", "sha256"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )

    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    part_sha256: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    # Set while a multipart upload is open; NULL once completed.
    upload_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # S3 object version written on completion (bucket is versioned).
    s3_version_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
