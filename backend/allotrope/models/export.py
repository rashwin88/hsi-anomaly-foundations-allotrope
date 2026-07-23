"""Export entity (abstractions-spec § 5.12).

A persisted snapshot of a Project's Result state, packaged into a
downloadable bundle on disk.

- Identity: id (uuid). Wire `export_<uuid>`.
- Ownership: Project.
- Lifecycle: async — created **only after** the project_export Job
  succeeds (Option B). The row's existence is the success signal.
  Fully immutable thereafter; no UPDATE path.
- Storage: allotrope_artifacts/projects/<project_id>/exports/<id>/<filename>
- Cascade: CASCADE on projects.id.
"""

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Export(Base):
    __tablename__ = "exports"

    __table_args__ = (Index("exports_project_id_idx", "project_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    bundle_path: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # "zip" in v1; format kept text-typed so we can add tar.gz later
    # without a schema migration.
    format: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Export(id={self.id!s}, project_id={self.project_id!s})>"
