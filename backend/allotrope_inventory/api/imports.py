"""Imports — server-side copy from another S3 bucket into the inventory.

For data already in S3 (the SpectralEarth bucket, other prod buckets), the
bytes never need to leave AWS. The client still reads each object once to
compute its hashes — that is what makes the record trustworthy — then
declares the file here, and the api copies it with UploadPartCopy, checking
the SHA-256 S3 computes for every copied part against the declaration.
Finalise is the same call as for uploads.

One file per request: a 300 GiB object is ~4,800 parts and takes minutes,
so files are imported concurrently by issuing parallel requests. The
dataset lock is held only while declaring, never during the copy.
Re-sending a request resumes: parts S3 already holds are skipped.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from .. import s3
from ..auth import require_key
from ..db import get_db
from ..events import record
from ..models import DatasetFile
from .uploads import FileDeclaration, declare_file, object_key, pending_or_409

router = APIRouter(prefix="/datasets", tags=["imports"])


# --- Schemas ----------------------------------------------------------


class ImportFile(FileDeclaration):
    source_uri: str = Field(description="s3://bucket/key the api will copy from.")
    note: str | None = Field(
        default=None, max_length=500,
        description="Recorded on the provenance row, e.g. 'md5 matches provider checksum'.",
    )

    @field_validator("source_uri")
    @classmethod
    def _s3_only(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("imports copy from s3:// URIs only")
        s3.parse_s3_uri(v)            # raises on a bucket-only URI
        return v


class ImportResult(BaseModel):
    relative_path: str
    parts_copied: int
    parts_already_held: int


# --- Routes -----------------------------------------------------------


@router.post(
    "/{dataset_id}/imports",
    response_model=ImportResult,
    summary="Copy one file in from another S3 bucket, server-side (resumable)",
)
def import_file(
    dataset_id: str,
    body: ImportFile,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> ImportResult:
    """422 `source_size_mismatch` if the source isn't the declared size; 409
    `checksum_mismatch` if S3's hash of any copied part differs from the
    declaration (the upload is discarded); plus the upload errors."""
    src_bucket, src_key = s3.parse_s3_uri(body.source_uri)
    if s3.source_size(src_bucket, src_key) != body.size_bytes:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="source_size_mismatch")

    d = pending_or_409(dataset_id, db)
    row = declare_file(db, d, body, actor, note=body.note)
    key, upload_id, dataset_uuid, row_id = object_key(d, body.relative_path), row.upload_id, d.id, row.id
    db.commit()                       # release the dataset lock before the long copy

    held = set(s3.uploaded_parts(key, upload_id))
    bad = s3.copy_parts(
        key, upload_id, src_bucket, src_key, body.size_bytes, body.part_sha256, skip=held,
    )
    if bad:
        s3.abort_upload(key, upload_id)
        row = db.get(DatasetFile, row_id)
        if row is not None:
            db.delete(row)
        record(db, "file.rejected", actor, dataset_uuid, path=body.relative_path,
               reason=f"copied part hash mismatch: parts {bad[:20]}")
        db.commit()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={"code": "checksum_mismatch", "paths": [body.relative_path], "parts": bad[:20]},
        )
    copied = len(body.part_sha256) - len(held)
    record(db, "file.copied", actor, dataset_uuid,
           path=body.relative_path, source=body.source_uri, parts=copied)
    db.commit()
    return ImportResult(
        relative_path=body.relative_path, parts_copied=copied, parts_already_held=len(held),
    )
