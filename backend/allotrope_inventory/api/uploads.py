"""Uploads and downloads — moving a delivery's bytes in and out of S3.

1. POST /datasets/{id}/uploads declares files (size, whole-file sha256 hex,
   per-part sha256 base64 on PART_SIZE boundaries) and returns one presigned
   PUT per part still missing. Each PUT must send the returned
   `x-amz-checksum-sha256` header; S3 rejects a body that doesn't match.
2. Calling it again with the same declaration resumes: same upload, fresh
   URLs, only for parts S3 doesn't hold yet. A changed declaration for a
   file still uploading discards the old upload and starts over.
3. POST /datasets/{id}/finalize completes every open upload, verifies size
   and composite checksum, writes manifest.json, and marks the dataset
   `available`. Files that
   verify stay done even if others don't, so a retry only redoes the rest.
4. GET /datasets/{id}/download hands out version-pinned presigned GETs.
"""

import base64
import binascii
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import s3
from ..auth import require_key
from ..db import get_db
from ..events import record
from ..models import Dataset, DatasetFile, Provenance
from ..wireformat import to_prefixed
from .datasets import (
    SOURCE_SCHEMES, DatasetPublic, dataset_or_404, tags_by_dataset, write_manifest,
)

router = APIRouter(prefix="/datasets", tags=["uploads"])


# --- Schemas ----------------------------------------------------------


def check_relative_path(v: str) -> str:
    """A clean forward-slash path that cannot escape payload/ (shared with bulk)."""
    segments = v.split("/")
    if "\\" in v or any(s in ("", ".", "..") for s in segments) or not v.isprintable():
        raise ValueError("relative_path must be a clean forward-slash path inside payload/")
    return v


class FileDeclaration(BaseModel):
    relative_path: str = Field(
        min_length=1, max_length=512, description="Path under payload/, forward slashes."
    )
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="Whole-file digest, hex.")
    part_sha256: list[str] = Field(
        description="Base64 SHA-256 of each PART_SIZE part, in order."
    )
    source_uri: str | None = Field(default=None, description="Where this file was fetched from.")

    @field_validator("relative_path")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        return check_relative_path(v)

    @field_validator("source_uri")
    @classmethod
    def _known_scheme(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(SOURCE_SCHEMES):
            raise ValueError(f"unsupported source scheme: {v}")
        return v

    @model_validator(mode="after")
    def _parts_match_size(self) -> "FileDeclaration":
        if len(self.part_sha256) != s3.part_count(self.size_bytes):
            raise ValueError(
                f"{self.relative_path}: expected {s3.part_count(self.size_bytes)} part "
                f"hashes for {self.size_bytes} B at part_size {s3.PART_SIZE}"
            )
        for d in self.part_sha256:
            try:
                if len(base64.b64decode(d, validate=True)) != 32:
                    raise ValueError
            except (binascii.Error, ValueError):
                raise ValueError(f"{self.relative_path}: part hash is not base64 SHA-256")
        return self


class UploadRequest(BaseModel):
    files: list[FileDeclaration] = Field(min_length=1, max_length=1000)


class PartUpload(BaseModel):
    part_number: int
    url: str
    headers: dict[str, str]       # must be sent verbatim with the PUT


class FileUpload(BaseModel):
    relative_path: str
    parts: list[PartUpload]       # only the parts S3 doesn't hold yet
    uploaded_parts: list[int]
    duplicate_of: list[str]       # ds_<uuid> of available datasets holding identical bytes


class UploadPlan(BaseModel):
    dataset_id: str
    part_size: int
    files: list[FileUpload]


# --- Helpers ----------------------------------------------------------


def pending_or_409(dataset_id: str, db: Session) -> Dataset:
    """Row-locked so concurrent uploads/finalize on one dataset serialise."""
    d = dataset_or_404(dataset_id, db)
    db.refresh(d, with_for_update=True)
    if d.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT, detail="dataset_not_pending")
    return d


def object_key(d: Dataset, relative_path: str) -> str:
    return f"{d.s3_prefix}/payload/{relative_path}"


def _duplicates(db: Session, sha256: str, own: uuid.UUID) -> list[str]:
    rows = db.scalars(
        select(Dataset.id).join(DatasetFile, DatasetFile.dataset_id == Dataset.id)
        .where(DatasetFile.sha256 == sha256, Dataset.status == "available", Dataset.id != own)
        .distinct()
    )
    return [to_prefixed("ds", i) for i in rows]


def declare_file(
    db: Session, d: Dataset, decl: FileDeclaration, actor: str, note: str | None = None,
) -> DatasetFile:
    """The file's row with an open multipart upload, ready for parts.

    Same declaration as before → the existing row and upload (resume).
    Different declaration for a file still uploading → old upload aborted,
    fresh one opened. Already verified → 409 `file_exists`. Shared by
    presigned uploads and server-side imports so both follow one rule.
    """
    key = object_key(d, decl.relative_path)
    row = db.scalar(select(DatasetFile).where(
        DatasetFile.dataset_id == d.id, DatasetFile.relative_path == decl.relative_path
    ))
    if row is not None and row.upload_id is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="file_exists")
    same = row is not None and (row.size_bytes, row.sha256, row.part_sha256) == (
        decl.size_bytes, decl.sha256, decl.part_sha256
    )
    if row is not None and not same:
        s3.abort_upload(key, row.upload_id)
        db.delete(row)
        db.flush()
        row = None
    if row is None:
        row = DatasetFile(
            dataset_id=d.id, relative_path=decl.relative_path, size_bytes=decl.size_bytes,
            sha256=decl.sha256, part_sha256=decl.part_sha256, upload_id=s3.start_upload(key),
        )
        db.add(row)
        db.flush()
        if decl.source_uri:
            db.add(Provenance(
                dataset_id=d.id, dataset_file_id=row.id,
                source_uri=decl.source_uri, note=note, fetched_by=actor,
            ))
        record(db, "file.declared", actor, d.id,
               path=decl.relative_path, size_bytes=decl.size_bytes, sha256=decl.sha256)
    return row


# --- Routes -----------------------------------------------------------


@router.post(
    "/{dataset_id}/uploads",
    response_model=UploadPlan,
    summary="Declare files and get presigned part URLs (resumable)",
)
def start_uploads(
    dataset_id: str,
    body: UploadRequest,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> UploadPlan:
    """409 `dataset_not_pending` once finalised; 409 `file_exists` for a path
    already uploaded and verified."""
    d = pending_or_409(dataset_id, db)
    paths = [f.relative_path for f in body.files]
    if len(set(paths)) != len(paths):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="duplicate_paths")

    out: list[FileUpload] = []
    for decl in body.files:
        key = object_key(d, decl.relative_path)
        row = declare_file(db, d, decl, actor)
        held = s3.uploaded_parts(key, row.upload_id)
        urls = s3.part_urls(key, row.upload_id, decl.part_sha256)
        out.append(FileUpload(
            relative_path=decl.relative_path,
            parts=[
                PartUpload(part_number=n, url=u, headers={"x-amz-checksum-sha256": h})
                for n, (u, h) in enumerate(zip(urls, decl.part_sha256), start=1)
                if n not in held
            ],
            uploaded_parts=sorted(held),
            duplicate_of=_duplicates(db, decl.sha256, d.id),
        ))
    db.commit()
    return UploadPlan(dataset_id=dataset_id, part_size=s3.PART_SIZE, files=out)


@router.post(
    "/{dataset_id}/finalize",
    response_model=DatasetPublic,
    summary="Complete and verify every upload; mark the dataset available",
)
def finalize(
    dataset_id: str,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DatasetPublic:
    """409 `no_files` with nothing declared. 409 `upload_incomplete` (with
    `missing`: path → part numbers) if S3 lacks parts. 409 `checksum_mismatch`
    (with `paths`) if stored bytes disagree with the declaration — those files
    are removed and must be declared and uploaded again."""
    d = pending_or_409(dataset_id, db)
    files = db.scalars(select(DatasetFile).where(DatasetFile.dataset_id == d.id)).all()
    if not files:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="no_files")

    missing: dict[str, list[int]] = {}
    mismatched: list[str] = []
    for f in files:
        if f.upload_id is None:
            continue
        try:
            f.s3_version_id = s3.complete_upload(
                object_key(d, f.relative_path), f.upload_id, f.part_sha256, f.size_bytes
            )
            f.upload_id = None
            record(db, "file.uploaded", actor, d.id, path=f.relative_path)
        except s3.UploadIncomplete as e:
            missing[f.relative_path] = e.missing
        except ValueError as e:
            mismatched.append(f.relative_path)
            record(db, "file.rejected", actor, d.id, path=f.relative_path, reason=str(e))
            db.delete(f)
        # Commit per file: a verified file stays verified whatever follows.
        db.commit()
        db.refresh(d, with_for_update=True)

    if mismatched:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail={"code": "checksum_mismatch", "paths": mismatched}
        )
    if missing:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail={"code": "upload_incomplete", "missing": missing}
        )

    d.status = "available"
    d.finalized_at = datetime.now(UTC)
    record(db, "dataset.finalized", actor, d.id,
           files=len(files), total_bytes=sum(f.size_bytes for f in files))
    db.flush()
    # Manifest before commit: if S3 refuses it, the dataset stays pending
    # (files remain verified) and finalize can simply be retried.
    write_manifest(db, d)
    db.commit()
    db.refresh(d)
    return DatasetPublic.from_orm_dataset(d, tags_by_dataset(db, [d.id])[d.id])


class FileDownload(BaseModel):
    relative_path: str
    size_bytes: int
    sha256: str
    url: str


class DownloadPlan(BaseModel):
    dataset_id: str
    expires_in: int               # seconds; call again for fresh URLs
    manifest_url: str
    files: list[FileDownload]


@router.get(
    "/{dataset_id}/download",
    response_model=DownloadPlan,
    summary="Presigned GET URLs for every file of an available dataset",
)
def download(
    dataset_id: str,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DownloadPlan:
    """409 `dataset_not_available` unless finalised. Logged, since handing
    out URLs is how data leaves the bucket."""
    d = dataset_or_404(dataset_id, db)
    if d.status != "available":
        raise HTTPException(status.HTTP_409_CONFLICT, detail="dataset_not_available")
    files = db.scalars(
        select(DatasetFile).where(DatasetFile.dataset_id == d.id).order_by(DatasetFile.relative_path)
    ).all()
    record(db, "dataset.downloaded", actor, d.id)
    db.commit()
    return DownloadPlan(
        dataset_id=dataset_id,
        expires_in=s3.URL_TTL_SECONDS,
        manifest_url=s3.download_url(f"{d.s3_prefix}/manifest.json", None),
        files=[
            FileDownload(
                relative_path=f.relative_path, size_bytes=f.size_bytes, sha256=f.sha256,
                url=s3.download_url(object_key(d, f.relative_path), f.s3_version_id),
            )
            for f in files
        ],
    )
