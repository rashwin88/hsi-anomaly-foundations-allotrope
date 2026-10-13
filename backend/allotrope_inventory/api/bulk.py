"""Bulk file registration and retirement — the two operations a collection needs.

Bulk registration exists for collections like SpectralEarth's ~500k patches,
where a presigned URL per file would be absurd. The unpacker writes each file
straight under the dataset's own prefix as a *single* PUT carrying its
SHA-256 — for a single-part object S3 verifies the true whole-file hash, a
stronger guarantee than the multipart scheme — and then registers rows here
in batches. The api trusts nothing it is told: every row is checked with a
HEAD against the stored object's size and S3-computed SHA-256 before it is
inserted. Re-sending a batch is harmless (already-present paths are skipped).

Retirement deletes a dataset's payload (every object version) and keeps the
row, its file list, manifests and audit trail — the record that it existed,
what it held, and what was derived from it. Used for an archive once its
contents have been unpacked into a collection.
"""

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .. import s3
from ..auth import require_key
from ..db import get_db
from ..events import record
from ..models import DatasetFile
from .datasets import DatasetPublic, dataset_or_404, tags_by_dataset, write_manifest
from .uploads import check_relative_path, object_key, pending_or_409

router = APIRouter(prefix="/datasets", tags=["bulk"])

HEAD_WORKERS = 32
SINGLE_PUT_MAX = 5 * 1024**3          # S3's limit for one PUT; larger files use /uploads


# --- Schemas ----------------------------------------------------------


class BulkFile(BaseModel):
    relative_path: str = Field(min_length=1, max_length=512)
    size_bytes: int = Field(ge=0, le=SINGLE_PUT_MAX)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        return check_relative_path(v)


class BulkRequest(BaseModel):
    files: list[BulkFile] = Field(min_length=1, max_length=5000)


class BulkFailure(BaseModel):
    relative_path: str
    reason: str


class BulkResult(BaseModel):
    registered: int
    already_present: int
    failed: list[BulkFailure]     # not inserted; fix and re-send


class RetireRequest(BaseModel):
    confirm: str = Field(description="Must equal the dataset's sensor_scene_id.")
    reason: str = Field(min_length=1, max_length=500)


# --- Routes -----------------------------------------------------------


@router.post(
    "/{dataset_id}/files/bulk",
    response_model=BulkResult,
    summary="Register up to 5,000 already-stored single-PUT files, each verified by HEAD",
)
def bulk_register(
    dataset_id: str,
    body: BulkRequest,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> BulkResult:
    """409 `dataset_not_pending` unless the dataset is still open. The HEADs
    run before the dataset lock is taken, so a slow batch never blocks reads."""
    d = dataset_or_404(dataset_id, db)
    if d.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT, detail="dataset_not_pending")
    paths = [f.relative_path for f in body.files]
    present = set(db.scalars(select(DatasetFile.relative_path).where(
        DatasetFile.dataset_id == d.id, DatasetFile.relative_path.in_(paths)
    )))
    todo = [f for f in body.files if f.relative_path not in present]

    def check(f: BulkFile) -> tuple[BulkFile, str | None, str | None]:
        version, problem = s3.verify_single_put(object_key(d, f.relative_path), f.size_bytes, f.sha256)
        return f, version, problem

    with ThreadPoolExecutor(max_workers=HEAD_WORKERS) as pool:
        results = list(pool.map(check, todo))
    good = [(f, v) for f, v, problem in results if problem is None]
    failed = [BulkFailure(relative_path=f.relative_path, reason=problem)
              for f, _, problem in results if problem is not None]

    d = pending_or_409(dataset_id, db)
    inserted = 0
    if good:
        inserted = db.execute(insert(DatasetFile).values([
            {"dataset_id": d.id, "relative_path": f.relative_path, "size_bytes": f.size_bytes,
             "sha256": f.sha256, "part_sha256": [s3.hex_to_b64(f.sha256)],
             "upload_id": None, "s3_version_id": v}
            for f, v in good
        ]).on_conflict_do_nothing(constraint="dataset_files_path_uq")).rowcount
    record(db, "files.bulk_registered", actor, d.id,
           registered=inserted, already_present=len(present), failed=len(failed))
    db.commit()
    return BulkResult(registered=inserted, already_present=len(present) + len(good) - inserted,
                      failed=failed)


@router.post(
    "/{dataset_id}/retire",
    response_model=DatasetPublic,
    summary="Delete a dataset's payload, keep its record",
)
def retire(
    dataset_id: str,
    body: RetireRequest,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DatasetPublic:
    """Irreversible. 409 `dataset_not_available` unless available; 422
    `confirm_mismatch` unless `confirm` repeats the scene id."""
    d = dataset_or_404(dataset_id, db)
    db.refresh(d, with_for_update=True)
    if d.status != "available":
        raise HTTPException(status.HTTP_409_CONFLICT, detail="dataset_not_available")
    if body.confirm != d.sensor_scene_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="confirm_mismatch")
    objects, size = s3.delete_all_versions(f"{d.s3_prefix}/payload/")
    d.status = "retired"
    record(db, "dataset.retired", actor, d.id, reason=body.reason,
           objects_deleted=objects, bytes_deleted=size)
    db.flush()
    write_manifest(db, d)             # file list kept as the record of what it held
    db.commit()
    db.refresh(d)
    return DatasetPublic.from_orm_dataset(d, tags_by_dataset(db, [d.id])[d.id])
