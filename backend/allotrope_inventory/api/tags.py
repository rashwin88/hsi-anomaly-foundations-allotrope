"""Tags — free-form labels on datasets.

A tag is `name` or `key:value` (`cloud_free`, `site:jharia`, `gt:set-2`).
PUT/DELETE are idempotent: tagging twice or untagging an absent tag both
succeed, and both return the dataset's full tag list. A tag's `tags` row is
created on first use and kept after its last dataset is untagged. Tag
changes on an available dataset rewrite its manifest.json.
"""

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..auth import require_key
from ..db import get_db
from ..events import record
from ..models import Dataset, DatasetTag, Tag
from .datasets import dataset_or_404, tags_by_dataset, write_manifest

router = APIRouter(tags=["tags"])

TAG_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,47}(:[A-Za-z0-9][A-Za-z0-9_.-]{0,63})?$"


# --- Schemas ----------------------------------------------------------


class TagPublic(BaseModel):
    name: str
    description: str | None
    dataset_count: int


class DatasetTags(BaseModel):
    dataset_id: str               # ds_<uuid>
    tags: list[str]


# --- Helpers ----------------------------------------------------------


def _sync_manifest(db: Session, d: Dataset) -> None:
    """Available datasets carry their tags in manifest.json; rewrite it before
    the commit so an S3 failure rolls the tag change back."""
    if d.status == "available":
        db.flush()
        write_manifest(db, d, with_files=False)


# --- Routes -----------------------------------------------------------


@router.get("/tags", response_model=list[TagPublic], summary="List tags with usage counts")
def list_tags(
    _actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> list[TagPublic]:
    rows = db.execute(
        select(Tag.name, Tag.description, func.count(DatasetTag.dataset_id))
        .outerjoin(DatasetTag, DatasetTag.tag == Tag.name)
        .group_by(Tag.name).order_by(Tag.name)
    )
    return [TagPublic(name=n, description=d, dataset_count=c) for n, d, c in rows]


@router.put(
    "/datasets/{dataset_id}/tags/{tag}",
    response_model=DatasetTags,
    summary="Tag a dataset (idempotent)",
)
def add_tag(
    dataset_id: str,
    tag: str = Path(pattern=TAG_PATTERN),
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DatasetTags:
    d = dataset_or_404(dataset_id, db)
    # ON CONFLICT DO NOTHING keeps both inserts race-free and idempotent.
    db.execute(insert(Tag).values(name=tag).on_conflict_do_nothing())
    added = db.execute(
        insert(DatasetTag).values(dataset_id=d.id, tag=tag, created_by=actor)
        .on_conflict_do_nothing()
    ).rowcount
    if added:
        record(db, "tag.added", actor, d.id, tag=tag)
        _sync_manifest(db, d)
    db.commit()
    return DatasetTags(dataset_id=dataset_id, tags=tags_by_dataset(db, [d.id])[d.id])


@router.delete(
    "/datasets/{dataset_id}/tags/{tag}",
    response_model=DatasetTags,
    summary="Untag a dataset (idempotent)",
)
def remove_tag(
    dataset_id: str,
    tag: str,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DatasetTags:
    d = dataset_or_404(dataset_id, db)
    link = db.get(DatasetTag, (d.id, tag))
    if link is not None:
        db.delete(link)
        record(db, "tag.removed", actor, d.id, tag=tag)
        _sync_manifest(db, d)
        db.commit()
    return DatasetTags(dataset_id=dataset_id, tags=tags_by_dataset(db, [d.id])[d.id])
