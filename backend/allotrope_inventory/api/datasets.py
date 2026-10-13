"""Datasets — one scene delivery each.

Lifecycle: POST registers a `pending` row and fixes its S3 prefix; files
are then uploaded and the dataset finalised (later routes). Footprints go in
and out as GeoJSON geometries in EPSG:4326.
"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from geoalchemy2.shape import from_shape, to_shape
from pydantic import BaseModel, Field, field_validator
from shapely.geometry import mapping, shape
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import s3
from ..auth import require_key
from ..config import settings
from ..db import get_db
from ..events import record
from ..models import Dataset, DatasetFile, DatasetTag, Event, Provenance, Sensor
from ..wireformat import parse_prefixed_id, to_prefixed

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Origins a delivery can be fetched from; the scheme doubles as the kind.
SOURCE_SCHEMES = ("s3://", "gdrive://", "file://", "https://", "http://")


# --- Schemas ----------------------------------------------------------


class DatasetCreate(BaseModel):
    sensor_type: str = Field(examples=["prisma"])
    sensor_scene_id: str = Field(
        min_length=1, max_length=300,
        description="The provider's own product id, e.g. the delivery filename stem.",
    )
    # Becomes an S3 path segment, hence the tight pattern.
    processing_level: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$", examples=["L2D"])
    acquisition_at: datetime | None = None
    footprint: dict[str, Any] | None = Field(
        default=None, description="GeoJSON geometry, EPSG:4326 lon/lat."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(
        default_factory=list,
        description="Where this delivery was fetched from (s3://, gdrive://, file://, https://).",
    )

    @field_validator("footprint")
    @classmethod
    def _valid_geometry(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            try:
                geom = shape(v)
            except Exception as e:
                raise ValueError(f"not a GeoJSON geometry: {e}") from e
            if geom.is_empty or not geom.is_valid:
                raise ValueError("footprint geometry is empty or invalid")
        return v

    @field_validator("sources")
    @classmethod
    def _known_schemes(cls, v: list[str]) -> list[str]:
        for uri in v:
            if not uri.startswith(SOURCE_SCHEMES):
                raise ValueError(f"unsupported source scheme: {uri}")
        return v


class DatasetPublic(BaseModel):
    id: str                       # ds_<uuid>
    sensor_type: str
    sensor_scene_id: str
    processing_level: str
    status: str                   # pending | available | failed
    acquisition_at: datetime | None
    footprint: dict[str, Any] | None
    s3_uri: str                   # s3://<bucket>/<prefix>/
    metadata: dict[str, Any]
    tags: list[str]
    created_at: datetime
    created_by: str
    finalized_at: datetime | None

    @classmethod
    def from_orm_dataset(cls, d: Dataset, tags: list[str]) -> "DatasetPublic":
        return cls(
            id=to_prefixed("ds", d.id),
            sensor_type=d.sensor_type,
            sensor_scene_id=d.sensor_scene_id,
            processing_level=d.processing_level,
            status=d.status,
            acquisition_at=d.acquisition_at,
            footprint=mapping(to_shape(d.footprint)) if d.footprint is not None else None,
            s3_uri=f"s3://{settings.bucket}/{d.s3_prefix}/",
            metadata=d.extra_metadata,
            tags=tags,
            created_at=d.created_at,
            created_by=d.created_by,
            finalized_at=d.finalized_at,
        )


class DatasetFilePublic(BaseModel):
    relative_path: str            # under payload/
    size_bytes: int
    sha256: str
    uploaded: bool                # False while its multipart upload is open
    s3_version_id: str | None


class ProvenancePublic(BaseModel):
    source_uri: str
    relative_path: str | None     # None when the origin covers the whole delivery
    note: str | None
    fetched_by: str
    fetched_at: datetime


class DatasetDetail(DatasetPublic):
    """DatasetPublic plus its files and origins (GET /datasets/{id} only)."""

    files: list[DatasetFilePublic]
    total_bytes: int
    sources: list[ProvenancePublic]


class DatasetsPage(BaseModel):
    """Paginated list of datasets."""

    items: list[DatasetPublic]
    total: int          # rows matching the filter (excluding limit/offset)
    limit: int
    offset: int


# --- Helpers ----------------------------------------------------------


def dataset_or_404(dataset_id: str, db: Session) -> Dataset:
    """Resolve a `ds_<uuid>` wire id; 404 `dataset_not_found` otherwise."""
    d = db.get(Dataset, parse_prefixed_id("ds", dataset_id))
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="dataset_not_found")
    return d


def tags_by_dataset(db: Session, ids: list[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
    """One query for the tags of many datasets (avoids N+1 on list pages)."""
    out: dict[uuid.UUID, list[str]] = {i: [] for i in ids}
    rows = db.execute(
        select(DatasetTag.dataset_id, DatasetTag.tag)
        .where(DatasetTag.dataset_id.in_(ids)).order_by(DatasetTag.tag)
    )
    for dataset_id, tag in rows:
        out[dataset_id].append(tag)
    return out


def dataset_detail(db: Session, d: Dataset) -> DatasetDetail:
    """The full read view of one dataset: row, tags, files, origins."""
    files = db.scalars(
        select(DatasetFile).where(DatasetFile.dataset_id == d.id).order_by(DatasetFile.relative_path)
    ).all()
    path_of = {f.id: f.relative_path for f in files}
    origins = db.scalars(
        select(Provenance).where(Provenance.dataset_id == d.id).order_by(Provenance.fetched_at)
    ).all()
    public = DatasetPublic.from_orm_dataset(d, tags_by_dataset(db, [d.id])[d.id])
    return DatasetDetail(
        **public.model_dump(),
        files=[
            DatasetFilePublic(
                relative_path=f.relative_path, size_bytes=f.size_bytes, sha256=f.sha256,
                uploaded=f.upload_id is None, s3_version_id=f.s3_version_id,
            )
            for f in files
        ],
        total_bytes=sum(f.size_bytes for f in files),
        sources=[
            ProvenancePublic(
                source_uri=o.source_uri, relative_path=path_of.get(o.dataset_file_id),
                note=o.note, fetched_by=o.fetched_by, fetched_at=o.fetched_at,
            )
            for o in origins
        ],
    )


def write_manifest(db: Session, d: Dataset) -> None:
    """Rewrite `<prefix>/manifest.json` from the current rows.

    Called on finalise and on every tag change of an available dataset, so
    the bucket alone is enough to rebuild this database. Part hashes ride
    along so a rebuilt index can still verify the stored objects.
    """
    body = dataset_detail(db, d).model_dump(mode="json")
    part_hashes = dict(db.execute(
        select(DatasetFile.relative_path, DatasetFile.part_sha256)
        .where(DatasetFile.dataset_id == d.id)
    ).all())
    for f in body["files"]:
        f["part_sha256"] = part_hashes[f["relative_path"]]
    body["part_size"] = s3.PART_SIZE
    s3.put_manifest(d.s3_prefix, body)


# --- Routes -----------------------------------------------------------


@router.post(
    "",
    response_model=DatasetPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Register a dataset (status=pending)",
)
def register_dataset(
    body: DatasetCreate,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DatasetPublic:
    """Create the row and fix its S3 prefix; nothing is uploaded yet.

    422 `unknown_sensor` if the sensor isn't registered; 409 `dataset_exists`
    if this (sensor, scene id, level) is already catalogued.
    """
    if db.get(Sensor, body.sensor_type) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="unknown_sensor")

    dataset_id = uuid.uuid4()
    d = Dataset(
        id=dataset_id,
        sensor_type=body.sensor_type,
        sensor_scene_id=body.sensor_scene_id,
        processing_level=body.processing_level,
        acquisition_at=body.acquisition_at,
        footprint=from_shape(shape(body.footprint), srid=4326) if body.footprint else None,
        s3_prefix=f"{body.sensor_type}/{body.processing_level}/{to_prefixed('ds', dataset_id)}",
        extra_metadata=body.metadata,
        created_by=actor,
    )
    db.add(d)
    for uri in body.sources:
        db.add(Provenance(dataset_id=dataset_id, source_uri=uri, fetched_by=actor))
    record(db, "dataset.registered", actor, dataset_id, sources=body.sources)
    try:
        db.commit()
    except IntegrityError:
        # The unique (sensor, scene id, level) constraint is the only one a
        # client can trip here; checking first would still race.
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail="dataset_exists")
    db.refresh(d)
    return DatasetPublic.from_orm_dataset(d, tags=[])


@router.get(
    "",
    response_model=DatasetsPage,
    status_code=status.HTTP_200_OK,
    summary="List datasets (filtered, paginated)",
)
def list_datasets(
    limit: int = Query(50, ge=1, le=500, description="Max items per page (1-500)"),
    offset: int = Query(0, ge=0, description="Offset into the result set"),
    sensor_type: str | None = Query(None, description="e.g. prisma"),
    processing_level: str | None = Query(None, description="e.g. L2D"),
    status_: str | None = Query(None, alias="status", description="pending | available | failed"),
    tag: list[str] = Query([], description="Repeatable; a dataset must carry every tag given"),
    q: str | None = Query(None, max_length=200, description="Substring of sensor_scene_id"),
    bbox: str | None = Query(
        None, description="min_lon,min_lat,max_lon,max_lat — footprint must intersect it"
    ),
    acquired_from: datetime | None = Query(None, description="acquisition_at >= this"),
    acquired_to: datetime | None = Query(None, description="acquisition_at < this"),
    _actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DatasetsPage:
    """Newest-registered first. Datasets without a footprint never match a
    bbox filter, nor without acquisition_at a time filter."""
    base = select(Dataset)
    if sensor_type is not None:
        base = base.where(Dataset.sensor_type == sensor_type)
    if processing_level is not None:
        base = base.where(Dataset.processing_level == processing_level)
    if status_ is not None:
        base = base.where(Dataset.status == status_)
    for t in tag:
        base = base.where(
            select(DatasetTag.tag)
            .where(DatasetTag.dataset_id == Dataset.id, DatasetTag.tag == t)
            .exists()
        )
    if q is not None and q.strip():
        # Escape LIKE wildcards rather than strip them: product ids are full
        # of underscores (PRS_L2D_STD_…), and stripping would never match.
        needle = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        base = base.where(Dataset.sensor_scene_id.ilike(f"%{needle}%", escape="\\"))
    if bbox is not None:
        try:
            min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox.split(","))
        except ValueError:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="bad_bbox")
        envelope = func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
        base = base.where(func.ST_Intersects(Dataset.footprint, envelope))
    if acquired_from is not None:
        base = base.where(Dataset.acquisition_at >= acquired_from)
    if acquired_to is not None:
        base = base.where(Dataset.acquisition_at < acquired_to)

    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = db.scalars(
        base.order_by(Dataset.created_at.desc()).limit(limit).offset(offset)
    ).all()
    tags = tags_by_dataset(db, [d.id for d in rows])

    return DatasetsPage(
        items=[DatasetPublic.from_orm_dataset(d, tags[d.id]) for d in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetDetail,
    status_code=status.HTTP_200_OK,
    summary="Get one dataset with its files and origins",
)
def get_dataset(
    dataset_id: str,
    _actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> DatasetDetail:
    return dataset_detail(db, dataset_or_404(dataset_id, db))


class EventPublic(BaseModel):
    kind: str                     # dataset.registered, file.uploaded, tag.added, …
    actor: str
    payload: dict[str, Any]
    created_at: datetime


@router.get(
    "/{dataset_id}/events",
    response_model=list[EventPublic],
    status_code=status.HTTP_200_OK,
    summary="Audit trail of one dataset, oldest first",
)
def list_events(
    dataset_id: str,
    _actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> list[EventPublic]:
    d = dataset_or_404(dataset_id, db)
    rows = db.scalars(
        select(Event).where(Event.dataset_id == d.id).order_by(Event.created_at).limit(1000)
    ).all()
    return [
        EventPublic(kind=e.kind, actor=e.actor, payload=e.payload, created_at=e.created_at)
        for e in rows
    ]
