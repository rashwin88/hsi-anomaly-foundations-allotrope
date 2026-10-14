"""Stats — the inventory as a whole, for the UI's summary tiles."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from ..auth import require_key
from ..db import get_db
from ..models import Dataset, DatasetFile, Tag

router = APIRouter(tags=["stats"])


# --- Schemas ----------------------------------------------------------


class StatsCell(BaseModel):
    sensor_type: str
    status: str
    datasets: int
    bytes: int


class InventoryStats(BaseModel):
    datasets: int
    bytes: int
    tags: int
    cells: list[StatsCell]        # one per (sensor_type, status) that has rows


# --- Routes -----------------------------------------------------------


@router.get("/stats", response_model=InventoryStats, summary="Counts and bytes by sensor x status")
def stats(
    _actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> InventoryStats:
    rows = db.execute(
        select(
            Dataset.sensor_type, Dataset.status,
            func.count(distinct(Dataset.id)),
            func.coalesce(func.sum(DatasetFile.size_bytes), 0),
        )
        .outerjoin(DatasetFile, DatasetFile.dataset_id == Dataset.id)
        .group_by(Dataset.sensor_type, Dataset.status)
        .order_by(Dataset.sensor_type, Dataset.status)
    ).all()
    cells = [StatsCell(sensor_type=s, status=st, datasets=n, bytes=b) for s, st, n, b in rows]
    return InventoryStats(
        datasets=sum(c.datasets for c in cells),
        bytes=sum(c.bytes for c in cells),
        tags=db.scalar(select(func.count()).select_from(Tag)) or 0,
        cells=cells,
    )
