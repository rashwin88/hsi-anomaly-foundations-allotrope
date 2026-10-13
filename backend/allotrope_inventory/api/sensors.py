"""Sensors — the instruments datasets can belong to.

Adding a sensor is a POST, not a code change. Slugs are the same strings the
main app uses in `scenes.sensor_type`, so keep new ones lowercase snake_case.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import require_key
from ..db import get_db
from ..events import record
from ..models import Sensor

router = APIRouter(prefix="/sensors", tags=["sensors"])


# --- Schemas ----------------------------------------------------------


class SensorCreate(BaseModel):
    slug: str = Field(pattern=r"^[a-z][a-z0-9_]{1,31}$", examples=["desis"])
    name: str = Field(min_length=1, max_length=100, examples=["DESIS"])
    family: str = Field(
        pattern=r"^[a-z][a-z0-9_]{1,31}$",
        description="hyperspectral | thermal | multispectral | … (free text, snake_case)",
    )
    description: str | None = Field(default=None, max_length=2000)


class SensorPublic(BaseModel):
    slug: str
    name: str
    family: str
    description: str | None
    created_at: datetime

    @classmethod
    def from_orm_sensor(cls, s: Sensor) -> "SensorPublic":
        return cls(
            slug=s.slug, name=s.name, family=s.family,
            description=s.description, created_at=s.created_at,
        )


# --- Routes -----------------------------------------------------------


@router.get("", response_model=list[SensorPublic], summary="List sensors")
def list_sensors(
    _actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> list[SensorPublic]:
    """All sensors, alphabetical. Small table — no pagination."""
    rows = db.scalars(select(Sensor).order_by(Sensor.slug)).all()
    return [SensorPublic.from_orm_sensor(s) for s in rows]


@router.post(
    "",
    response_model=SensorPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Add a sensor",
)
def create_sensor(
    body: SensorCreate,
    actor: str = Depends(require_key),
    db: Session = Depends(get_db),
) -> SensorPublic:
    """409 `sensor_exists` if the slug is taken."""
    if db.get(Sensor, body.slug) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="sensor_exists")
    s = Sensor(**body.model_dump())
    db.add(s)
    record(db, "sensor.created", actor, slug=s.slug)
    db.commit()
    db.refresh(s)
    return SensorPublic.from_orm_sensor(s)
