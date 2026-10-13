"""Inventory service — FastAPI entry point.

Routers live under `allotrope_inventory/api/` and are wired up here. The
probes stay unauthenticated so container health checks need no key.
"""

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from .api.datasets import router as datasets_router
from .api.sensors import router as sensors_router
from .api.stats import router as stats_router
from .api.tags import router as tags_router
from .api.uploads import router as uploads_router
from .db import get_db

app = FastAPI(
    title="Allotrope inventory",
    version="0.0.1",
)

app.include_router(sensors_router)
app.include_router(datasets_router)
app.include_router(tags_router)
app.include_router(uploads_router)
app.include_router(stats_router)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness — does the process answer?"""
    return {"status": "ok"}


@app.get("/healthz/db")
def healthz_db(db: Session = Depends(get_db)) -> dict[str, str]:
    """Readiness — is the DB reachable?"""
    db.execute(text("SELECT 1"))
    return {"status": "ok", "db": "connected"}
