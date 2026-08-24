"""Allotrope api — FastAPI entry point.

Routes are split into routers under `allotrope/api/`. This module wires
them up and defines liveness / readiness probes.
"""

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from .api.action_threshold import anomaly_threshold_router
from .api.actions import (
    action_outputs_router,
    action_types_router,
    actions_router,
    project_actions_router,
)
from .api.admin import router as admin_router
from .api.annotations import (
    catalog_router as annotation_types_router,
    router as annotations_router,
)
from .api.auth import router as auth_router
from .api.jobs import router as jobs_router
from .api.models import router as models_router
from .api.action_templates import router as action_templates_router
from .api.metrics import router as metrics_router
from .api.result_and_exports import (
    exports_router as exports_detail_router,
    project_exports_router,
    project_result_router,
)
from .api.notes import (
    notes_router as note_detail_router,
    project_notes_router,
)
from .api.project_visualizations import (
    project_visualizations_router,
    visualizations_router as project_viz_detail_router,
)
from .api.projects import router as projects_router
from .api.scenes import router as scenes_router
from .api.visualizations import router as visualizations_router
from .db import get_db

app = FastAPI(
    title="Allotrope api",
    version="0.0.1",
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(scenes_router)
app.include_router(visualizations_router)
app.include_router(annotations_router)
app.include_router(annotation_types_router)
app.include_router(projects_router)
app.include_router(jobs_router)
app.include_router(models_router)
app.include_router(project_actions_router)
app.include_router(actions_router)
# Same /actions prefix as actions_router; the threshold flow lives in its own
# module because its three endpoints share a cache and must not be separated.
app.include_router(anomaly_threshold_router)
app.include_router(action_outputs_router)
app.include_router(action_types_router)
app.include_router(project_visualizations_router)
app.include_router(project_viz_detail_router)
app.include_router(project_notes_router)
app.include_router(note_detail_router)
app.include_router(action_templates_router)
app.include_router(metrics_router)
app.include_router(project_result_router)
app.include_router(project_exports_router)
app.include_router(exports_detail_router)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness — does the process answer?"""
    return {"status": "ok"}


@app.get("/healthz/db")
def healthz_db(db: Session = Depends(get_db)) -> dict[str, str]:
    """Readiness — is the DB reachable?"""
    db.execute(text("SELECT 1"))
    return {"status": "ok", "db": "connected"}
