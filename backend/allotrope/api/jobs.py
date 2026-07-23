"""Jobs endpoints.

Currently:
    GET /jobs/{job_id}   read a single job (for client-side status polling)

The Jobs *list* endpoint (`GET /jobs?status=…`) lands with the Jobs
destination in Step 19. Until then, the frontend polls a single job by id
after kicking off a workflow.

Sequence diagram: final design/diagrams/scene-onboard.drawio (Step 7e —
this endpoint shows up as the polling lane after the worker handler runs).
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..db import get_db
from ..models import Job
from .deps import current_user_claims
from .wireformat import parse_prefixed_id, to_prefixed

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobPublic(BaseModel):
    """Wire shape for a Job row.

    `payload` is exposed because the frontend may want to render input
    metadata while a job is in flight (e.g. "onboarding PRS_L2D_FOO.he5").
    `claimed_by` is intentionally NOT exposed — internal attribution only.
    """

    id: str                              # job_<uuid>
    type: str
    status: str                          # queued | running | complete | failed | cancelled
    project_id: str | None               # project_<uuid> when set
    payload: dict[str, Any]
    target_kind: str | None
    target_id: str | None                # `<target_kind>_<uuid>` when set
    failure_reason: str | None
    cancellation_requested: bool
    started_at: datetime | None
    last_heartbeat_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    @classmethod
    def from_orm_job(cls, j: Job) -> "JobPublic":
        # target_id is polymorphic — wire prefix matches target_kind so the
        # client can route back to the correct entity (Scene Detail, etc.).
        target_id_wire: str | None = None
        if j.target_id is not None and j.target_kind is not None:
            target_id_wire = f"{j.target_kind}_{j.target_id}"
        return cls(
            id=f"job_{j.id}",
            type=j.type,
            status=j.status,
            project_id=to_prefixed("project", j.project_id),
            payload=j.payload or {},
            target_kind=j.target_kind,
            target_id=target_id_wire,
            failure_reason=j.failure_reason,
            cancellation_requested=j.cancellation_requested,
            started_at=j.started_at,
            last_heartbeat_at=j.last_heartbeat_at,
            completed_at=j.completed_at,
            created_at=j.created_at,
        )


class JobsPage(BaseModel):
    items: list[JobPublic]
    total: int
    limit: int
    offset: int


# Status / type values are validated client-side via the picker; on the
# wire we accept anything and let the result set just come back empty for
# unknown values. Tightening would lock us into a code release every time
# we add a job type.
_JOB_STATUSES = {"queued", "running", "complete", "failed", "cancelled"}


@router.get(
    "",
    response_model=JobsPage,
    status_code=status.HTTP_200_OK,
    summary="List jobs (newest first), optionally filtered",
)
def list_jobs(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status_: str | None = Query(
        None,
        alias="status",
        description=(
            "Filter by status (queued | running | complete | failed | "
            "cancelled). Unknown values return an empty page."
        ),
    ),
    type_: str | None = Query(
        None,
        alias="type",
        description="Filter by job type slug (e.g. action_run, scene_onboard).",
    ),
    project_id: str | None = Query(
        None,
        description="Filter to one project's jobs (wire id project_<uuid>).",
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> JobsPage:
    """Return a page of jobs newest-first.

    Backs the Jobs destination (Step 19). Sorts by created_at desc so the
    UI defaults to "what just happened" — most useful while the demo is
    actively producing work. Worker claim queries are unaffected (they go
    through jobs_claim_idx with their own ORDER BY).
    """
    filters = []
    if status_ is not None:
        if status_ not in _JOB_STATUSES:
            # Don't 422 — just return zero rows. Frontend filters drift
            # ahead of backend renames otherwise.
            return JobsPage(items=[], total=0, limit=limit, offset=offset)
        filters.append(Job.status == status_)
    if type_ is not None:
        filters.append(Job.type == type_)
    if project_id is not None:
        pid = parse_prefixed_id("project", project_id)
        filters.append(Job.project_id == pid)

    total = db.scalar(select(func.count(Job.id)).where(*filters)) or 0
    rows = list(
        db.scalars(
            select(Job)
            .where(*filters)
            .order_by(Job.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    )
    return JobsPage(
        items=[JobPublic.from_orm_job(j) for j in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{job_id}",
    response_model=JobPublic,
    status_code=status.HTTP_200_OK,
    summary="Get one job by id",
)
def get_job(
    job_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> JobPublic:
    """Return one job's row by wire-format id.

    Authenticated users may read any job (Library is shared, Jobs is too —
    project-scoping for project-scoped jobs lands with the projects ACL in
    Step 10/11). Returns 404 with detail="job_not_found" when the id
    parses but no row exists.
    """
    raw_id = parse_prefixed_id("job", job_id)
    job = db.get(Job, raw_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job_not_found",
        )
    return JobPublic.from_orm_job(job)
