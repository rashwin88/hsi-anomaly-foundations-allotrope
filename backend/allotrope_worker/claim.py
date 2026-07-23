"""Job claim path — `SELECT … FOR UPDATE SKIP LOCKED`.

One claim attempt per tick:
    1. BEGIN
    2. SELECT one queued row of a supported type, FOR UPDATE SKIP LOCKED
    3. If none → COMMIT (no-op, fall back to sleep)
    4. UPDATE that row: status='running', started_at=now(),
       claimed_by=<worker_id>, last_heartbeat_at=now()
    5. COMMIT — releases the row lock; from here, *application-level*
       atomicity (the heartbeat) protects ownership

Why two phases (lock-then-update-then-commit, then handler runs without a
held row lock)? A held row lock blocks the reaper and the api's own status
reads. The heartbeat column is the substitute: a worker with a fresh
heartbeat owns the row in *behaviour* even though the row isn't locked.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from allotrope.models import Job

from .handlers import supported_types

logger = logging.getLogger("allotrope.worker.claim")


def claim_one(session: Session, worker_id: str) -> Job | None:
    """Claim a single queued job. Returns None if the queue is empty.

    Runs inside its own short transaction (the caller does NOT need to
    open one). On return, the job row is `status='running'` with
    `claimed_by=worker_id` already committed.
    """
    types = supported_types()

    # FIFO over (status='queued') AND (type ∈ supported), with the per-row
    # lock that SKIP LOCKED makes nonblocking across replicas.
    stmt = (
        select(Job)
        .where(Job.status == "queued")
        .where(Job.type.in_(types))
        .order_by(Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    job = session.scalars(stmt).first()
    if job is None:
        # Important: still commit so we don't leave an idle-in-transaction
        # connection holding a snapshot. The empty SELECT is harmless to
        # commit.
        session.commit()
        return None

    now = datetime.now(timezone.utc)
    job.status = "running"
    job.started_at = now
    job.last_heartbeat_at = now
    job.claimed_by = worker_id
    session.commit()

    logger.info(
        "claimed job id=%s type=%s (worker=%s)",
        job.id,
        job.type,
        worker_id,
    )
    return job


def mark_complete(
    session: Session,
    job: Job,
    target_kind: str | None,
    target_id: uuid.UUID | None,
) -> None:
    """Mark a running job as complete and record its produced target."""
    now = datetime.now(timezone.utc)
    job.status = "complete"
    job.completed_at = now
    job.last_heartbeat_at = now
    job.target_kind = target_kind
    job.target_id = target_id
    job.claimed_by = None
    session.commit()
    logger.info("completed job id=%s type=%s", job.id, job.type)


def mark_failed(
    session: Session,
    job: Job,
    reason: str,
) -> None:
    """Mark a running job as failed with a human-readable reason."""
    now = datetime.now(timezone.utc)
    job.status = "failed"
    job.completed_at = now
    job.last_heartbeat_at = now
    job.failure_reason = reason[:1000]  # cap, in case of huge tracebacks
    job.claimed_by = None
    session.commit()
    logger.warning(
        "failed job id=%s type=%s reason=%s",
        job.id,
        job.type,
        reason.splitlines()[0] if reason else "",
    )
