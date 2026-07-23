"""Stale-job reaper.

A `status='running'` row whose `last_heartbeat_at` is older than the
threshold is assumed to be owned by a dead worker (process crashed, OOM,
node power loss, …). The reaper flips those rows to `failed` so:

    - The api/UI no longer shows the job as "still working".
    - For `action_run` jobs, the api can mirror that into `actions.status`
      via its own logic (Step 12).

Concurrency safety:
    Multiple workers calling `reap_stale` at the same time is fine. The
    UPDATE filters by `status='running'`, so whichever transaction commits
    first changes the row to `'failed'`; the others find no matching rows
    and update zero. No locks needed.

Wall-clock note:
    The cutoff is computed in Python (UTC) and compared against
    `last_heartbeat_at` in Postgres. Both are timestamptz so this is
    apples-to-apples regardless of the worker container's TZ env.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from allotrope.models import Action, Job

logger = logging.getLogger("allotrope.worker.reaper")


def reap_stale(session: Session, timeout_sec: float) -> int:
    """Mark `status='running'` rows with no recent heartbeat as failed.

    Also mirrors the failure onto any `actions` row whose `action_run`
    Job we just reaped — keeping the lockstep invariant from
    abstractions-spec § 5.5 ('actions.status and jobs.status are aligned
    at every transaction boundary') alive even when a worker dies
    mid-handler. Without this mirror, the api UI would show the Action
    stuck at 'running' forever because the handler's exception path
    never ran.

    Returns the number of Job rows reaped (0 most ticks).
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=timeout_sec)

    # First: find the stale running jobs so we can also mirror onto
    # actions before flipping the Job rows. We need the action_run
    # subset's target_ids — those are the action_uuids.
    stale_action_run_ids = list(
        session.scalars(
            select(Job.target_id)
            .where(Job.status == "running")
            .where(Job.last_heartbeat_at < cutoff)
            .where(Job.type == "action_run")
            .where(Job.target_kind == "action")
        ).all()
    )
    stale_action_run_ids = [u for u in stale_action_run_ids if u is not None]

    failure_reason = f"worker heartbeat lost (>{int(timeout_sec)}s)"

    # Mirror onto actions first. Filter on status='running' to avoid
    # racing with a worker that JUST finished the action — if the
    # action already wrote complete/failed, leave it.
    actions_mirrored = 0
    if stale_action_run_ids:
        mirror_stmt = (
            update(Action)
            .where(Action.id.in_(stale_action_run_ids))
            .where(Action.status == "running")
            .values(
                status="failed",
                completed_at=now,
                failure_reason=failure_reason,
            )
        )
        mirror_result = session.execute(mirror_stmt)
        actions_mirrored = mirror_result.rowcount or 0

    # Now flip the Job rows.
    stmt = (
        update(Job)
        .where(Job.status == "running")
        .where(Job.last_heartbeat_at < cutoff)
        .values(
            status="failed",
            failure_reason=failure_reason,
            completed_at=now,
            claimed_by=None,
        )
    )
    result = session.execute(stmt)
    session.commit()
    count = result.rowcount or 0
    if count > 0:
        logger.warning(
            "reaped %d stale running job(s) (heartbeat older than %.0fs); "
            "mirrored %d onto actions",
            count,
            timeout_sec,
            actions_mirrored,
        )
    return count
