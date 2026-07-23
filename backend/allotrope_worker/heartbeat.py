"""Background heartbeat thread.

While a handler is running, this thread bumps `jobs.last_heartbeat_at = now()`
on a fixed interval. The reaper uses that timestamp to decide which
`status='running'` rows are owned by a still-alive worker vs ones whose
worker crashed mid-job.

Why a thread (not asyncio, not signal-driven):
    - Handlers are sync Python — torch / numpy / file I/O. asyncio would
      either require turning every handler into a coroutine (we won't) or
      blocking the event loop (defeats the point).
    - signal.SIGALRM-based timers are process-wide and clash with the
      shutdown handlers we already register.
    - One daemon thread per running job is cheap, well-isolated, and exits
      when the process exits.

Each tick uses its own short-lived Session — sharing the runner's session
would block whichever connection is doing the actual work.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

from sqlalchemy import update

from allotrope.db import SessionLocal
from allotrope.models import Job

logger = logging.getLogger("allotrope.worker.heartbeat")


class Heartbeat:
    """Drive periodic heartbeat updates for a single in-flight job.

    Usage:
        hb = Heartbeat(job_id, interval_sec=5.0)
        hb.start()
        try:
            ... handler runs ...
        finally:
            hb.stop()
    """

    def __init__(self, job_id: uuid.UUID, interval_sec: float) -> None:
        self._job_id = job_id
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"hb-{str(job_id)[:8]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_sec: float = 5.0) -> None:
        """Signal the thread to exit and wait briefly for it.

        We don't care if the thread is mid-update when we ask it to stop —
        the next runner step (mark_complete / mark_failed) will overwrite
        the same row's state anyway.
        """
        self._stop.set()
        self._thread.join(timeout=timeout_sec)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        # First tick happens after one interval — we already wrote
        # `last_heartbeat_at` at claim time, so an immediate tick is
        # redundant.
        while not self._stop.wait(self._interval):
            try:
                self._beat()
            except Exception:
                # A failed heartbeat is recoverable — the next tick will
                # try again. Only escalate (= let the reaper take it) if
                # heartbeats stay broken past the stale threshold.
                logger.exception(
                    "heartbeat tick failed for job %s", self._job_id
                )

    def _beat(self) -> None:
        now = datetime.now(timezone.utc)
        session = SessionLocal()
        try:
            session.execute(
                update(Job)
                .where(Job.id == self._job_id)
                .values(last_heartbeat_at=now)
            )
            session.commit()
            logger.debug("heartbeat job=%s @%s", self._job_id, now.isoformat())
        finally:
            session.close()
