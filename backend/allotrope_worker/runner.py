"""Worker main loop.

Each tick:
    1. If `WORKER_REAP_INTERVAL_SEC` has elapsed since last reap, run
       `reap_stale()`. This catches jobs whose owning worker died.
    2. Open a Session and try to claim one queued job
       (`SELECT … FOR UPDATE SKIP LOCKED`).
    3. If claimed, start a `Heartbeat` thread, dispatch to the per-type
       handler, then stop the heartbeat.
    4. On success → `mark_complete(target_kind, target_id)`.
       On exception → `mark_failed(reason=str(exc))`.
    5. Close session, loop.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
import traceback
import uuid

from allotrope.db import SessionLocal

from .claim import claim_one, mark_complete, mark_failed
from .cleanup import reclaim_memory
from .handlers import HANDLERS, supported_types
from .heartbeat import Heartbeat
from .reaper import reap_stale

logger = logging.getLogger("allotrope.worker")


def _configure_logging() -> None:
    level = os.environ.get("WORKER_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def _worker_id() -> str:
    return f"{socket.gethostname()}/{uuid.uuid4().hex[:8]}"


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        logger.info("shutdown requested (%s) — finishing current tick", name)
        self.requested = True


class _Config:
    """Resolved-at-startup tuning knobs (all env-driven)."""

    def __init__(self) -> None:
        self.poll_interval = float(
            os.environ.get("WORKER_POLL_INTERVAL_SEC", "2.0")
        )
        self.heartbeat_interval = float(
            os.environ.get("WORKER_HEARTBEAT_INTERVAL_SEC", "5.0")
        )
        # Threshold past which a `running` row with no fresh heartbeat is
        # reaped. Must be ≥ ~3× heartbeat_interval to absorb jitter.
        self.heartbeat_timeout = float(
            os.environ.get("WORKER_HEARTBEAT_TIMEOUT_SEC", "60.0")
        )
        self.reap_interval = float(
            os.environ.get("WORKER_REAP_INTERVAL_SEC", "30.0")
        )

    def summary(self) -> str:
        return (
            f"poll={self.poll_interval:.1f}s "
            f"heartbeat={self.heartbeat_interval:.1f}s "
            f"timeout={self.heartbeat_timeout:.0f}s "
            f"reap_every={self.reap_interval:.0f}s"
        )


def _process_one_tick(worker_id: str, cfg: _Config) -> bool:
    """Run a single claim-and-dispatch tick.

    Returns True if a job was processed (success or failure), False if the
    queue was empty.
    """
    session = SessionLocal()
    job = None  # bound for the finally clause even on early return
    try:
        job = claim_one(session, worker_id)
        if job is None:
            return False

        handler = HANDLERS.get(job.type)
        if handler is None:
            mark_failed(
                session, job, f"no handler registered for type={job.type!r}"
            )
            return True

        # Heartbeat lives only for the duration of the handler call. The
        # try/finally guarantees stop() runs even on a handler exception.
        hb = Heartbeat(job.id, cfg.heartbeat_interval)
        hb.start()
        try:
            target_kind, target_id = handler(session, job)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("handler raised for job id=%s", job.id)
            hb.stop()
            mark_failed(
                session, job, f"{exc.__class__.__name__}: {exc}\n{tb}"
            )
            return True
        else:
            hb.stop()

        mark_complete(session, job, target_kind, target_id)
        return True
    finally:
        session.close()
        # Drop torch / numpy / vendable references the handler held, then
        # ask glibc to return arena pages to the OS. Only worth doing
        # when we actually ran a job — empty ticks have nothing to free.
        # See backend/allotrope_worker/cleanup.py for the rationale.
        if job is not None:
            reclaim_memory(reason=f"job={job.id}")


def _maybe_reap(state: dict[str, float], cfg: _Config) -> None:
    """Run the reaper if the configured interval has elapsed."""
    now = time.monotonic()
    if now - state["last_reap_at"] < cfg.reap_interval:
        return
    state["last_reap_at"] = now
    session = SessionLocal()
    try:
        reap_stale(session, cfg.heartbeat_timeout)
    except Exception:
        logger.exception("reap pass failed; will retry next interval")
    finally:
        session.close()


def main() -> None:
    _configure_logging()
    cfg = _Config()
    worker_id = _worker_id()

    flag = _ShutdownFlag()
    signal.signal(signal.SIGTERM, flag.request)
    signal.signal(signal.SIGINT, flag.request)

    logger.info(
        "worker starting (id=%s %s types=%s)",
        worker_id,
        cfg.summary(),
        ",".join(supported_types()),
    )

    # Reap on the very first tick so a worker coming up after a crash
    # cleans up its predecessor's stale rows promptly.
    state: dict[str, float] = {"last_reap_at": 0.0}

    while not flag.requested:
        try:
            _maybe_reap(state, cfg)
            did_work = _process_one_tick(worker_id, cfg)
        except Exception:
            logger.exception("tick failed; backing off")
            did_work = False

        if did_work:
            continue

        slept = 0.0
        slice_sec = 0.25
        while slept < cfg.poll_interval and not flag.requested:
            time.sleep(slice_sec)
            slept += slice_sec

    logger.info("worker stopped cleanly (id=%s)", worker_id)
