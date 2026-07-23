"""Host + workload metrics (Step 20).

Routes:
    GET /metrics/host       psutil + nvidia-smi snapshot of the api host
    GET /metrics/workload   Postgres-derived queue + throughput numbers

The api host on the demo machine is Docker Desktop's VM (Mac). That VM
has no GPU passthrough, so /metrics/host reports `gpu.available=false`
unless an NVIDIA stack is wired (Linux + nvidia-container-toolkit + the
worker's GPU compose override). The host endpoint never raises on
GPU absence — it just reports it.

These are SNAPSHOT endpoints, not streams. The frontend polls every
~1 s for the topbar sparklines, every ~5 s for the Monitoring page —
cheap, no websockets in v1.

Sequence diagram: final design/diagrams/metrics-host.drawio
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Job
from .deps import current_user_claims

logger = logging.getLogger("allotrope.api.metrics")

router = APIRouter(prefix="/metrics", tags=["metrics"])


# --- Schemas ----------------------------------------------------------


class CpuMetrics(BaseModel):
    percent: float                 # overall % busy across all logical cores
    count_logical: int
    load_average_1m: float | None  # None on platforms without it (Windows)


class MemoryMetrics(BaseModel):
    total_bytes: int
    used_bytes: int
    available_bytes: int
    percent: float


class DiskMetrics(BaseModel):
    mountpoint: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent: float


class GpuMetrics(BaseModel):
    available: bool
    note: str = ""
    devices: list[dict[str, Any]] = Field(default_factory=list)


class HostMetrics(BaseModel):
    timestamp: datetime
    cpu: CpuMetrics
    memory: MemoryMetrics
    disks: list[DiskMetrics]
    gpu: GpuMetrics


class JobsCountByStatus(BaseModel):
    queued: int = 0
    running: int = 0
    complete: int = 0
    failed: int = 0
    cancelled: int = 0


class WorkloadMetrics(BaseModel):
    timestamp: datetime
    queue_depth: int
    by_status: JobsCountByStatus
    by_type_running: dict[str, int]
    by_type_queued: dict[str, int]
    # Number of jobs that completed in the last 60 / 600 / 3600 seconds
    # (rolling windows). Drives the throughput sparkline.
    completed_last_minute: int
    completed_last_10_minutes: int
    completed_last_hour: int
    # Mean / p95 wall-clock seconds for jobs that completed in the last
    # hour. None if the bucket is empty.
    completed_mean_seconds: float | None
    completed_p95_seconds: float | None
    # Largest gap between a queued row's created_at and now() — the
    # canonical "is the queue keeping up?" signal.
    oldest_queued_age_seconds: float | None


# --- Helpers ---------------------------------------------------------


def _disks() -> list[DiskMetrics]:
    """Snapshot only the mountpoints we actually care about for the demo.

    On the api container these are the three named volume mounts
    (/data, /artifacts, /models). The container's `/` is short-lived
    and not interesting to surface.
    """
    paths = ["/data", "/artifacts", "/models"]
    out: list[DiskMetrics] = []
    for p in paths:
        if not Path(p).is_dir():
            continue
        try:
            u = psutil.disk_usage(p)
        except OSError:
            continue
        out.append(
            DiskMetrics(
                mountpoint=p,
                total_bytes=u.total,
                used_bytes=u.used,
                free_bytes=u.free,
                percent=u.percent,
            )
        )
    # Fallback: report `/` if none of the named mounts exist (helps
    # local dev without compose volumes).
    if not out:
        u = psutil.disk_usage("/")
        out.append(
            DiskMetrics(
                mountpoint="/",
                total_bytes=u.total,
                used_bytes=u.used,
                free_bytes=u.free,
                percent=u.percent,
            )
        )
    return out


def _gpu() -> GpuMetrics:
    """Best-effort GPU summary via `nvidia-smi`.

    Mac Docker has no MPS/CUDA passthrough, so this gracefully reports
    `available=false` rather than raising. On Linux + nvidia stack,
    parses the `--query-gpu` CSV for the standard fields.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return GpuMetrics(
            available=False,
            note="nvidia-smi not on PATH — Mac Docker has no GPU passthrough.",
        )
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return GpuMetrics(available=False, note=f"nvidia-smi error: {exc!r}")

    devices: list[dict[str, Any]] = []
    for line in (l.strip() for l in proc.stdout.splitlines() if l.strip()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append(
                {
                    "name": parts[0],
                    "utilization_percent": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "temperature_c": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return GpuMetrics(available=bool(devices), devices=devices)


def _cpu() -> CpuMetrics:
    # interval=None → instantaneous read against the cached prior snapshot,
    # so polling at 1 Hz from the topbar isn't latency-blocking.
    pct = psutil.cpu_percent(interval=None)
    load1: float | None
    if hasattr(os, "getloadavg"):
        try:
            load1 = float(os.getloadavg()[0])
        except OSError:
            load1 = None
    else:
        load1 = None
    return CpuMetrics(
        percent=pct,
        count_logical=psutil.cpu_count(logical=True) or 0,
        load_average_1m=load1,
    )


def _memory() -> MemoryMetrics:
    vm = psutil.virtual_memory()
    return MemoryMetrics(
        total_bytes=vm.total,
        used_bytes=vm.used,
        available_bytes=vm.available,
        percent=vm.percent,
    )


# --- /metrics/host ---------------------------------------------------


# Warm up psutil's CPU sampler at import time so the first /metrics/host
# read isn't a blank baseline. The follow-up reads come from this seed.
psutil.cpu_percent(interval=None)


@router.get(
    "/host",
    response_model=HostMetrics,
    status_code=status.HTTP_200_OK,
    summary="Host CPU / RAM / disk / GPU snapshot",
)
def get_host_metrics(
    _claims: Claims = Depends(current_user_claims),
) -> HostMetrics:
    return HostMetrics(
        timestamp=datetime.now(timezone.utc),
        cpu=_cpu(),
        memory=_memory(),
        disks=_disks(),
        gpu=_gpu(),
    )


# --- /metrics/workload -----------------------------------------------


@router.get(
    "/workload",
    response_model=WorkloadMetrics,
    status_code=status.HTTP_200_OK,
    summary="Queue depth + rolling throughput from the jobs table",
)
def get_workload_metrics(
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> WorkloadMetrics:
    now = datetime.now(timezone.utc)
    # Status totals — single aggregate query is enough.
    counts_rows = db.execute(
        select(Job.status, func.count(Job.id)).group_by(Job.status)
    ).all()
    by_status = JobsCountByStatus()
    for st, n in counts_rows:
        if hasattr(by_status, st):
            setattr(by_status, st, int(n))

    # Type breakdown for the rows that are visibly active.
    type_running_rows = db.execute(
        select(Job.type, func.count(Job.id))
        .where(Job.status == "running")
        .group_by(Job.type)
    ).all()
    type_queued_rows = db.execute(
        select(Job.type, func.count(Job.id))
        .where(Job.status == "queued")
        .group_by(Job.type)
    ).all()
    by_type_running = {str(t): int(n) for t, n in type_running_rows}
    by_type_queued = {str(t): int(n) for t, n in type_queued_rows}

    # Rolling completion windows. completed_at is set on the terminal
    # transition by the worker (Step 6c).
    one_min = func.now() - func.cast("1 minute", postgresql_type=None)  # noqa: F841 — placeholder
    # SQLAlchemy core makes interval arithmetic awkward; use raw SQL.
    completed_last_minute = (
        db.scalar(
            select(func.count(Job.id)).where(
                Job.status == "complete",
                Job.completed_at >= func.now() - func.make_interval(0, 0, 0, 0, 0, 1, 0),
            )
        )
        or 0
    )
    completed_last_10_minutes = (
        db.scalar(
            select(func.count(Job.id)).where(
                Job.status == "complete",
                Job.completed_at >= func.now() - func.make_interval(0, 0, 0, 0, 0, 10, 0),
            )
        )
        or 0
    )
    completed_last_hour = (
        db.scalar(
            select(func.count(Job.id)).where(
                Job.status == "complete",
                Job.completed_at >= func.now() - func.make_interval(0, 0, 0, 0, 1, 0, 0),
            )
        )
        or 0
    )

    # Latency over the last hour. extract(epoch FROM completed_at - started_at).
    secs_col = func.extract("epoch", Job.completed_at - Job.started_at)
    mean_secs = db.scalar(
        select(func.avg(secs_col)).where(
            Job.status == "complete",
            Job.started_at.is_not(None),
            Job.completed_at >= func.now() - func.make_interval(0, 0, 0, 0, 1, 0, 0),
        )
    )
    # Postgres percentile_cont aggregate
    p95_secs = db.scalar(
        select(func.percentile_cont(0.95).within_group(secs_col)).where(
            Job.status == "complete",
            Job.started_at.is_not(None),
            Job.completed_at >= func.now() - func.make_interval(0, 0, 0, 0, 1, 0, 0),
        )
    )

    oldest_queued_age_seconds = db.scalar(
        select(
            func.extract("epoch", func.now() - func.min(Job.created_at))
        ).where(Job.status == "queued")
    )

    return WorkloadMetrics(
        timestamp=now,
        queue_depth=by_status.queued + by_status.running,
        by_status=by_status,
        by_type_running=by_type_running,
        by_type_queued=by_type_queued,
        completed_last_minute=int(completed_last_minute),
        completed_last_10_minutes=int(completed_last_10_minutes),
        completed_last_hour=int(completed_last_hour),
        completed_mean_seconds=(float(mean_secs) if mean_secs is not None else None),
        completed_p95_seconds=(float(p95_secs) if p95_secs is not None else None),
        oldest_queued_age_seconds=(
            float(oldest_queued_age_seconds)
            if oldest_queued_age_seconds is not None
            else None
        ),
    )


# Silence "imported but unused" linter complaints about Path/time/settings —
# we keep them imported because the file is a natural home for future
# extensions (per-volume IO counters, p99 latencies, etc.).
_ = Path, time, settings
