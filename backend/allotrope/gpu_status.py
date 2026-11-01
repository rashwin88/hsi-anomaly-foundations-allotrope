"""
nvidia-smi sampling and the worker GPU snapshot, shared by the api and the worker.

The worker samples on a timer (allotrope_worker/status_sampler.py) and writes the latest snapshot to the
``worker_status`` table; ``GET /metrics/host`` (api/metrics.py) reports its own host's nvidia-smi when it has one,
else the freshest worker snapshot.

Why: the GPU compose override gives the GPU to the worker only, where the models run, so the api's own nvidia-smi
always said "no GPU", even on the H200 VM (docs/proposals/class-driven-flow.md, addition E). The worker also reports
the torch device it actually uses, so "GPU present but jobs on CPU" shows.

Gotchas: nvidia-smi's CSV may print "[N/A]" / "[Not Supported]" for a field (MIG, some datacentre cards); such a
field is None, the device is kept. A snapshot older than STALE_AFTER_S is reported as unavailable, not as the GPU.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from typing import Any

QUERY_FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
# A worker snapshot older than this is stale: the worker is down or wedged, and its numbers are not the GPU now.
# Matches the reaper's default heartbeat timeout (WORKER_HEARTBEAT_TIMEOUT_SEC).
STALE_AFTER_S = 60.0
_KEYS = ("utilization_percent", "memory_used_mb", "memory_total_mb", "temperature_c")


def parse_smi_csv(text: str) -> list[dict[str, Any]]:
    """Devices from ``nvidia-smi --query-gpu=QUERY_FIELDS --format=csv,noheader,nounits`` output."""
    devices: list[dict[str, Any]] = []
    for line in (raw.strip() for raw in text.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if not line or len(parts) < 5 or parts[0].lower() == "name":
            continue
        dev: dict[str, Any] = {"name": parts[0]}
        for key, value in zip(_KEYS, parts[1:5]):
            try:
                dev[key] = float(value)
            except ValueError:
                dev[key] = None
        devices.append(dev)
    return devices


def sample_nvidia_smi(timeout_s: float = 2.0) -> tuple[list[dict[str, Any]], str]:
    """(devices, note). No nvidia-smi or an error gives no devices and a note saying why; never raises."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return [], "nvidia-smi not on PATH (no NVIDIA driver in this container)"
    try:
        proc = subprocess.run([smi, f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=timeout_s, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return [], f"nvidia-smi error: {exc!r}"
    devices = parse_smi_csv(proc.stdout)
    return devices, "" if devices else "nvidia-smi listed no GPU"


def freshest(rows: list[Any], now: datetime) -> tuple[Any, float] | None:
    """(the most recently updated row, its age in seconds) of rows with ``updated_at``, or None."""
    if not rows:
        return None
    best = max(rows, key=lambda r: r.updated_at)
    return best, max(0.0, (now - best.updated_at).total_seconds())
