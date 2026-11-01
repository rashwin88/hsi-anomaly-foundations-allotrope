"""
Background thread: the worker's GPU snapshot (nvidia-smi + its torch device), written to ``worker_status``.

Started once by runner.main(), it runs while the worker is idle too, so the monitoring pane shows the GPU before a
job starts. GET /metrics/host reads the freshest row (api/metrics.py), because only the worker container is given
the GPU (docs/proposals/class-driven-flow.md, addition E).

A failed sample or write is logged and retried on the next tick; it never touches job processing. Rows older than
PRUNE_AFTER are deleted by the same writer, since each worker start has a new worker_id.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert

from allotrope.db import SessionLocal
from allotrope.gpu_status import sample_nvidia_smi
from allotrope.models import WorkerStatus

logger = logging.getLogger("allotrope.worker.status")

PRUNE_AFTER = timedelta(days=1)


def torch_device_label() -> str:
    """The device the models run on - the same choice as app/utils/torch_helpers/device_selection.get_device."""
    import torch

    if torch.cuda.is_available():
        return f"cuda:0 ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StatusSampler:
    """Every ``interval_sec``: sample, then upsert this worker's row."""

    def __init__(self, worker_id: str, interval_sec: float,
                 sample: Callable = sample_nvidia_smi, device: Callable[[], str] = torch_device_label) -> None:
        self._worker_id = worker_id
        self._interval = interval_sec
        self._sample, self._device = sample, device
        self._device_label: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="worker-status", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_sec)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("worker status tick failed")
            self._stop.wait(self._interval)

    def snapshot(self) -> dict:
        """The row's values now. The torch device is resolved once: it does not change while the process lives."""
        if self._device_label is None:
            try:
                self._device_label = self._device()
            except Exception as exc:          # noqa: BLE001 - reported, not fatal
                self._device_label = f"unknown ({type(exc).__name__})"
        devices, note = self._sample()
        return {"worker_id": self._worker_id, "updated_at": datetime.now(timezone.utc), "gpu_devices": devices,
                "gpu_note": note, "torch_device": self._device_label}

    def tick(self) -> None:
        row = self.snapshot()
        session = SessionLocal()
        try:
            stmt = insert(WorkerStatus).values(**row)
            session.execute(stmt.on_conflict_do_update(
                index_elements=[WorkerStatus.worker_id],
                set_={k: stmt.excluded[k] for k in ("updated_at", "gpu_devices", "gpu_note", "torch_device")}))
            session.execute(delete(WorkerStatus).where(WorkerStatus.updated_at < row["updated_at"] - PRUNE_AFTER))
            session.commit()
        finally:
            session.close()
