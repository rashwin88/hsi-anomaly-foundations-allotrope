"""WorkerStatus: the latest GPU / device snapshot each worker reports (allotrope/gpu_status.py).

One row per worker process (``worker_id`` is host/random, new at each start), upserted every few seconds by
allotrope_worker/status_sampler.py; rows a day old are pruned by the same writer. Read by GET /metrics/host.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class WorkerStatus(Base):
    __tablename__ = "worker_status"

    worker_id: Mapped[str] = mapped_column(Text, primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # nvidia-smi devices: [{name, utilization_percent, memory_used_mb, memory_total_mb, temperature_c}]
    gpu_devices: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    gpu_note: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    # The torch device the worker's models run on: "cuda:0 (<name>)", "mps" or "cpu".
    torch_device: Mapped[str | None] = mapped_column(Text, nullable=True)
