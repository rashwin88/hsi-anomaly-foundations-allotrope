"""
GPU monitoring: nvidia-smi parsing, the worker's snapshot, and /metrics/host's fallback to it.

The local stack has no GPU, so nvidia-smi output is a captured CSV (an H200 and an L40S line) and the torch device is
mocked. The api half needs fastapi (api image); the parsing and the sampler do not.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")

# nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
CAPTURED = """NVIDIA H200, 87, 61234, 143771, 54
NVIDIA L40S, [N/A], 512, 46068, 31
"""


def test_parse_captured_csv():
    from allotrope.gpu_status import parse_smi_csv

    devs = parse_smi_csv(CAPTURED)
    assert devs[0] == {"name": "NVIDIA H200", "utilization_percent": 87.0, "memory_used_mb": 61234.0,
                       "memory_total_mb": 143771.0, "temperature_c": 54.0}
    assert devs[1]["name"] == "NVIDIA L40S" and devs[1]["utilization_percent"] is None
    assert parse_smi_csv("") == [] and parse_smi_csv("name, utilization.gpu, a, b, c\n") == []


def test_freshest_and_age():
    from allotrope.gpu_status import freshest

    now = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    rows = [SimpleNamespace(worker_id="a", updated_at=now - timedelta(seconds=90)),
            SimpleNamespace(worker_id="b", updated_at=now - timedelta(seconds=4))]
    row, age = freshest(rows, now)
    assert row.worker_id == "b" and age == pytest.approx(4)
    assert freshest([], now) is None


def test_sampler_snapshot_with_a_mocked_device():
    pytest.importorskip("psycopg")                              # allotrope.db, imported by the worker module
    from allotrope.gpu_status import parse_smi_csv
    from allotrope_worker.status_sampler import StatusSampler

    calls = []
    s = StatusSampler("host/1234", 5.0, sample=lambda: (parse_smi_csv(CAPTURED)[:1], ""),
                      device=lambda: calls.append(1) or "cuda:0 (NVIDIA H200)")
    snap = s.snapshot()
    assert snap["torch_device"] == "cuda:0 (NVIDIA H200)" and snap["gpu_devices"][0]["name"] == "NVIDIA H200"
    s.snapshot()
    assert len(calls) == 1                                      # the device is resolved once
    cpu = StatusSampler("host/5678", 5.0, sample=lambda: ([], "nvidia-smi not on PATH"), device=lambda: "cpu")
    assert cpu.snapshot()["torch_device"] == "cpu" and cpu.snapshot()["gpu_devices"] == []


def _db_with(rows):
    db = MagicMock()
    db.scalars.return_value = rows
    return db


def _row(age_s, devices, device="cuda:0 (NVIDIA H200)", note=""):
    return SimpleNamespace(worker_id="worker-1/abcd", updated_at=datetime.now(timezone.utc) - timedelta(seconds=age_s),
                           gpu_devices=devices, gpu_note=note, torch_device=device)


def test_host_metrics_fall_back_to_the_worker(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("psutil")                               # api/metrics.py; the api image has it
    from allotrope import gpu_status
    from allotrope.api import metrics

    monkeypatch.setattr(gpu_status, "sample_nvidia_smi", lambda: ([], "nvidia-smi not on PATH"))
    devs = gpu_status.parse_smi_csv(CAPTURED)[:1]

    fresh = metrics._gpu(_db_with([_row(3, devs)]))
    assert fresh.available and fresh.source == "worker" and fresh.devices[0]["name"] == "NVIDIA H200"
    assert fresh.torch_device.startswith("cuda") and "from worker worker-1/abcd" in fresh.note

    stale = metrics._gpu(_db_with([_row(300, devs)]))
    assert not stale.available and "stale" in stale.note and stale.torch_device is None

    no_gpu = metrics._gpu(_db_with([_row(2, [], device="cpu", note="nvidia-smi not on PATH")]))
    assert not no_gpu.available and no_gpu.torch_device == "cpu" and no_gpu.source == "worker"

    nobody = metrics._gpu(_db_with([]))
    assert not nobody.available and nobody.source == "none" and "no worker" in nobody.note

    broken = MagicMock()
    broken.scalars.side_effect = RuntimeError("relation worker_status does not exist")
    assert not metrics._gpu(broken).available                  # unmigrated database: no 500


def test_api_host_gpu_wins(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("psutil")                               # api/metrics.py; the api image has it
    from allotrope import gpu_status
    from allotrope.api import metrics

    monkeypatch.setattr(gpu_status, "sample_nvidia_smi", lambda: (gpu_status.parse_smi_csv(CAPTURED), ""))
    out = metrics._gpu(_db_with([]))
    assert out.available and out.source == "api" and len(out.devices) == 2
