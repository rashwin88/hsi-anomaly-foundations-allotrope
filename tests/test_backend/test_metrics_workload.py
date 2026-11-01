"""
GET /metrics/workload builds its queries without raising (it returned 500 on a stray
func.cast(..., postgresql_type=None) placeholder). The Session is a stub: this checks that
every SQL expression constructs and the response model validates, not the SQL results.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")    # the api's deps; absent from the science-only venv

# allotrope.config builds Settings at import and requires these; nothing here connects.
os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")


def test_workload_metrics_with_an_empty_jobs_table():
    from allotrope.api.metrics import get_workload_metrics

    db = MagicMock()
    db.execute.return_value.all.return_value = []
    db.scalar.return_value = None
    out = get_workload_metrics(_claims=None, db=db)
    assert out.queue_depth == 0
    assert out.completed_last_hour == 0
    assert out.completed_p95_seconds is None
