"""Foundation model catalog endpoints (Step "Models").

Routes:
    GET /models                  — list all 7 reconstruction architectures
    GET /models/{architecture}   — single model + full metadata

The catalog is read directly from the per-architecture `current.json`
manifests under `${MODELS_DIR}/<arch>/current.json` (mounted from the
`allotrope_models` named volume). Manifests are the source of truth;
the api adds no DB row for them.

Sequence diagrams:
    final design/diagrams/models-list.drawio
    final design/diagrams/models-detail.drawio
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..auth.jwt import Claims
from ..config import settings
from .deps import current_user_claims

logger = logging.getLogger("allotrope.api.models")

router = APIRouter(prefix="/models", tags=["models"])


# --- Schemas ----------------------------------------------------------


class CodenameBlock(BaseModel):
    name: str
    script: str
    meaning: str
    why: str


class CurrentCheckpoint(BaseModel):
    file: str
    version: str
    epoch: int
    params: int
    val_loss: float
    encoder_dims: str | None = None
    decoder_dim: int | None = None
    spectral_dim_D: int | None = None


class NormalizationBlock(BaseModel):
    mode: str  # "baked_in" | "none"
    applied_in_forward: bool
    stats_shape: list[int] | None
    mean: float | str | None
    std: float | str | None
    source: str


class ModelSummary(BaseModel):
    """Compact wire shape for the catalog list view."""

    architecture: str
    label: str
    codename: CodenameBlock
    sensor: str
    family: str
    params: int
    val_loss: float
    version: str
    epoch: int
    normalization_mode: str
    doc: str
    # Anomaly-scoring capabilities (driven by the per-architecture
    # capability table in foundation_models.resolver). The
    # anomaly_scoring NewActionDialog drives per-model scoring +
    # patch/stride/batch override pickers from these.
    scoring_methods: list[str] = []
    default_scoring_method: str | None = None
    default_patch_size: int | None = None
    default_stride: int | None = None
    default_batch_size: int | None = None
    valid_patch_sizes: list[int] = []


class ModelDetail(BaseModel):
    """Full wire shape for the per-model detail view."""

    architecture: str
    label: str
    codename: CodenameBlock
    sensor: str
    family: str
    current: CurrentCheckpoint
    alternatives: list[dict[str, Any]] = []
    normalization: NormalizationBlock
    doc: str
    inferencer: str
    inferencer_module: str
    notes: str
    updated_at: str | None = None


# --- Loader ------------------------------------------------------------


def _models_root() -> Path:
    return Path(settings.models_dir)


def _read_manifest(architecture: str) -> dict[str, Any]:
    path = _models_root() / architecture / "current.json"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown model architecture: {architecture}",
        )
    return json.loads(path.read_text())


def _list_architectures() -> list[str]:
    root = _models_root()
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "current.json").exists()
    )


def _summary_from_manifest(m: dict[str, Any]) -> ModelSummary:
    # Pull scoring capabilities from the resolver's per-architecture
    # capability table. Failure (e.g. a brand-new architecture without
    # an entry) leaves the fields empty — the anomaly_scoring dialog
    # treats that as "model not available for scoring yet".
    from ..foundation_models.resolver import _CAPABILITIES  # local import

    cur = m["current"]
    caps = _CAPABILITIES.get(m["architecture"])
    return ModelSummary(
        architecture=m["architecture"],
        label=m["label"],
        codename=CodenameBlock(**m["codename"]),
        sensor=m["sensor"],
        family=m["family"],
        params=cur["params"],
        val_loss=cur["val_loss"],
        version=cur["version"],
        epoch=cur["epoch"],
        normalization_mode=m.get("normalization", {}).get("mode", "unknown"),
        doc=m["doc"],
        scoring_methods=list(caps.scoring_methods) if caps else [],
        default_scoring_method=caps.default_scoring_method if caps else None,
        default_patch_size=caps.default_patch_size if caps else None,
        default_stride=caps.default_stride if caps else None,
        default_batch_size=caps.default_batch_size if caps else None,
        valid_patch_sizes=list(caps.valid_patch_sizes) if caps else [],
    )


# --- Routes -----------------------------------------------------------


@router.get("", response_model=list[ModelSummary])
def list_models(
    _claims: Claims = Depends(current_user_claims),
) -> list[ModelSummary]:
    """List every model with a manifest. Foundation models first (sorted
    by val_loss ascending, best first), classical detectors last (val_loss=0
    isn't comparable to a learned loss, so they get their own section)."""
    archs = _list_architectures()
    summaries = [_summary_from_manifest(_read_manifest(a)) for a in archs]
    return sorted(
        summaries,
        key=lambda s: (
            1 if s.family == "classical" else 0,
            s.val_loss,
        ),
    )


@router.get("/{architecture}", response_model=ModelDetail)
def get_model(
    architecture: str,
    _claims: Claims = Depends(current_user_claims),
) -> ModelDetail:
    """Read a single model manifest by architecture slug."""
    m = _read_manifest(architecture)
    return ModelDetail(
        architecture=m["architecture"],
        label=m["label"],
        codename=CodenameBlock(**m["codename"]),
        sensor=m["sensor"],
        family=m["family"],
        current=CurrentCheckpoint(**m["current"]),
        alternatives=m.get("alternatives", []),
        normalization=NormalizationBlock(**m["normalization"]),
        doc=m["doc"],
        inferencer=m["inferencer"],
        inferencer_module=m["inferencer_module"],
        notes=m.get("notes", ""),
        updated_at=m.get("updated_at"),
    )
