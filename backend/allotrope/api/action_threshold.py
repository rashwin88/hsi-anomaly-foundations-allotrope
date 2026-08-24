"""
The interactive threshold flow for anomaly_detection_prep.

Split out of actions.py, which had grown past 1,300 lines. These three
endpoints form one conversation with the user and are meaningless apart:

    POST /actions/{id}/anomaly_detection_preview        try a threshold
    GET  /actions/{id}/anomaly_detection_preview_mask   fetch what it looked like
    POST /actions/{id}/anomaly_detection_commit         keep it

Why this is interactive at all: an absolute score cut is indefensible across
scenes - typical residuals differ by an order of magnitude between a calm lake
and a fire-affected scene - so a human picks the percentile by eye. The Action
sits at status `needs_threshold` until commit, which is the one place the
"complete implies exactly one output" invariant is deliberately relaxed.

They MUST stay together: preview stashes its PNG in a module-level LRU that the
mask endpoint reads back. Separating them would leave the mask endpoint reading
an empty dict and 404-ing on every call, with nothing to indicate why.

That cache is process-local, so it also breaks under more than one api worker.
Safe today only because the container runs a single-process uvicorn - see
docs/09-known-issues.md.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, Project
from ._action_common import action_or_404, output_for_action
from .deps import current_user_claims

logger = logging.getLogger("allotrope.api.action_threshold")

# Same prefix and tags as the actions_router in actions.py, so the mounted
# paths are byte-identical to before the split.
anomaly_threshold_router = APIRouter(prefix="/actions", tags=["actions"])

# --- POST /actions/{id}/anomaly_detection_preview -------------------
#
# Interactive Apply endpoint for prep actions sitting in
# ``needs_threshold``. The user moves a slider in the viewer, presses
# Apply, and this endpoint computes the binary anomaly mask + (if GT
# attached) precision/recall/F1 for *that specific* threshold choice.
# The mask + metrics are ephemeral â€” recomputed every call. See
# Roadmap step 14.5 for the design discussion.


class AnomalyDetectionPreviewRequest(BaseModel):
    """Body for the Apply round-trip."""

    threshold: float = Field(
        ...,
        description=(
            "Slider value. Interpreted as a percentile (0..100) when "
            "threshold_mode='percentile', or as a raw composite value "
            "in [0, 1] when threshold_mode='absolute'."
        ),
    )
    threshold_mode: str = Field(
        default="percentile",
        description="'percentile' (default) or 'absolute'.",
    )
    dilation_kernel: int = Field(
        default=0,
        ge=0,
        description=(
            "Odd integer (or 0 to disable). Side of the square structuring "
            "element used to dilate the binary anomaly mask before metrics."
        ),
    )


class AnomalyDetectionPreviewResponse(BaseModel):
    """Wire shape returned by the Apply endpoint.

    The mask PNG itself is served via a sibling endpoint that the
    frontend hits as an image URL â€” keeps JSON small and lets the
    browser cache-bust on the threshold tuple.
    """

    threshold_absolute: float
    threshold_percentile: float
    dilation_kernel: int
    n_anomalous: int
    n_kept: int
    metrics: dict | None
    # Frontend constructs a deterministic URL from these params to
    # request the rendered PNG separately.
    mask_url: str


@anomaly_threshold_router.post(
    "/{action_id}/anomaly_detection_preview",
    response_model=AnomalyDetectionPreviewResponse,
    summary="Apply a threshold to the composite score and return the binary mask + metrics",
)
def anomaly_detection_preview(
    action_id: str,
    body: AnomalyDetectionPreviewRequest,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> AnomalyDetectionPreviewResponse:
    """Interactive threshold preview for an ``anomaly_detection_prep`` action.

    The action must be in status ``needs_threshold`` and its output dir
    must contain ``composite_score.tif`` + ``summary.json``. Everything
    else (the mask, the metrics) is recomputed fresh on each call â€”
    nothing is persisted.

    Returns the rendered binary mask via a sibling URL (cached in this
    process's memory and served by ``GET .../anomaly_detection_preview_mask``).
    """
    from ._anomaly_detection_preview import (
        compute_preview,
        load_gt_mask_for_action,
    )

    action = action_or_404(action_id, db)
    if action.type != "anomaly_detection_prep":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="action_type_mismatch",
        )
    if action.status not in ("needs_threshold", "complete"):
        # We tolerate `complete` so users who later committed a
        # threshold can still re-explore alternatives in the prep
        # viewer until the commit pathway exists.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"action_not_in_needs_threshold (status={action.status!r})",
        )

    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    output_dir = (artifacts_root / output.artifact_path).resolve()
    composite_path = output_dir / "composite_score.tif"
    if not composite_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="composite_score_not_found",
        )

    # Optional GT â€” drives metrics when present.
    gt_mask = None
    annotation_id = (action.configuration or {}).get("input_annotation_id")
    composite_shape = None
    if annotation_id:
        # Resolve scene_id via the action's project â€” same scene_dir
        # convention anomaly_scoring uses for its own GT loader.
        project = db.get(Project, action.project_id)
        scene_id_for_gt = str(project.scene_id) if project else None

        # The composite raster's shape is the GT lookup constraint.
        # Read the composite header cheaply to get it without pulling
        # the whole raster into RAM here (the cache in
        # _anomaly_detection_preview handles the full read).
        import rasterio
        with rasterio.open(composite_path) as src:
            composite_shape = src.shape
        if scene_id_for_gt is not None:
            data_root = Path(settings.data_dir).resolve()
            gt_mask = load_gt_mask_for_action(
                data_root=data_root,
                scene_id=scene_id_for_gt,
                annotation_id=annotation_id,
                expected_shape=composite_shape,
            )

    try:
        result = compute_preview(
            composite_path=composite_path,
            threshold=body.threshold,
            threshold_mode=body.threshold_mode,
            dilation_kernel=body.dilation_kernel,
            gt_mask=gt_mask,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Stash the rendered PNG in the per-request cache so the sibling
    # GET endpoint can serve it without recomputing.
    _stash_preview_mask(
        action_id=str(action.id),
        params=(body.threshold, body.threshold_mode, body.dilation_kernel),
        png_bytes=result.mask_png_bytes,
    )
    # Browser-relative URL â€” the frontend's nginx proxies /api/* to
    # this api process, so we include the /api prefix so an <img src>
    # in the SPA dereferences correctly.
    mask_url = (
        f"/api/actions/{action_id}/anomaly_detection_preview_mask"
        f"?t={body.threshold}&mode={body.threshold_mode}"
        f"&dk={body.dilation_kernel}"
    )
    return AnomalyDetectionPreviewResponse(
        threshold_absolute=result.threshold_absolute,
        threshold_percentile=result.threshold_percentile,
        dilation_kernel=result.dilation_kernel,
        n_anomalous=result.n_anomalous,
        n_kept=result.n_kept,
        metrics=result.metrics,
        mask_url=mask_url,
    )


# Tiny per-process cache: the POST renders the PNG, the GET serves it
# bytewise. Keyed by (action_id, threshold, mode, kernel). LRU-capped.
_PREVIEW_MASK_CACHE_MAX = 16
_preview_mask_cache: "dict[tuple, bytes]" = {}
_preview_mask_lru: "list[tuple]" = []


def _stash_preview_mask(*, action_id: str, params: tuple, png_bytes: bytes) -> None:
    key = (action_id, *params)
    _preview_mask_cache[key] = png_bytes
    if key in _preview_mask_lru:
        _preview_mask_lru.remove(key)
    _preview_mask_lru.append(key)
    while len(_preview_mask_lru) > _PREVIEW_MASK_CACHE_MAX:
        oldest = _preview_mask_lru.pop(0)
        _preview_mask_cache.pop(oldest, None)


@anomaly_threshold_router.get(
    "/{action_id}/anomaly_detection_preview_mask",
    summary="Stream the binary anomaly mask PNG matching the last Apply call's params",
)
def anomaly_detection_preview_mask(
    action_id: str,
    t: float = Query(...),
    mode: str = Query("percentile"),
    dk: int = Query(0),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """Serve a previously-rendered binary anomaly mask PNG.

    The POST endpoint computes the PNG and stashes it; this sibling
    GET serves the bytes. If the cache lost the entry (process
    restarted, or another Apply call evicted it), the client should
    re-POST to recompute.
    """
    _action = action_or_404(action_id, db)
    key = (str(_action.id), t, mode, dk)
    png_bytes = _preview_mask_cache.get(key)
    if png_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="preview_mask_not_cached_repost_apply",
        )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# --- POST /actions/{id}/anomaly_detection_commit --------------------
#
# Lock in the user's chosen threshold + dilation on a prep action.
# Writes ``anomaly_mask.tif`` (binary geotiff) + ``metrics.json`` into
# the action's existing output directory, merges the chosen parameters
# into ``actions.configuration`` so they're a permanent property of
# the action row, marks the action_output's summary with
# ``committed: true`` so downstream pickers can filter, and flips the
# action's status from ``needs_threshold`` (or already-``complete`` on
# a re-commit) to ``complete``.
#
# Apply is unaffected â€” a committed prep still accepts further Apply
# calls. Users can re-explore and re-commit at any time; re-commit
# overwrites the canonical mask + metrics + configuration.


class AnomalyDetectionCommitRequest(BaseModel):
    threshold: float = Field(
        ...,
        description=(
            "Slider value to lock in. Same semantics as the preview "
            "endpoint: 'above pX' when threshold_mode='percentile', "
            "raw composite value when threshold_mode='absolute'."
        ),
    )
    threshold_mode: str = Field(default="percentile")
    dilation_kernel: int = Field(default=0, ge=0)


class AnomalyDetectionCommitResponse(BaseModel):
    """Wire shape returned after a successful commit."""

    threshold_absolute: float
    threshold_percentile: float
    dilation_kernel: int
    n_anomalous: int
    n_kept: int
    metrics: dict | None
    # Convenience pointer to the saved binary mask raster â€” frontend
    # can use this to surface a download link.
    mask_tif_path: str


@anomaly_threshold_router.post(
    "/{action_id}/anomaly_detection_commit",
    response_model=AnomalyDetectionCommitResponse,
    summary="Lock the chosen threshold on an anomaly_detection_prep action",
)
def anomaly_detection_commit(
    action_id: str,
    body: AnomalyDetectionCommitRequest,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> AnomalyDetectionCommitResponse:
    """Commit the user's chosen threshold on a prep action.

    Re-runs the exact preview math (so the saved mask matches what the
    viewer last showed) and writes:

      - ``<output_dir>/anomaly_mask.tif``  â€” uint8 binary geotiff
      - ``<output_dir>/metrics.json``      â€” threshold + dilation +
                                             P / R / F1 + TP/FP/FN
                                             (P/R/F1 only when GT
                                             attached)

    Then mutates the action row + its output row:

      - ``action.configuration.committed_threshold|mode|dilation`` set
      - ``action_output.summary.committed = true`` + the same params
      - ``action.status = "complete"``

    Re-commit is allowed â€” overwrites the prior mask + metrics + flags.
    """
    import json as _json

    import numpy as np
    import rasterio

    from ._anomaly_detection_preview import (
        compute_preview,
        load_gt_mask_for_action,
    )

    action = action_or_404(action_id, db)
    if action.type != "anomaly_detection_prep":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="action_type_mismatch",
        )
    if action.status not in ("needs_threshold", "complete"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"action_not_committable (status={action.status!r})",
        )

    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    output_dir = (artifacts_root / output.artifact_path).resolve()
    composite_path = output_dir / "composite_score.tif"
    if not composite_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="composite_score_not_found",
        )

    # Optional GT â€” drives metrics in the commit payload the same way
    # Apply does.
    gt_mask = None
    annotation_id = (action.configuration or {}).get("input_annotation_id")
    if annotation_id:
        project = db.get(Project, action.project_id)
        scene_id_for_gt = str(project.scene_id) if project else None
        with rasterio.open(composite_path) as src:
            composite_shape = src.shape
        if scene_id_for_gt is not None:
            data_root = Path(settings.data_dir).resolve()
            gt_mask = load_gt_mask_for_action(
                data_root=data_root,
                scene_id=scene_id_for_gt,
                annotation_id=annotation_id,
                expected_shape=composite_shape,
            )

    try:
        result = compute_preview(
            composite_path=composite_path,
            threshold=body.threshold,
            threshold_mode=body.threshold_mode,
            dilation_kernel=body.dilation_kernel,
            gt_mask=gt_mask,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # --- Persist the binary mask raster -------------------------------
    # Recompute the mask at full resolution here (the preview's
    # rendered PNG was downsampled). Cheap on this same composite.
    with rasterio.open(composite_path) as src:
        composite = src.read(1).astype("float32", copy=False)
        profile = src.profile
    finite = np.isfinite(composite)
    if body.threshold_mode == "percentile":
        finite_vals = np.sort(composite[finite])
        if finite_vals.size == 0:
            absolute_threshold = float("inf")
        else:
            quantile_pct = max(0.0, min(100.0, float(body.threshold)))
            absolute_threshold = float(np.percentile(finite_vals, quantile_pct))
    else:
        absolute_threshold = float(body.threshold)

    raw_mask = (composite >= np.float32(absolute_threshold)) & finite
    if body.dilation_kernel and body.dilation_kernel > 1:
        from ._anomaly_detection_preview import _square_dilate
        raw_mask = _square_dilate(raw_mask, int(body.dilation_kernel)) & finite
    binary_mask = raw_mask.astype("uint8")

    profile.update(
        dtype="uint8",
        count=1,
        nodata=None,
        compress="deflate",
        predictor=2,
    )
    mask_tif_path_abs = output_dir / "anomaly_mask.tif"
    with rasterio.open(mask_tif_path_abs, "w", **profile) as dst:
        dst.write(binary_mask, 1)

    # --- Persist metrics.json -----------------------------------------
    metrics_payload: dict[str, Any] = {
        "threshold_absolute": result.threshold_absolute,
        "threshold_percentile": result.threshold_percentile,
        "threshold_mode": body.threshold_mode,
        "dilation_kernel": int(body.dilation_kernel),
        "n_anomalous": result.n_anomalous,
        "n_kept": result.n_kept,
        "has_gt": result.metrics is not None,
        "committed_at": datetime.utcnow().isoformat() + "Z",
    }
    if result.metrics:
        metrics_payload.update({
            "precision": result.metrics["precision"],
            "recall": result.metrics["recall"],
            "f1": result.metrics["f1"],
            "tp": result.metrics["tp"],
            "fp": result.metrics["fp"],
            "fn": result.metrics["fn"],
            "tn": result.metrics["tn"],
            "n_gt_positives": result.metrics["n_gt_positives"],
        })
    (output_dir / "metrics.json").write_text(_json.dumps(metrics_payload, indent=2))

    # --- Stamp the action + output rows -------------------------------
    cfg = dict(action.configuration or {})
    cfg["committed_threshold"] = float(body.threshold)
    cfg["committed_threshold_mode"] = body.threshold_mode
    cfg["committed_threshold_absolute"] = float(result.threshold_absolute)
    cfg["committed_dilation_kernel"] = int(body.dilation_kernel)
    action.configuration = cfg

    # Updating a JSONB column on a row that was loaded from another
    # transaction requires marking it modified explicitly when the
    # mutation is a dict-replace; SQLAlchemy's ORM picks up the
    # assignment above just fine, but the nested merge below needs a
    # flag_modified to make sure the change is written.
    from sqlalchemy.orm.attributes import flag_modified

    output_summary = dict(output.summary or {})
    output_summary["committed"] = True
    output_summary["committed_threshold"] = float(body.threshold)
    output_summary["committed_threshold_mode"] = body.threshold_mode
    output_summary["committed_threshold_absolute"] = float(result.threshold_absolute)
    output_summary["committed_dilation_kernel"] = int(body.dilation_kernel)
    if result.metrics:
        output_summary["committed_metrics"] = {
            "precision": result.metrics["precision"],
            "recall": result.metrics["recall"],
            "f1": result.metrics["f1"],
        }
    output.summary = output_summary
    flag_modified(output, "summary")

    action.status = "complete"
    action.completed_at = datetime.utcnow()
    db.commit()

    return AnomalyDetectionCommitResponse(
        threshold_absolute=float(result.threshold_absolute),
        threshold_percentile=float(result.threshold_percentile),
        dilation_kernel=int(body.dilation_kernel),
        n_anomalous=int(result.n_anomalous),
        n_kept=int(result.n_kept),
        metrics=result.metrics,
        mask_tif_path=f"/api/actions/{action_id}/files/anomaly_mask.tif",
    )


