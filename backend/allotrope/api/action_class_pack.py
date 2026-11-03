"""
The class-driven export of a spectral_library_match action, and the counts its button shows.

    POST /actions/{id}/export?classes=...&confident_only=...&manmade_p=...   (dispatched from action_export.py)
    GET  /actions/{id}/spectral_library_match/selection_counts?classes=...&confident_only=...&manmade_p=...

The match screen lets the analyst tick classes (natural, man-made, or material classes such as oil_fuel), keep
confident objects only, and move the man-made cut; the export is then exactly that selection
(app/spectral_match/class_pack.py), and selection_counts gives "Export oil + plastic (64 objects)" without building
the zip. Both use app/spectral_match/class_selection.py, so they cannot disagree.

What this module adds over the pack builder is what needs the database: the scene georef (via action_export's
``_scene_georef``) and the MANIFEST "chain" - the band filter, segmentation, scoring and prep actions behind the
match, found by following each action's ``input_*_output_id``. Design: docs/proposals/class-driven-flow.md.

Tokens are checked against namer/classes.py MANMADE_CLASSES at request time; an unknown one, or an action without
namer results, is a 422 whose detail says why.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, ActionOutput, Scene
from .deps import current_user_claims

logger = logging.getLogger("allotrope.api.action_class_pack")

class_pack_router = APIRouter(prefix="/actions", tags=["actions"])


def _422(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def parse_tokens(classes: str) -> tuple[str, ...]:
    """The comma list as tokens, or a 422 naming the unknown ones."""
    from app.spectral_match.class_selection import SelectionError, parse_classes

    try:
        return parse_classes(classes)
    except SelectionError as exc:
        raise _422(str(exc)) from exc


def class_pack_response(action: Action, scene: Scene, artifact_dir: Path, db: Session, *, transform, crs,
                        classes: str, confident_only: bool, manmade_p: float | None,
                        software_version: str, georef_note: str | None = None) -> StreamingResponse:
    """The pack as a streamed zip. 422 for an unknown token or an action without namer results."""
    from app.spectral_match.class_pack import ClassPackSpec, build_class_pack
    from app.spectral_match.class_selection import SelectionError

    if action.type != "spectral_library_match":
        raise _422("classes= applies to spectral_library_match exports only")
    spec = ClassPackSpec(
        action_id=f"action_{action.id}", scene_id=f"scene_{scene.id}", sensor_type=scene.sensor_type,
        artifact_dir=artifact_dir, software_version=software_version, crs=crs, transform=transform,
        tokens=parse_tokens(classes), confident_only=confident_only, manmade_p=manmade_p,
        chain=resolve_chain(action, scene, db), georef_note=georef_note)
    try:
        blob, name, sel = build_class_pack(spec)
    except SelectionError as exc:
        raise _422(str(exc)) from exc
    logger.info("class pack %s: %d objects, %d pixels", name, sel.n_objects, sel.n_pixels)
    return StreamingResponse(io.BytesIO(blob), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{name}"', "Content-Length": str(len(blob))})


class SelectionCounts(BaseModel):
    objects: int
    pixels: int
    per_class: dict[str, int]
    manmade_p: float
    slug: str


@class_pack_router.get(
    "/{action_id}/spectral_library_match/selection_counts",
    response_model=SelectionCounts,
    summary="Objects and pixels a class-driven export of a spectral_library_match Action would contain",
)
def selection_counts(
    action_id: str,
    classes: str = Query(..., description="Comma list: natural, manmade, or man-made material class keys."),
    confident_only: bool = Query(False),
    manmade_p: float | None = Query(None, ge=0.0, le=1.0, description="Man-made cut; default the action's 0.5."),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> SelectionCounts:
    from app.spectral_match.class_pack import namer_legend
    from app.spectral_match.class_selection import SelectionError, select

    from .action_export import _export_context

    action, _output, _scene, artifact_dir = _export_context(action_id, db)
    if action.type != "spectral_library_match":
        raise _422("selection_counts applies to spectral_library_match actions only")
    tokens = parse_tokens(classes)
    try:
        sel = select(artifact_dir, tokens, confident_only=confident_only, manmade_p=manmade_p,
                     confident_p=float(namer_legend(artifact_dir).get("confident_p", 0.55)))
    except SelectionError as exc:
        raise _422(str(exc)) from exc
    return SelectionCounts(**sel.counts(), manmade_p=sel.manmade_p, slug=sel.slug())


# --- the chain behind the match --------------------------------------------


def resolve_chain(action: Action, scene: Scene, db: Session) -> dict[str, Any]:
    """MANIFEST "chain": scene -> band filter -> segmentation -> scoring -> prep -> match, as recorded by each
    action's configuration and summary. A missing link is null, never an error (older actions)."""
    match_out = _output_of(db, action.id)
    prep_action, prep_out = _upstream(db, action.configuration.get("input_anomaly_detection_output_id"))
    score_action, score_out = _upstream(
        db, (prep_action.configuration if prep_action else {}).get("input_anomaly_scoring_output_id"))
    score_cfg = score_action.configuration if score_action else {}
    seg_action, seg_out = _upstream(db, score_cfg.get("input_scene_segmentation_output_id"))
    bf_action, bf_out = _upstream(db, score_cfg.get("input_band_filter_output_id"))
    return {
        "scene": {"id": f"scene_{scene.id}", "sensor": scene.sensor_type, "name": scene.name,
                  "sensor_scene_id": scene.sensor_scene_id, "file": scene.raw_path},
        "band_filter": _band_filter(bf_action, bf_out),
        "segmentation": _segmentation(seg_action, seg_out, score_cfg),
        "scoring": _scoring(score_action, score_out),
        "prep": _prep(prep_action, prep_out),
        "match": _match(action, match_out),
    }


def _output_of(db: Session, action_id: uuid.UUID) -> ActionOutput | None:
    from ._action_common import output_for_action

    return output_for_action(action_id, db)


def _upstream(db: Session, output_wire: str | None) -> tuple[Action | None, ActionOutput | None]:
    if not output_wire or not output_wire.startswith("output_"):
        return None, None
    try:
        out = db.get(ActionOutput, uuid.UUID(output_wire[len("output_"):]))
    except ValueError:
        return None, None
    return (db.get(Action, out.action_id), out) if out else (None, None)


def _summary(out: ActionOutput | None) -> dict:
    return dict(out.summary or {}) if out else {}


def _band_filter(a: Action | None, out: ActionOutput | None) -> dict | None:
    if a is None:
        return None
    s = _summary(out)
    return {"action_id": f"action_{a.id}", "bands_kept": s.get("band_count"),
            "wavelengths_nm": [s.get("wavelengths_min_nm"), s.get("wavelengths_max_nm")],
            "exclusion_ranges_nm": s.get("exclusion_ranges_nm"), "edge_bands_to_trim": s.get("edge_bands_to_trim"),
            "common_165_grid": s.get("common_wavelength_grid_applied")}


def _segmentation(a: Action | None, out: ActionOutput | None, score_cfg: dict) -> dict | None:
    if a is None:
        return None
    s, cfg = _summary(out), a.configuration or {}
    return {"action_id": f"action_{a.id}", "method": s.get("method", cfg.get("method", "indices")),
            "model": s.get("model_codename"),
            "classes_left_out_at_segmentation": s.get("exclude_classes", cfg.get("exclude_classes")),
            "classes_left_out_at_scoring": score_cfg.get("exclude_classes"),
            "thin_cloud_is_cloud": s.get("thin_cloud_is_cloud"),
            "class_pct_of_valid": s.get("class_pct_of_valid")}


def _scoring(a: Action | None, out: ActionOutput | None) -> dict | None:
    if a is None:
        return None
    s, cfg = _summary(out), a.configuration or {}
    return {"action_id": f"action_{a.id}", "models": s.get("model_codenames", cfg.get("model_codenames")),
            "methods": {m.get("codename"): m.get("method") for m in s.get("per_model", [])},
            "exclude_classes": cfg.get("exclude_classes"), "keep_source": s.get("keep_source"),
            "kept_pct": s.get("kept_pct")}


def _prep(a: Action | None, out: ActionOutput | None) -> dict | None:
    if a is None:
        return None
    cfg = a.configuration or {}
    metrics = {}
    if out is not None:
        path = Path(settings.artifacts_dir) / out.artifact_path / "metrics.json"
        if path.is_file():
            metrics = json.loads(path.read_text())
    return {"action_id": f"action_{a.id}", "threshold_mode": cfg.get("committed_threshold_mode"),
            "threshold": cfg.get("committed_threshold"),
            "threshold_absolute": cfg.get("committed_threshold_absolute"),
            "dilation_kernel": cfg.get("committed_dilation_kernel"),
            "algorithm_weights": cfg.get("algorithm_weights"),
            "committed_pixels": metrics.get("n_anomalous"), "committed_at": metrics.get("committed_at")}


def _match(a: Action, out: ActionOutput | None) -> dict:
    s, cfg = _summary(out), a.configuration or {}
    namer = s.get("namer") or {}
    return {"action_id": f"action_{a.id}", "mode": cfg.get("mode"), "namer_status": namer.get("status"),
            "namer_codename": namer.get("codename"), "namer_checkpoint": namer.get("checkpoint"),
            "namer_version": namer.get("version"), "namer_epoch": namer.get("epoch"),
            "confident_p": namer.get("confident_p", cfg.get("namer_confident_p")),
            "n_objects": namer.get("n_objects"), "n_pixels": namer.get("n_pixels")}
