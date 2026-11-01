"""
The story GIF of a spectral_library_match action: GET /actions/{id}/story.gif.

Walks the action's lineage (match -> prep -> scoring -> segmentation -> band filter, through the
``input_*_output_id`` fields of each action's configuration) and hands the output directories to
app/story/frames.py, which draws the frames; app/story/render.py makes the GIF. Design:
docs/lld/story-gif.md.

Why it exists: the committee demo (2026-11-04) shows how a scene was narrowed down to the selected
anomalies as one file. The selection (``classes``, ``confident_only``, ``manmade_p``) has the meaning of
the class-driven export's and goes through the same helper, app/spectral_match/class_selection.py.

NOT a submission artefact. Frame 4 is a heatmap and the rules disqualify heatmap outputs, so the GIF is
never added to an export zip. For the same reason it is cached BESIDE the action's output directory
(``actions/<id>/story/``), not inside it: the project export copies every file under ``output/``.

Runs synchronously in the api process; numpy / PIL / rasterio / pyarrow are imported inside the handler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, ActionOutput, Project, Scene
from ._action_common import action_or_404, output_for_action
from .deps import current_user_claims

logger = logging.getLogger("allotrope.api.action_story")

action_story_router = APIRouter(prefix="/actions", tags=["actions"])

# Bump when the frames or the renderer change, so cached GIFs are not served stale.
STORY_VERSION = "1"
DEFAULT_CLASSES = "manmade"
CACHE_DIR = "story"


@action_story_router.get(
    "/{action_id}/story.gif",
    summary="Animated GIF of how a spectral_library_match action narrowed its scene down (presentation only)",
    response_class=Response,
)
def action_story_gif(
    action_id: str,
    classes: str | None = Query(None, description="Comma list, as the class-driven export; default manmade"),
    confident_only: bool | None = Query(None, description="Drop tentative man-made objects; default false"),
    manmade_p: float | None = Query(None, ge=0.0, le=1.0, description="Man-made cut; default the action's 0.5"),
    size: int = Query(960, ge=320, le=1600, description="Longer side of the map, px"),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """The story as image/gif, cached per parameter set. 422 for another action type, for a selection
    asked of an action without spectral-namer results, or for a selection that cannot be made."""
    from app.spectral_match.class_selection import SelectionError, parse_classes
    from app.story.chain import SelectionParams, StoryChain, StoryError, has_namer
    from app.story.frames import build_story
    from app.story.render import render_gif

    action = action_or_404(action_id, db)
    check_story_action(action.type)
    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="output_not_ready")
    project = db.get(Project, action.project_id)
    scene = db.get(Scene, project.scene_id) if project and project.scene_id else None
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_bound")
    chain = StoryChain(sensor=scene.sensor_type, scene_name=scene.name, acquired=scene.acquisition_at,
                       scene_bands=scene.band_count, match=_step(output, action.configuration))
    _walk_upstream(chain, action, db)
    try:
        explicit = classes is not None or confident_only is not None or manmade_p is not None
        check_selection_possible(has_namer(chain), explicit)
        sel = SelectionParams(tokens=parse_classes(classes or DEFAULT_CLASSES),
                              confident_only=bool(confident_only), manmade_p=manmade_p)
        cache = _cache_path(output, chain, sel, size)
        if cache.is_file():
            data = cache.read_bytes()
        else:
            frames, title = build_story(chain, sel, size)
            data = render_gif(frames, title=title, size=size)
            _write_cache(cache, data)
    except (SelectionError, StoryError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    name = f"allotrope_story_{scene.id.hex[:8]}_{action.id.hex[:8]}.gif"
    return Response(content=data, media_type="image/gif",
                    headers={"Content-Disposition": f'attachment; filename="{name}"',
                             "Cache-Control": "private, max-age=300"})


def check_story_action(action_type: str) -> None:
    """422 unless the action is a spectral_library_match (the story ends at its answer)."""
    if action_type != "spectral_library_match":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="the story GIF is made from a spectral_library_match action; "
                                   f"this one is {action_type}")


def check_selection_possible(namer_present: bool, explicit_selection: bool) -> None:
    """422 when a class selection is asked of an action without spectral-namer results. Without one, and
    with no selection asked, the story is told up to the threshold frame."""
    if explicit_selection and not namer_present:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="this action has no spectral-namer results (it ran without the namer, "
                                   "or the namer failed), so its story cannot end in a class selection; "
                                   "ask without classes / confident_only / manmade_p for the story up to "
                                   "the threshold")


# --- lineage -----------------------------------------------------------------------


def _walk_upstream(chain, match: Action, db: Session) -> None:
    """Fill chain.prep / scoring / segmentation / band_filter from the configs' input ids. A link that
    does not resolve (deleted, another project, no output yet) leaves that step None."""
    prep = _resolve(match.configuration.get("input_anomaly_detection_output_id"), match.project_id, db)
    chain.prep = prep
    scoring = _resolve(prep.config.get("input_anomaly_scoring_output_id"), match.project_id, db) if prep else None
    chain.scoring = scoring
    if scoring is None:
        return
    chain.segmentation = _resolve(scoring.config.get("input_scene_segmentation_output_id"), match.project_id, db)
    bf_id = scoring.config.get("input_band_filter_output_id")
    if not bf_id and chain.segmentation is not None:
        bf_id = chain.segmentation.config.get("input_band_filter_output_id")
    chain.band_filter = _resolve(bf_id, match.project_id, db)


def _resolve(output_wire: str | None, project_id: uuid.UUID, db: Session):
    if not output_wire or not str(output_wire).startswith("output_"):
        return None
    try:
        out = db.get(ActionOutput, uuid.UUID(output_wire[len("output_"):]))
    except ValueError:
        return None
    if out is None:
        return None
    act = db.get(Action, out.action_id)
    if act is None or act.project_id != project_id:
        return None
    step = _step(out, act.configuration)
    return step if step.dir.is_dir() else None


def _step(output: ActionOutput, config: dict | None):
    from app.story.chain import Step

    return Step(dir=Path(settings.artifacts_dir) / output.artifact_path, config=dict(config or {}),
                summary=dict(output.summary or {}))


# --- cache -------------------------------------------------------------------------


def _cache_path(output: ActionOutput, chain, sel, size: int) -> Path:
    """actions/<id>/story/story_<key>.gif, keyed by the parameters and the files they are drawn from."""
    stamps = []
    for step in (chain.match, chain.prep):
        for name in ("namer_objects.parquet", "anomaly_mask.tif", "summary.json"):
            p = step.dir / name if step else None
            if p is not None and p.is_file():
                stamps.append(f"{name}:{p.stat().st_mtime_ns}")
    key = json.dumps([STORY_VERSION, list(sel.tokens), sel.confident_only, sel.manmade_p, size, stamps])
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return chain.match.dir.parent / CACHE_DIR / f"story_{digest}.gif"


def _write_cache(path: Path, data: bytes) -> None:
    """Best effort: a read-only volume costs a re-render next time, not the response."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("story cache not written (%s): %s", path, exc)
