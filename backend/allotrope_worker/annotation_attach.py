"""annotation_attach worker handler.

Inbound payload (from `POST /scenes/{id}/annotations`):
    {
      "scene_id":            "scene_<uuid>",
      "annotation_type":     "<KIND>",
      "name":                "...",
      "description":         "..." | null,
      "filename":            "<sanitised>",
      "staging_dir":         "staging/<job_id>",
      "uploaded_by_user_id": "<user_uuid>",
      "uploaded_at":         "<iso-8601>",
    }

Flow (per-type behaviour delegated to the annotation_types registry):
    1. Resolve scene + look up the type spec.
    2. spec.materialise(staging, final_dir) → moves/transforms the file.
    3. spec.render_overlay(final, dest_png) → optional RGBA layer.
    4. spec.extract_metadata(final) → JSONB extras.
    5. INSERT Annotation + UPDATE scenes.has_annotations=true atomically.
    6. Return ("annotation", annotation_id).

Failure path: anything before the INSERT raises → runner marks job
failed, file moves rolled back, no Annotation row inserted, scene
unchanged.

Registry contract: backend/allotrope_worker/annotation_types/__init__.py
Sequence diagram:   final design/diagrams/annotation-attach.drawio (Step 9e)
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from allotrope.config import settings
from allotrope.models import Annotation, Job, Scene

from allotrope.annotation_types import get_spec

logger = logging.getLogger("allotrope.worker.annotation_attach")


def _strip_prefix(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    return value[len(expected):] if value.startswith(expected) else value


def handle_annotation_attach(
    session: Session, job: Job
) -> tuple[str, uuid.UUID]:
    """Worker handler for `annotation_attach` jobs."""
    payload = job.payload or {}
    scene_id_wire = payload.get("scene_id")
    ann_type = payload.get("annotation_type")
    name = payload.get("name")
    description = payload.get("description")
    filename = payload.get("filename")
    staging_rel = payload.get("staging_dir")
    uploaded_by_user_id = payload.get("uploaded_by_user_id")

    for label, val in [
        ("scene_id", scene_id_wire),
        ("annotation_type", ann_type),
        ("name", name),
        ("filename", filename),
        ("staging_dir", staging_rel),
    ]:
        if not val:
            raise ValueError(f"payload missing required field: {label}")

    # Type spec — looked up here rather than in the api so an unknown
    # type at api time (which shouldn't happen since the api also
    # validates) lands as a job-level failure with a clear reason.
    try:
        spec = get_spec(ann_type)
    except KeyError as exc:
        raise ValueError(f"no registered annotation type for {ann_type!r}") from exc

    # Resolve Scene.
    try:
        scene_uuid = uuid.UUID(_strip_prefix(scene_id_wire, "scene"))
    except ValueError as exc:
        raise ValueError(f"bad scene_id in payload: {scene_id_wire!r}") from exc
    scene = session.get(Scene, scene_uuid)
    if scene is None:
        raise FileNotFoundError(f"scene not found: {scene_id_wire}")

    data_root = Path(settings.data_dir)
    artifacts_root = Path(settings.artifacts_dir)

    staging_root = data_root / staging_rel
    staging_file = staging_root / filename
    if not staging_file.is_file():
        raise FileNotFoundError(f"staged file missing: {staging_file}")

    # Mint annotation_id and plan final paths.
    ann_id = uuid.uuid4()
    final_dir_rel = f"scenes/{scene.id}/annotations/{ann_id}"
    final_dir_abs = data_root / final_dir_rel

    # 1. Materialise (type-specific move/transform).
    final_file_abs = spec.materialise(staging_file, final_dir_abs)
    final_file_rel = f"{final_dir_rel}/{final_file_abs.name}"

    # Best-effort: clean the (now-empty) staging dir.
    shutil.rmtree(staging_root, ignore_errors=True)

    # 2. Overlay (optional per type) + 3. metadata extraction.
    overlay_rel = f"scenes/{scene.id}/annotations/{ann_id}/overlay.png"
    overlay_abs = artifacts_root / overlay_rel
    has_overlay = False
    extras: dict[str, object] = {}
    try:
        has_overlay = bool(spec.render_overlay(final_file_abs, overlay_abs))
        extras = dict(spec.extract_metadata(final_file_abs) or {})
    except Exception:
        logger.exception(
            "type-specific render/metadata failed for annotation %s; rolling back move",
            ann_id,
        )
        if final_file_abs.exists():
            final_file_abs.unlink(missing_ok=True)
        if final_dir_abs.exists():
            shutil.rmtree(final_dir_abs, ignore_errors=True)
        raise

    # Always include the file path in metadata so the UI can show it.
    extras.setdefault("filename", final_file_abs.name)
    if has_overlay:
        extras["overlay_path"] = overlay_rel

    # 4. INSERT Annotation row + flip Scene.has_annotations.
    created_by: uuid.UUID | None
    try:
        created_by = (
            uuid.UUID(uploaded_by_user_id) if uploaded_by_user_id else None
        )
    except (TypeError, ValueError):
        created_by = None

    annotation = Annotation(
        id=ann_id,
        scene_id=scene.id,
        type=ann_type,
        name=name,
        description=description,
        file_path=final_file_rel,
        extra_metadata=extras,
        created_by_user_id=created_by,
    )
    session.add(annotation)
    scene.has_annotations = True
    session.flush()

    logger.info(
        "annotation_attach ok: type=%s annotation=%s scene=%s file=%s overlay=%s",
        ann_type, ann_id, scene.id, final_file_abs.name, has_overlay,
    )
    return ("annotation", ann_id)
