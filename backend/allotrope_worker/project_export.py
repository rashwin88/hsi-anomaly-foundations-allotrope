"""Worker handler for `project_export` jobs (Step 17).

Bundles a Project's current artifact state into a single zip under
`allotrope_artifacts/projects/<pid>/exports/<eid>/` and INSERTs the
Export row only after the zip is closed and sized — Option B
(spec § 5.12) keeps the row's existence as the success signal.

Bundle layout (zip):
    manifest.json                       — top-level metadata + index
    result.json                         — same payload as GET /projects/{id}/result
    scene/metadata.json                 — scene row snapshot
    scene/thumbnail.png                 — copied from artifacts/scenes/<id>/
    actions/<action_id>/output/...      — every completed Action's artifact tree
    visualizations/<viz_id>/image.png   — every saved Visualization image
    notes/<note_id>.md                  — markdown body of each Note
    annotations/<ann_id>/*              — overlays + per-annotation metadata

This makes the bundle self-describing: a recipient can read manifest.json
and walk the tree without having to query the api. Notes and
annotations skip references that no longer resolve (cascade has fired).

Sequence diagram: final design/diagrams/project-export.drawio
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from allotrope.config import settings
from allotrope.models import (
    Action,
    ActionOutput,
    Annotation,
    Export,
    Job,
    Note,
    NoteReference,
    Project,
    Scene,
    Visualization,
)

logger = logging.getLogger("allotrope.worker.project_export")


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d is not None else None


def _walk_files(base: Path) -> list[Path]:
    """Sorted list of files under `base` (rglob), no dirs. Stable order
    matters for reproducible zip contents."""
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def _add_dir_to_zip(
    zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str
) -> int:
    """Add every file under `src_dir` to the zip under `arc_prefix`.

    Returns the count of files written. Skips silently when the source
    directory is absent (e.g. an Action whose output dir got removed).
    """
    if not src_dir.is_dir():
        return 0
    written = 0
    for f in _walk_files(src_dir):
        rel = f.relative_to(src_dir)
        zf.write(f, arcname=f"{arc_prefix}/{rel.as_posix()}")
        written += 1
    return written


def _build_result_dict(
    project: Project,
    scene: Scene,
    actions_with_outputs: list[tuple[Action, ActionOutput | None]],
    viz_count: int,
    note_count: int,
    annotation_count: int,
) -> dict:
    """Inline equivalent of GET /projects/{id}/result — without a session
    roundtrip from the api so the worker stays self-contained."""
    last_completed: datetime | None = None
    for a, _ in actions_with_outputs:
        if a.status == "complete" and a.completed_at is not None:
            if last_completed is None or a.completed_at > last_completed:
                last_completed = a.completed_at
    return {
        "project": {
            "id": f"project_{project.id}",
            "name": project.name,
            "description": project.description,
            "created_at": _iso(project.created_at),
            "scene_id": f"scene_{scene.id}",
            "scene_name": scene.name,
            "scene_sensor_type": scene.sensor_type,
        },
        "actions": [
            {
                "id": f"action_{a.id}",
                "type": a.type,
                "status": a.status,
                "started_at": _iso(a.started_at),
                "completed_at": _iso(a.completed_at),
                "failure_reason": a.failure_reason,
                "output_id": (f"output_{o.id}" if o is not None else None),
                "summary": (o.summary if o is not None else None),
            }
            for a, o in actions_with_outputs
        ],
        "visualization_count": viz_count,
        "note_count": note_count,
        "annotation_count": annotation_count,
        "last_action_completed_at": _iso(last_completed),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def handle_project_export(
    session: Session, job: Job
) -> tuple[str | None, uuid.UUID | None]:
    """Bundle the project's Result state into a zip and INSERT Export.

    Returns (target_kind='export', target_id=<export.id>) so the runner
    can wire jobs.target_id to the produced row.
    """
    if job.project_id is None:
        raise ValueError("project_export job has no project_id")
    project = session.get(Project, job.project_id)
    if project is None:
        raise ValueError(f"project_export: project {job.project_id} not found")
    scene = session.get(Scene, project.scene_id)
    if scene is None:
        raise ValueError(
            f"project_export: scene {project.scene_id} not found"
        )

    # Pull every entity we need in one pass. Keep this transaction read-
    # only until the very end so the bundle is a coherent snapshot.
    actions = list(
        session.scalars(
            select(Action)
            .where(Action.project_id == project.id)
            .order_by(Action.created_at.asc())
        ).all()
    )
    outputs_by_action: dict[uuid.UUID, ActionOutput] = {}
    for o in session.scalars(
        select(ActionOutput).where(
            ActionOutput.action_id.in_([a.id for a in actions] or [uuid.uuid4()])
        )
    ).all():
        outputs_by_action[o.action_id] = o
    actions_with_outputs = [(a, outputs_by_action.get(a.id)) for a in actions]

    visualizations = list(
        session.scalars(
            select(Visualization).where(
                Visualization.project_id == project.id
            )
        ).all()
    )
    notes = list(
        session.scalars(
            select(Note)
            .where(Note.project_id == project.id)
            .order_by(Note.created_at.asc())
        ).all()
    )
    annotations = list(
        session.scalars(
            select(Annotation).where(Annotation.scene_id == project.scene_id)
        ).all()
    )

    export_id = uuid.uuid4()
    snapshot_at = datetime.now(timezone.utc)
    rel_dir = Path("projects") / str(project.id) / "exports" / str(export_id)
    abs_dir = Path(settings.artifacts_dir) / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    bundle_filename = (
        f"allotrope_{project.name.replace(' ', '_')}_{export_id}.zip"
    )
    # Sanitise — strip path-traversal characters defensively even though
    # project.name is api-validated.
    bundle_filename = bundle_filename.replace("/", "_").replace("..", "_")
    bundle_rel = rel_dir / bundle_filename
    bundle_abs = abs_dir / bundle_filename

    artifacts_root = Path(settings.artifacts_dir)

    # Manifest accumulates as we walk — kept inline so we don't have to
    # walk a second time.
    manifest = {
        "schema_version": 1,
        "exported_at": snapshot_at.isoformat(),
        "project_id": f"project_{project.id}",
        "project_name": project.name,
        "scene_id": f"scene_{scene.id}",
        "scene_sensor_type": scene.sensor_type,
        "entries": {
            "actions": [],
            "visualizations": [],
            "notes": [],
            "annotations": [],
        },
    }
    result_payload = _build_result_dict(
        project=project,
        scene=scene,
        actions_with_outputs=actions_with_outputs,
        viz_count=len(visualizations),
        note_count=len(notes),
        annotation_count=len(annotations),
    )

    try:
        with zipfile.ZipFile(
            bundle_abs,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as zf:
            # --- scene metadata + thumbnail -----------------------------
            scene_meta = {
                "id": f"scene_{scene.id}",
                "name": scene.name,
                "sensor_type": scene.sensor_type,
                "acquisition_at": _iso(scene.acquisition_at),
                "band_count": scene.band_count,
                "bbox": {
                    "min_lon": scene.bbox_min_lon,
                    "min_lat": scene.bbox_min_lat,
                    "max_lon": scene.bbox_max_lon,
                    "max_lat": scene.bbox_max_lat,
                },
            }
            zf.writestr(
                "scene/metadata.json",
                json.dumps(scene_meta, indent=2, default=str),
            )
            if scene.thumbnail_path:
                thumb_abs = artifacts_root / scene.thumbnail_path
                if thumb_abs.is_file():
                    zf.write(thumb_abs, arcname="scene/thumbnail.png")

            # --- actions + outputs -------------------------------------
            for a, o in actions_with_outputs:
                action_meta = {
                    "id": f"action_{a.id}",
                    "type": a.type,
                    "status": a.status,
                    "configuration": a.configuration,
                    "started_at": _iso(a.started_at),
                    "completed_at": _iso(a.completed_at),
                    "failure_reason": a.failure_reason,
                    "output_id": (
                        f"output_{o.id}" if o is not None else None
                    ),
                    "summary": (o.summary if o is not None else None),
                    "artifact_path": (
                        o.artifact_path if o is not None else None
                    ),
                }
                arc_prefix = f"actions/{a.id}"
                zf.writestr(
                    f"{arc_prefix}/action.json",
                    json.dumps(action_meta, indent=2, default=str),
                )
                files_written = 0
                if o is not None and o.artifact_path:
                    files_written = _add_dir_to_zip(
                        zf,
                        artifacts_root / o.artifact_path,
                        f"{arc_prefix}/output",
                    )
                manifest["entries"]["actions"].append(
                    {
                        "id": f"action_{a.id}",
                        "type": a.type,
                        "status": a.status,
                        "output_files": files_written,
                    }
                )

            # --- visualizations ----------------------------------------
            for v in visualizations:
                v_meta = {
                    "id": f"viz_{v.id}",
                    "source_kind": v.source_kind,
                    "source_scene_id": (
                        f"scene_{v.source_scene_id}"
                        if v.source_scene_id
                        else None
                    ),
                    "source_action_output_id": (
                        f"output_{v.source_action_output_id}"
                        if v.source_action_output_id
                        else None
                    ),
                    "name": v.name,
                    "description": v.description,
                    "view_state": v.view_state,
                    "created_at": _iso(v.created_at),
                }
                arc_prefix = f"visualizations/{v.id}"
                zf.writestr(
                    f"{arc_prefix}/visualization.json",
                    json.dumps(v_meta, indent=2, default=str),
                )
                img_abs = artifacts_root / v.artifact_path
                if img_abs.is_file():
                    zf.write(
                        img_abs,
                        arcname=f"{arc_prefix}/{img_abs.name}",
                    )
                manifest["entries"]["visualizations"].append(
                    {
                        "id": f"viz_{v.id}",
                        "name": v.name,
                        "source_kind": v.source_kind,
                    }
                )

            # --- notes (+ references) ----------------------------------
            for n in notes:
                refs = list(
                    session.scalars(
                        select(NoteReference).where(
                            NoteReference.note_id == n.id
                        )
                    ).all()
                )
                refs_wire = []
                for r in refs:
                    if r.ref_project_id:
                        refs_wire.append(
                            {"kind": "project", "id": f"project_{r.ref_project_id}"}
                        )
                    elif r.ref_action_id:
                        refs_wire.append(
                            {"kind": "action", "id": f"action_{r.ref_action_id}"}
                        )
                    elif r.ref_output_id:
                        refs_wire.append(
                            {"kind": "output", "id": f"output_{r.ref_output_id}"}
                        )
                    elif r.ref_viz_id:
                        refs_wire.append(
                            {"kind": "viz", "id": f"viz_{r.ref_viz_id}"}
                        )
                    elif r.ref_scene_id:
                        refs_wire.append(
                            {"kind": "scene", "id": f"scene_{r.ref_scene_id}"}
                        )
                arc_prefix = f"notes/{n.id}"
                zf.writestr(
                    f"{arc_prefix}/note.json",
                    json.dumps(
                        {
                            "id": f"note_{n.id}",
                            "created_at": _iso(n.created_at),
                            "updated_at": _iso(n.updated_at),
                            "references": refs_wire,
                        },
                        indent=2,
                        default=str,
                    ),
                )
                zf.writestr(f"{arc_prefix}/content.md", n.content)
                manifest["entries"]["notes"].append(
                    {"id": f"note_{n.id}", "references": len(refs_wire)}
                )

            # --- annotations -------------------------------------------
            for an in annotations:
                an_meta = {
                    "id": f"annotation_{an.id}",
                    "scene_id": f"scene_{an.scene_id}",
                    "annotation_type": an.annotation_type,
                    "name": an.name,
                    "description": an.description,
                    "created_at": _iso(an.created_at),
                    "raster_path": an.raster_path,
                }
                arc_prefix = f"annotations/{an.id}"
                zf.writestr(
                    f"{arc_prefix}/annotation.json",
                    json.dumps(an_meta, indent=2, default=str),
                )
                if an.raster_path:
                    raster_abs = (
                        Path(settings.data_dir) / an.raster_path
                    )
                    if raster_abs.is_file():
                        zf.write(
                            raster_abs,
                            arcname=f"{arc_prefix}/{raster_abs.name}",
                        )
                manifest["entries"]["annotations"].append(
                    {"id": f"annotation_{an.id}", "type": an.annotation_type}
                )

            # --- top-level manifest + result snapshot ------------------
            zf.writestr(
                "manifest.json", json.dumps(manifest, indent=2, default=str)
            )
            zf.writestr(
                "result.json",
                json.dumps(result_payload, indent=2, default=str),
            )

        size_bytes = bundle_abs.stat().st_size
    except Exception:
        # On any error after directory creation, scrub the partial bundle
        # so we don't litter exports/ with half-written zips.
        shutil.rmtree(abs_dir, ignore_errors=True)
        raise

    # Option B: only insert the Export row once the bundle is fully written.
    export = Export(
        id=export_id,
        project_id=project.id,
        bundle_path=str(bundle_rel),
        snapshot_at=snapshot_at,
        size_bytes=size_bytes,
        format="zip",
    )
    session.add(export)
    session.flush()
    logger.info(
        "project_export complete project=%s export=%s bytes=%d",
        project.id,
        export.id,
        size_bytes,
    )
    return ("export", export.id)
