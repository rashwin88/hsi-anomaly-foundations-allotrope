"""
Serving Action artifacts to the browser.

Split out of actions.py. Three endpoints that all answer the same question -
"give me bytes an Action produced":

    GET /actions/{id}/files/{filename}      a named artifact in the output dir
    GET /actions/{id}/output/{relpath:path} anything nested beneath it
    GET /action-outputs/{id}                the ActionOutput row itself

Every path is resolved under the Action's own output directory and checked
before serving. Artifact paths are stored RELATIVE in the database and joined
to the artifacts volume at read time, so a stored path can never escape it.

These carry the browser's whole view of a finished Action: the spectral-match
viewer alone pulls matches.parquet plus two .npz snapshots through here so it
can serve hover probes client-side without an api round-trip per pixel.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Action, ActionOutput
from ._action_common import (
    ActionOutputPublic,
    action_or_404,
    output_for_action,
    output_to_wire,
)
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.action_files")

# Same prefixes as their counterparts in actions.py, so mounted paths are
# byte-identical to before the split.
action_files_router = APIRouter(prefix="/actions", tags=["actions"])
outputs_router = APIRouter(prefix="/action-outputs", tags=["actions"])

# --- GET /actions/{id}/files/{filename} ------------------------------


_FILE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".pkl": "application/octet-stream",
}


@action_files_router.get(
    "/{action_id}/files/{filename}",
    status_code=status.HTTP_200_OK,
    summary="Stream an artifact file from the Action's output directory",
    response_class=FileResponse,
)
def get_action_file(
    action_id: str,
    filename: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream a file from `<artifacts>/<action_output.artifact_path>/<filename>`.

    Used by the Action card / Output viewer in the workspace to render
    `preview.png`, fetch `diagnostics.json`, and (later) ship raw rasters
    to client-side tooling.

    - 404 `action_not_found` for unknown action_id.
    - 404 `output_not_ready` when the action has no ActionOutput yet.
    - 404 `file_not_found` when the artifact dir exists but the named
      file is missing.
    - 422 `invalid_filename` for traversal attempts (`..`, slashes,
      empty names) â€” basename-only access is enforced.

    Path-traversal defence is two-fold: filename is rejected if it
    contains `/` or `\\` or `..` segments; the resolved absolute path
    is then asserted to live inside the artifacts root.
    """
    action = action_or_404(action_id, db)

    # Filename must be a single basename â€” no traversal, no nesting.
    if not filename or "/" in filename or "\\" in filename or filename in ("..", "."):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_filename",
        )
    if ".." in filename.split("."):
        # paranoid catch â€” shouldn't trigger because of the slash check above
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_filename",
        )

    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    full = (artifacts_root / output.artifact_path / filename).resolve()
    # Defence in depth â€” confine to artifacts root.
    try:
        full.relative_to(artifacts_root)
    except ValueError:
        logger.warning(
            "action %s output filename=%r escapes artifacts root",
            action.id,
            filename,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )
    if not full.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )

    media = _FILE_MEDIA_TYPES.get(full.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(full),
        media_type=media,
        # Action artifacts are write-once + immutable per Â§ 5.6 â€” cache aggressively.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --- GET /actions/{id}/output/{relpath:path} -------------------------
#
# Some action types (anomaly_scoring) write nested per-model artifact
# directories. The single-basename `/files/{filename}` route can't
# reach them. This sibling allows nested relpaths inside the same
# artifact root, with the same traversal defence.


@action_files_router.get(
    "/{action_id}/output/{relpath:path}",
    status_code=status.HTTP_200_OK,
    summary="Stream a nested artifact file from the Action's output directory",
    response_class=FileResponse,
)
def get_action_output_file(
    action_id: str,
    relpath: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream a nested file under `<artifacts>/<action_output.artifact_path>/<relpath>`.

    Used by the anomaly_scoring viewer to fetch
    `models/<codename>/anomaly_score.png` etc. without flattening the
    on-disk layout.
    """
    action = action_or_404(action_id, db)

    if not relpath or relpath.startswith("/") or "\\" in relpath:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_relpath",
        )
    parts = relpath.split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_relpath",
        )

    output = output_for_action(action.id, db)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_ready",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    full = (artifacts_root / output.artifact_path / relpath).resolve()
    try:
        full.relative_to(artifacts_root)
    except ValueError:
        logger.warning(
            "action %s output relpath=%r escapes artifacts root",
            action.id,
            relpath,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )
    if not full.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="file_not_found"
        )

    media = _FILE_MEDIA_TYPES.get(full.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(full),
        media_type=media,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --- GET /action-outputs/{id} ---------------------------------------
#
# Tiny lookup endpoint so the frontend can resolve an
# ``output_<uuid>`` to its producing ``action_<uuid>``. Used by the
# NewActionDialog when wiring an anomaly_detection_prep â€” once the user
# picks an upstream anomaly_scoring Output we need to fetch that
# Action's summary.json to discover which algorithms ran, so the
# dialog can render one weight input per algorithm.


@outputs_router.get(
    "/{output_id}",
    response_model=ActionOutputPublic,
    summary="Look up an ActionOutput by wire-format id",
)
def get_action_output(
    output_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionOutputPublic:
    """Resolve ``output_<uuid>`` â†’ its ActionOutput row (incl. action_id)."""
    raw_id = parse_prefixed_id("output", output_id)
    output = db.get(ActionOutput, raw_id)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="output_not_found",
        )
    return output_to_wire(output)

