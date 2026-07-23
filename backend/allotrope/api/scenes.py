"""Scenes endpoints (the Library).

Routes:
    GET  /scenes          list scenes (paginated, optional sensor filter)
    POST /scenes/onboard  upload files + enqueue scene_onboard job

Detail page (GET /scenes/{id}) lands in Step 8.

Sequence diagrams:
    final design/diagrams/scenes-list.drawio
    final design/diagrams/scene-onboard.drawio   (Step 7e)
"""

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Job, Project, Scene
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.scenes")

# v1 sensor types — kept here (not in a shared enum) until a second route
# needs them. List per abstractions-spec § 5.2.
_SUPPORTED_SENSORS = {"prisma", "landsat9", "enmap", "aviris_ng", "hotsat1"}

# Streaming chunk size for upload → disk. 1 MiB is a good balance between
# syscall count and per-request RAM. UploadFile is a SpooledTemporaryFile
# under the hood, so this read pattern keeps the api memory bounded even
# for large EnMAP folders.
_UPLOAD_CHUNK_BYTES = 1024 * 1024

router = APIRouter(prefix="/scenes", tags=["scenes"])


# --- Schemas ----------------------------------------------------------


class ScenePublic(BaseModel):
    """Wire shape for a Scene row.

    `id` and `created_by_user_id` are prefixed per CC-1; numeric columns
    that are NUMERIC in Postgres come back as floats (sufficient precision
    for lat/lon and percentages — graduate to Decimal-as-string only if a
    downstream consumer demands it).
    """

    id: str                           # scene_<uuid>
    sensor_type: str
    sensor_scene_id: str
    name: str
    acquisition_at: datetime | None
    processing_level: str | None
    product_type: str | None
    bbox_min_lon: float
    bbox_min_lat: float
    bbox_max_lon: float
    bbox_max_lat: float
    native_projection: str | None
    band_count: int
    cloud_cover_pct: float | None
    valid_pixel_pct: float | None
    has_annotations: bool
    metadata: dict[str, Any]          # the spec's `metadata` column (`extra_metadata` on the model)
    raw_path: str
    vendable_path: str
    thumbnail_path: str | None
    created_at: datetime
    created_by_user_id: str | None    # user_<uuid>

    @classmethod
    def from_orm_scene(cls, s: Scene) -> "ScenePublic":
        return cls(
            id=f"scene_{s.id}",
            sensor_type=s.sensor_type,
            sensor_scene_id=s.sensor_scene_id,
            name=s.name,
            acquisition_at=s.acquisition_at,
            processing_level=s.processing_level,
            product_type=s.product_type,
            bbox_min_lon=float(s.bbox_min_lon),
            bbox_min_lat=float(s.bbox_min_lat),
            bbox_max_lon=float(s.bbox_max_lon),
            bbox_max_lat=float(s.bbox_max_lat),
            native_projection=s.native_projection,
            band_count=s.band_count,
            cloud_cover_pct=(
                float(s.cloud_cover_pct) if s.cloud_cover_pct is not None else None
            ),
            valid_pixel_pct=(
                float(s.valid_pixel_pct) if s.valid_pixel_pct is not None else None
            ),
            has_annotations=s.has_annotations,
            metadata=s.extra_metadata,
            raw_path=s.raw_path,
            vendable_path=s.vendable_path,
            thumbnail_path=s.thumbnail_path,
            created_at=s.created_at,
            created_by_user_id=(
                f"user_{s.created_by_user_id}"
                if s.created_by_user_id is not None
                else None
            ),
        )


class ScenesPage(BaseModel):
    """Paginated list of scenes."""

    items: list[ScenePublic]
    total: int          # rows matching the filter (excluding limit/offset)
    limit: int
    offset: int


# --- Routes -----------------------------------------------------------


@router.get(
    "",
    response_model=ScenesPage,
    status_code=status.HTTP_200_OK,
    summary="List scenes (paginated)",
)
def list_scenes(
    limit: int = Query(50, ge=1, le=500, description="Max items per page (1-500)"),
    offset: int = Query(0, ge=0, description="Offset into the result set"),
    sensor_type: str | None = Query(
        None,
        description="Optional filter: prisma | landsat9 | enmap",
    ),
    q: str | None = Query(
        None,
        description="Optional case-insensitive substring match against scene name",
        max_length=200,
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ScenesPage:
    """Return a page of Scenes ordered by creation time (newest first).

    Library is shared (per CC-3) — any authenticated user can list. Filter
    + pagination semantics are deliberately simple offset/limit; we'll
    revisit cursor-based pagination if/when the table grows large enough
    to make offset slow.
    """
    base = select(Scene)
    if sensor_type is not None:
        base = base.where(Scene.sensor_type == sensor_type)
    if q is not None and q.strip():
        # Case-insensitive substring match. Strip user-side wildcards out
        # so a paste-y "%foo%" doesn't accidentally widen the match —
        # SQLAlchemy escapes the bound parameter against SQLi separately.
        needle = q.strip().replace("%", "").replace("_", "")
        if needle:
            base = base.where(Scene.name.ilike(f"%{needle}%"))

    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0

    rows = db.scalars(
        base.order_by(Scene.created_at.desc()).limit(limit).offset(offset)
    ).all()

    return ScenesPage(
        items=[ScenePublic.from_orm_scene(s) for s in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{scene_id}",
    response_model=ScenePublic,
    status_code=status.HTTP_200_OK,
    summary="Get one scene by id",
)
def get_scene(
    scene_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ScenePublic:
    """Return the full ScenePublic projection for a single scene.

    Library is shared (CC-3) — any authenticated user can read. Returns
    404 with `detail="scene_not_found"` for ids that parse but don't
    exist; 422 `bad_id_format` is raised by parse_prefixed_id for
    malformed input.
    """
    raw_id = parse_prefixed_id("scene", scene_id)
    scene = db.get(Scene, raw_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="scene_not_found",
        )
    return ScenePublic.from_orm_scene(scene)


# --- Onboarding -------------------------------------------------------


class OnboardAccepted(BaseModel):
    """Response shape for POST /scenes/onboard.

    The job runs asynchronously; the client polls GET /jobs/{job_id} for
    progress. On success, `target_id` on the Job will hold `scene_<uuid>`
    once the worker INSERTs the Scene row.
    """

    job_id: str         # job_<uuid>
    staged_count: int   # number of files written to the staging dir
    staged_bytes: int   # total bytes streamed (for client-side sanity)


def _safe_relative_path(filename: str | None) -> PurePosixPath:
    """Sanitize a multipart filename into a safe relative POSIX path.

    Browsers using `webkitdirectory` set the multipart filename to the
    file's path within the picked folder, e.g.
        "ENMAP_FOO/SPECTRAL_IMAGE.TIF"
    which we want to preserve so the worker can reconstruct the layout.

    Defenses:
    - empty / None → 422
    - absolute path        → reject (no leading "/")
    - any ".." component   → reject (no parent escape)
    - NUL bytes            → reject (filesystem footgun)
    - leading "./"         → strip
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="empty_filename",
        )
    if "\x00" in filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_filename",
        )
    # Browsers use forward slashes regardless of host OS — normalize first.
    cleaned = filename.replace("\\", "/")
    if cleaned.startswith("/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="absolute_path",
        )
    # Split BEFORE filtering — we want to see ".." if it's there. Empty
    # segments and "." are noise from leading "./" or doubled slashes;
    # ".." would be a parent-dir escape and is rejected outright.
    raw_parts = cleaned.split("/")
    if any(p == ".." for p in raw_parts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_filename",
        )
    parts = [p for p in raw_parts if p not in ("", ".")]
    if not parts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_filename",
        )
    return PurePosixPath(*parts)


@router.post(
    "/onboard",
    response_model=OnboardAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Stage uploaded files and enqueue a scene_onboard job",
)
def onboard_scene(
    sensor_type: str = Form(..., description="One of: prisma | landsat9 | enmap"),
    files: list[UploadFile] = File(..., description="One or more files; multi-file sensors preserve folder structure via filename"),
    claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> OnboardAccepted:
    """Accept a multipart upload, stage to disk, enqueue the worker job.

    Flow:
        1. Validate sensor_type and the file list.
        2. Generate the job_id up front (the staging dir is keyed on it).
        3. Stream each upload to allotrope_data/staging/<job_id>/<rel_path>.
        4. INSERT one `jobs` row with type='scene_onboard', payload pointing
           at the staging dir and recording the original relative paths.
        5. COMMIT, return the job id.

    Failure mid-write rmtree's the staging dir and raises — no partial
    state, no orphan job row. The worker (Step 7c) is responsible for
    moving files out of staging into their permanent location and deleting
    the staging dir on success.

    Auth: any authenticated user (Library is shared, CC-3).
    """
    # 1. validate ----------------------------------------------------------
    if sensor_type not in _SUPPORTED_SENSORS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unsupported_sensor_type",
        )
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="no_files",
        )

    # 2. job_id + staging dir ---------------------------------------------
    job_id = uuid.uuid4()
    staging_root = Path(settings.data_dir) / "staging" / str(job_id)
    try:
        staging_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        # `data_dir` should always exist (it's the volume mount point) but
        # surface a clean error if compose is misconfigured rather than
        # crashing inside FastAPI's exception layer.
        logger.exception("staging dir create failed: %s", staging_root)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="staging_unavailable",
        ) from exc

    # 3. stream uploads to disk -------------------------------------------
    staged: list[str] = []
    total_bytes = 0
    try:
        for upload in files:
            rel = _safe_relative_path(upload.filename)
            target = staging_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            # Copy the upload's underlying SpooledTemporaryFile into the
            # destination in chunks. shutil.copyfileobj uses an internal
            # 64 KiB buffer; we pass our own to keep the chunk size
            # consistent with the comment up top.
            with target.open("wb") as out:
                while True:
                    chunk = upload.file.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
                    total_bytes += len(chunk)
            staged.append(str(rel))
    except HTTPException:
        # Validation errors raised by _safe_relative_path — keep their
        # status/detail intact, but still rmtree the staging dir so a
        # rejected upload doesn't leak partial files.
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    except Exception:
        logger.exception("staging failed; cleaning up %s", staging_root)
        shutil.rmtree(staging_root, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="upload_failed",
        )

    # 4. enqueue ----------------------------------------------------------
    # `staging_dir` is the path the worker will read from. Stored as
    # POSIX-style and relative to data_dir so the same payload makes sense
    # regardless of where the volume mounts on the worker side (we currently
    # mount at the same `/data`, but keep the contract loose).
    payload: dict[str, Any] = {
        "sensor_type": sensor_type,
        "staging_dir": f"staging/{job_id}",
        "files": staged,                    # original relative paths, in upload order
        "uploaded_by_user_id": claims["sub"],
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    job = Job(
        id=job_id,
        type="scene_onboard",
        status="queued",
        payload=payload,
    )
    db.add(job)
    db.commit()

    logger.info(
        "queued scene_onboard job=%s sensor=%s files=%d bytes=%d",
        job.id,
        sensor_type,
        len(staged),
        total_bytes,
    )

    return OnboardAccepted(
        job_id=f"job_{job.id}",
        staged_count=len(staged),
        staged_bytes=total_bytes,
    )


# --- Thumbnail --------------------------------------------------------


@router.get(
    "/{scene_id}/thumbnail",
    status_code=status.HTTP_200_OK,
    summary="Stream the scene's thumbnail PNG",
    response_class=FileResponse,
)
def get_scene_thumbnail(
    scene_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream `<artifacts>/<scene.thumbnail_path>` to the client.

    Auth-gated (Library is shared, but only authenticated users can read).
    Returns 404 with `detail="thumbnail_unavailable"` when the scene exists
    but no thumbnail was rendered (e.g. older scenes from before
    Step 7c-extended, or onboarding completed with thumbnail_path=None).

    The path stored on the row is relative to allotrope_artifacts. We
    constrain the resolved path to the artifacts root so a malicious
    `thumbnail_path` (which could only land via direct DB write) can't
    escape via `..`.
    """
    raw_id = parse_prefixed_id("scene", scene_id)
    scene = db.get(Scene, raw_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="scene_not_found",
        )
    if not scene.thumbnail_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="thumbnail_unavailable",
        )

    artifacts_root = Path(settings.artifacts_dir).resolve()
    full = (artifacts_root / scene.thumbnail_path).resolve()
    # Defence in depth — confine to the artifacts root.
    try:
        full.relative_to(artifacts_root)
    except ValueError:
        logger.warning(
            "scene %s thumbnail_path=%r escapes artifacts root", scene.id, scene.thumbnail_path
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="thumbnail_unavailable",
        )
    if not full.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="thumbnail_unavailable",
        )
    # ETag-friendly defaults; aggressive caching since thumbnails are
    # write-once.
    return FileResponse(
        path=str(full),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --- DELETE /scenes/{id} ---------------------------------------------


@router.delete(
    "/{scene_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a Scene from the Library",
    response_class=Response,
)
def delete_scene(
    scene_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """Synchronous Scene delete.

    Blocked (409 with the list of project ids) when any Project still
    references this Scene — Scene→Project is RESTRICT per the spec, and
    we surface the dependency so the UI can prompt the user to delete
    the project first. Annotations CASCADE on the scenes FK and disappear
    with the row. After the DB row deletes, we rmtree both `/data/scenes/<id>/`
    (raw + vendable + staged annotation files) and `/artifacts/scenes/<id>/`
    (thumbnails, pre-rendered visualizations, annotation overlays).
    """
    raw_id = parse_prefixed_id("scene", scene_id)
    scene = db.get(Scene, raw_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_found"
        )

    # RESTRICT check — surface the blocking project ids so the frontend
    # can offer a "delete project first" prompt instead of just 409ing.
    blocking = list(
        db.scalars(
            select(Project.id).where(Project.scene_id == scene.id)
        ).all()
    )
    if blocking:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "scene_in_use",
                "blocking_project_ids": [f"project_{p}" for p in blocking],
            },
        )

    scene_uuid = scene.id
    db.delete(scene)
    db.commit()

    # Best-effort filesystem cleanup. Don't raise if a path is missing;
    # the DB row is the source of truth and is already gone.
    data_dir = Path(settings.data_dir) / "scenes" / str(scene_uuid)
    artifacts_dir = Path(settings.artifacts_dir) / "scenes" / str(scene_uuid)
    for d in (data_dir, artifacts_dir):
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError as exc:
                logger.warning(
                    "scene %s delete: rmtree(%s) failed: %s",
                    scene_uuid,
                    d,
                    exc,
                )

    logger.info("scene deleted id=%s", scene_uuid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
