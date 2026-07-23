"""Scene-visualization endpoints.

Routes (mounted under the `/scenes` router):

    GET /scenes/{scene_id}/visualizations
        List the pre-rendered visualizations available on disk for a
        scene. Convention-based discovery — no DB column per kind.

    GET /scenes/{scene_id}/visualizations/{kind}/image
        Stream a pre-rendered PNG (color | nir | swir | ndvi | band_mosaic).

    GET /scenes/{scene_id}/histogram
        Stream the histogram JSON the worker wrote at onboard time.
        Frontend renders this with uplot.

    GET /scenes/{scene_id}/spectrum?row=R&col=C
        Single-pixel reflectance + wavelengths from the persisted
        vendable. Hyperspectral only (Landsat returns 422).

    GET /scenes/{scene_id}/bands
        List of bands with wavelength + spectral family + validity.
        Drives the Band Browser selector.

    GET /scenes/{scene_id}/bands/{index}/image
        Single-band rendered as a percentile-stretched grayscale PNG
        on-demand from the persisted vendable. Hyperspectral only.

For the spectrum endpoint we maintain a tiny in-process LRU of unpickled
vendables (the pickle is 1.5 GB for PRISMA — re-reading per click is a
non-starter). Cache size is intentionally small; production would swap
to a real cache or rebuild on demand.
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # before pyplot import — headless container

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from matplotlib import pyplot as plt
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..config import settings
from ..db import get_db
from ..models import Scene
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.visualizations")

router = APIRouter(prefix="/scenes", tags=["scenes"])

# kinds the worker writes; we look for `<kind>.png` in the dir.
_PNG_KINDS = ("color", "nir", "swir", "ndvi", "band_mosaic")
_KIND_LABELS: dict[str, str] = {
    "color": "Color (true-colour / thermal)",
    "nir": "NIR false-colour",
    "swir": "SWIR composite",
    "ndvi": "NDVI",
    "band_mosaic": "Band mosaic",
}

# Cache size: 2 vendables × ~1.5 GB PRISMA = ~3 GB max. Tuneable via env.
_VENDABLE_CACHE_SIZE = 2


# ---------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------


class VisualizationItem(BaseModel):
    kind: str           # "color" | "nir" | "swir" | "ndvi" | "band_mosaic"
    label: str          # human-readable label for the UI
    image_url: str      # api-relative path: /scenes/<id>/visualizations/<kind>/image


class VisualizationList(BaseModel):
    items: list[VisualizationItem]
    histogram_url: str | None  # set if histogram.json exists


class SpectrumPoint(BaseModel):
    """One band's worth of data."""

    wavelength_nm: float
    reflectance: float
    spectral_family: str | None
    is_valid: bool


class SpectrumResponse(BaseModel):
    row: int
    col: int
    height: int
    width: int
    points: list[SpectrumPoint]


# ---------------------------------------------------------------------
# Vendable cache (process-local LRU)
# ---------------------------------------------------------------------


_cache_lock = threading.Lock()
_cache: "OrderedDict[str, Any]" = OrderedDict()


def _load_vendable(vendable_abs: Path) -> Any:
    """Return the unpickled vendable for `vendable_abs`. Cached per-path."""
    key = str(vendable_abs)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    # Heavy work — outside the lock so concurrent requests for *different*
    # scenes don't serialise. This means two threads could both unpickle
    # the same scene concurrently, but the inserts are idempotent.
    logger.info("loading vendable into cache: %s", vendable_abs)
    with vendable_abs.open("rb") as f:
        obj = pickle.load(f)
    with _cache_lock:
        _cache[key] = obj
        _cache.move_to_end(key)
        while len(_cache) > _VENDABLE_CACHE_SIZE:
            evicted_key, _ = _cache.popitem(last=False)
            logger.info("evicted vendable from cache: %s", evicted_key)
    return obj


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _scene_or_404(scene_id_wire: str, db: Session) -> Scene:
    raw_id = parse_prefixed_id("scene", scene_id_wire)
    scene = db.get(Scene, raw_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="scene_not_found"
        )
    return scene


def _viz_dir(scene: Scene) -> Path:
    """Convention-based path: scenes/<id>/visualizations/."""
    return Path(settings.artifacts_dir) / "scenes" / str(scene.id) / "visualizations"


def _confined_path(root: Path, candidate: Path) -> Path:
    """Confine `candidate` to `root` — defence in depth."""
    full = candidate.resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="not_found"
        ) from exc
    return full


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------


@router.get(
    "/{scene_id}/visualizations",
    response_model=VisualizationList,
    status_code=status.HTTP_200_OK,
    summary="List pre-rendered visualizations for a scene",
)
def list_visualizations(
    scene_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> VisualizationList:
    """Discover what the worker wrote into the visualizations dir.

    A landsat9 scene typically has only `color`; hyperspectral scenes
    have all of `color`, `nir`, `swir`, `ndvi`, `band_mosaic`. The
    histogram URL is only set when histogram.json exists.
    """
    scene = _scene_or_404(scene_id, db)
    viz_dir = _viz_dir(scene)
    items: list[VisualizationItem] = []
    for kind in _PNG_KINDS:
        if (viz_dir / f"{kind}.png").is_file():
            items.append(
                VisualizationItem(
                    kind=kind,
                    label=_KIND_LABELS[kind],
                    image_url=f"/scenes/{scene_id}/visualizations/{kind}/image",
                )
            )
    histogram_url = (
        f"/scenes/{scene_id}/histogram"
        if (viz_dir / "histogram.json").is_file()
        else None
    )
    return VisualizationList(items=items, histogram_url=histogram_url)


@router.get(
    "/{scene_id}/visualizations/{kind}/image",
    status_code=status.HTTP_200_OK,
    summary="Stream a pre-rendered visualization PNG",
    response_class=FileResponse,
)
def get_visualization_image(
    scene_id: str,
    kind: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> FileResponse:
    if kind not in _PNG_KINDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="unknown_kind"
        )
    scene = _scene_or_404(scene_id, db)
    viz_dir = _viz_dir(scene)
    path = _confined_path(viz_dir, viz_dir / f"{kind}.png")
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="visualization_unavailable"
        )
    return FileResponse(
        path=str(path),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get(
    "/{scene_id}/histogram",
    status_code=status.HTTP_200_OK,
    summary="Pixel-value histogram bins (JSON)",
)
def get_scene_histogram(
    scene_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    scene = _scene_or_404(scene_id, db)
    viz_dir = _viz_dir(scene)
    path = _confined_path(viz_dir, viz_dir / "histogram.json")
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="histogram_unavailable"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get(
    "/{scene_id}/spectrum",
    response_model=SpectrumResponse,
    status_code=status.HTTP_200_OK,
    summary="Reflectance spectrum at a single (row, col)",
)
def get_scene_spectrum(
    scene_id: str,
    row: int = Query(..., ge=0, description="Pixel row (0-based)"),
    col: int = Query(..., ge=0, description="Pixel column (0-based)"),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> SpectrumResponse:
    """Return the full spectrum at one pixel.

    Hyperspectral only — landsat9 scenes are single-band thermal and
    return 422 `unsupported_sensor`. Out-of-range pixels return 422
    `out_of_range`. Cold-cache calls take ~3–5 s for PRISMA while the
    pickle is read; warm-cache calls are <100 ms.
    """
    scene = _scene_or_404(scene_id, db)
    if scene.sensor_type == "landsat9":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unsupported_sensor",
        )

    vendable_abs = Path(settings.data_dir) / scene.vendable_path
    if not vendable_abs.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="vendable_unavailable"
        )

    vendable = _load_vendable(vendable_abs)
    cube = vendable.normalized_hyperspectral_cube  # (B, H, W)
    height = int(cube.shape[1])
    width = int(cube.shape[2])
    if row >= height or col >= width:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="out_of_range",
        )

    wavelengths = list(vendable.band_cw_order)
    spectrum = cube[:, row, col]
    families = [getattr(f, "value", str(f)) for f in vendable.spectral_family_order]
    band_validity = list(vendable.band_validity_by_position)

    points = [
        SpectrumPoint(
            wavelength_nm=float(wavelengths[i]),
            reflectance=float(spectrum[i]),
            spectral_family=families[i] if i < len(families) else None,
            is_valid=bool(band_validity[i]) if i < len(band_validity) else True,
        )
        for i in range(len(wavelengths))
    ]

    return SpectrumResponse(
        row=row,
        col=col,
        height=height,
        width=width,
        points=points,
    )


# ---------------------------------------------------------------------
# Band browser
# ---------------------------------------------------------------------


class BandInfo(BaseModel):
    index: int                       # 0-based position in the cube
    wavelength_nm: float
    spectral_family: str | None      # "VNIR" | "SWIR" | …
    is_valid: bool


class BandList(BaseModel):
    sensor_type: str
    band_count: int
    bands: list[BandInfo]


# Bound the on-demand band render so a malformed scene can't allocate a
# huge PNG. 1024 px is plenty for a band browser preview.
_BAND_IMAGE_MAX_DIM = 1024


# 256-entry inferno LUT — built once at module load. Same trick the
# worker uses (allotrope_worker.visualizations._apply_colormap_uint8):
# avoid materialising a (H, W, 4) float64 RGBA via cm.inferno(...),
# index a uint8 RGB LUT instead.
_INFERNO_LUT: np.ndarray = (
    plt.get_cmap("inferno")(np.linspace(0.0, 1.0, 256))[:, :3] * 255
).astype(np.uint8)


def _apply_inferno(norm01: np.ndarray) -> np.ndarray:
    """Return `(H, W, 3)` uint8 RGB with inferno applied. `norm01` is
    expected in `[0, 1]`; out-of-range values clip to LUT endpoints."""
    idx = np.clip(np.rint(norm01 * 255.0).astype(np.int32), 0, 255)
    return _INFERNO_LUT[idx]


@router.get(
    "/{scene_id}/bands",
    response_model=BandList,
    status_code=status.HTTP_200_OK,
    summary="List bands (wavelength + family + validity) for a hyperspectral scene",
)
def list_bands(
    scene_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> BandList:
    """Hyperspectral only — Landsat returns 422 unsupported_sensor.

    Reads `band_cw_order`, `spectral_family_order`, and
    `band_validity_by_position` directly from the persisted vendable
    (cached). No PNG rendering here — that's the per-band endpoint.
    """
    scene = _scene_or_404(scene_id, db)
    if scene.sensor_type == "landsat9":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unsupported_sensor",
        )

    vendable_abs = Path(settings.data_dir) / scene.vendable_path
    if not vendable_abs.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="vendable_unavailable"
        )
    vendable = _load_vendable(vendable_abs)

    wavelengths = list(vendable.band_cw_order)
    families = [getattr(f, "value", str(f)) for f in vendable.spectral_family_order]
    validity = list(vendable.band_validity_by_position)

    bands = [
        BandInfo(
            index=i,
            wavelength_nm=float(wavelengths[i]),
            spectral_family=families[i] if i < len(families) else None,
            is_valid=bool(validity[i]) if i < len(validity) else True,
        )
        for i in range(len(wavelengths))
    ]
    return BandList(
        sensor_type=scene.sensor_type,
        band_count=len(bands),
        bands=bands,
    )


@router.get(
    "/{scene_id}/bands/{index}/image",
    status_code=status.HTTP_200_OK,
    summary="Render a single band as a grayscale PNG on-demand",
)
def get_band_image(
    scene_id: str,
    index: int,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    """Stream `cube[index]` as a percentile-stretched grayscale PNG.

    Render path:
      - Load (cached) vendable.
      - Slice the band → downsample to ~1024 px (cap memory).
      - Mask invalid pixels via the 2-D validity reduction.
      - 2nd–98th percentile stretch over valid pixels.
      - Encode PNG into a memory buffer; stream as the response.

    Hyperspectral only. Out-of-range index → 422.
    """
    scene = _scene_or_404(scene_id, db)
    if scene.sensor_type == "landsat9":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unsupported_sensor",
        )
    if index < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_index",
        )

    vendable_abs = Path(settings.data_dir) / scene.vendable_path
    if not vendable_abs.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="vendable_unavailable"
        )
    vendable = _load_vendable(vendable_abs)

    cube = vendable.normalized_hyperspectral_cube  # (B, H, W)
    if index >= cube.shape[0]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="band_out_of_range",
        )

    # Stride-slice for memory bound; same trick as the worker
    # compositors so a 30k×30k future scene doesn't OOM the api.
    h, w = cube.shape[1], cube.shape[2]
    stride = max(1, max(h, w) // _BAND_IMAGE_MAX_DIM)
    plane = np.asarray(
        cube[index, ::stride, ::stride], dtype=np.float32
    )

    # Validity mask (any-band reduction across the cube), same stride.
    vc = getattr(vendable, "validity_cube", None)
    valid_mask: np.ndarray | None = None
    if vc is not None:
        vc_arr = np.asarray(vc)
        if vc_arr.ndim == 3:
            mask = np.any(vc_arr != 0, axis=0)
        elif vc_arr.ndim == 2:
            mask = vc_arr != 0
        else:
            mask = None
        if mask is not None:
            valid_mask = mask[::stride, ::stride].astype(np.uint8)

    if valid_mask is not None:
        sample = plane[(valid_mask != 0) & np.isfinite(plane)]
    else:
        sample = plane[np.isfinite(plane)]
    if sample.size == 0:
        # All-invalid band — return a black 1×1 PNG.
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), color=(0, 0, 0)).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    lo, hi = np.percentile(sample, [2, 98])
    if hi <= lo:
        norm01 = np.zeros_like(plane, dtype=np.float32)
    else:
        norm01 = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    rgb = _apply_inferno(norm01)
    if valid_mask is not None:
        # Force invalid pixels to black so the rotated zero-fill border
        # stays crisp instead of rendering as inferno's purple low end.
        rgb = np.where(valid_mask[..., None] != 0, rgb, 0).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        # Bands are deterministic given (scene, index, vendable.pkl);
        # cache aggressively so flipping back and forth is instant.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
