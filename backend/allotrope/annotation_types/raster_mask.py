"""raster_mask annotation type.

v1: binary or multi-class pixel labels stored as a single-band TIF.
Treats any non-zero pixel as anomaly. Multi-class palette support is a
future extension via `multi_class_raster_mask` (see spec § 5.3).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("allotrope.worker.annotation_types.raster_mask")

KIND: str = "raster_mask"
LABEL: str = "Raster mask"
ACCEPTED_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff")

# Cyan-on-thermal is the canonical palette in the benchmarking
# notebooks (`benchmarking/thermal/grx_benchmark.ipynb` cell 6: cyan
# scatter markers on inferno-mapped scenes). Reusing here so the api
# overlays match what an analyst sees in their own notebook.
_ANOMALY_FILL_RGBA = (0, 255, 255, 220)

# Same cap as the band-image endpoint — overlays render at the same
# resolution as composites (typically 1219×1210 PRISMA; ~4096 for
# Landsat thermal capped from 7700).
_OVERLAY_MAX_DIM = 4096

# Visible dot radius IN OUTPUT PIXELS (after the source→output coord
# map). Tuned for read-at-glance against composites at full-image-fit
# zoom. The api's overlay endpoint accepts ?radius=N so the frontend
# can override per annotation when the GT is dense and dots overlap,
# or sparse and need to be cranked up.
_OVERLAY_DOT_RADIUS_DEFAULT = 14
_OVERLAY_DOT_RADIUS_MIN = 2
_OVERLAY_DOT_RADIUS_MAX = 60


def validate_upload(filename: str) -> None:
    """Cheap api-side check. Worker re-validates on read."""
    lower = filename.lower()
    if not any(lower.endswith(ext) for ext in ACCEPTED_EXTENSIONS):
        raise ValueError(
            f"raster_mask requires one of {ACCEPTED_EXTENSIONS}; got {filename!r}"
        )


def materialise(staging_file: Path, final_dir: Path) -> Path:
    """Move the staged TIF into its final dir under the original name."""
    final_dir.mkdir(parents=True, exist_ok=True)
    final_file = final_dir / staging_file.name
    if final_file.exists():
        raise FileExistsError(f"target already exists: {final_file}")
    shutil.move(str(staging_file), str(final_file))
    return final_file


def render_overlay(
    final_file: Path,
    dest_png: Path,
    *,
    radius: int | None = None,
) -> bool:
    """Render an RGBA PNG with cyan circular dots at each anomaly pixel.

    Mirrors the technique in `benchmarking/thermal/grx_benchmark.ipynb`:
    instead of filling anomaly pixels with colour (which produces lumpy
    pixel-grid blobs at viewport zoom), draw a cyan circle of fixed
    radius at each anomaly's coordinate. The result reads as discrete
    markers, like matplotlib's `scatter(c="cyan", marker="o")`.

    `radius` is in OUTPUT pixels (after the source→output coord map);
    pass None to use the registry default. The api's overlay endpoint
    forwards a `?radius=` query param here so the frontend can tune
    dot size per-annotation without re-onboarding.

    Pipeline:
        1. Read mask from disk (rasterio).
        2. Find anomaly pixel coordinates in source space.
        3. Map each (row, col) to output space via the resolution cap.
        4. Stamp a circle at each output coordinate using PIL.ImageDraw.

    Output dimensions match the colour composite (capped at
    _OVERLAY_MAX_DIM in the longest axis), so the frontend can layer
    this PNG over the composite at the same scale.
    """
    from PIL import Image, ImageDraw

    r = radius if radius is not None else _OVERLAY_DOT_RADIUS_DEFAULT
    r = max(_OVERLAY_DOT_RADIUS_MIN, min(_OVERLAY_DOT_RADIUS_MAX, r))

    mask = _read_mask(final_file)
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D mask, got shape {mask.shape}")

    src_h, src_w = mask.shape
    stride = max(1, max(src_h, src_w) // _OVERLAY_MAX_DIM)
    out_h = src_h // stride
    out_w = src_w // stride
    if out_h == 0 or out_w == 0:
        out_h, out_w = src_h, src_w
        stride = 1

    # Anomaly coordinates in source space.
    rows_src, cols_src = np.where(mask != 0)

    # PIL canvas + draw context. RGBA so background is fully transparent.
    img = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

    if rows_src.size == 0:
        # All-zero mask — emit an empty transparent PNG so the api's
        # has_overlay check still flips true and the row stays consistent.
        dest_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_png, format="PNG", optimize=True)
        return True

    # Map source coords → output coords. Round, clip into bounds.
    rows_out = np.clip(rows_src // stride, 0, out_h - 1)
    cols_out = np.clip(cols_src // stride, 0, out_w - 1)

    # Deduplicate output coordinates so multiple source pixels
    # collapsing into one output pixel only stamp one dot. Speeds up
    # rendering for dense GTs without changing the visual outcome.
    flat = rows_out.astype(np.int64) * out_w + cols_out.astype(np.int64)
    unique = np.unique(flat)
    rows_uniq = (unique // out_w).astype(np.int32)
    cols_uniq = (unique % out_w).astype(np.int32)

    draw = ImageDraw.Draw(img)
    fill = _ANOMALY_FILL_RGBA
    # ImageDraw.ellipse takes (x0, y0, x1, y1) — the box bounding the
    # ellipse. We draw filled, no outline, since the cyan is bright
    # enough at alpha 220 to read against any background.
    for r_o, c_o in zip(rows_uniq, cols_uniq):
        draw.ellipse(
            (int(c_o) - r, int(r_o) - r, int(c_o) + r, int(r_o) + r),
            fill=fill,
        )

    dest_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_png, format="PNG", optimize=True)
    return True


def extract_metadata(final_file: Path) -> dict[str, Any]:
    """Pixel counts: total / anomaly / coverage %."""
    try:
        mask = _read_mask(final_file)
    except Exception:
        logger.exception("metadata extraction failed for %s", final_file)
        return {"filename": final_file.name}
    total = int(mask.size)
    anomaly = int((mask != 0).sum())
    coverage_pct = (100.0 * anomaly / total) if total else 0.0
    return {
        "filename": final_file.name,
        "shape": list(mask.shape),
        "anomaly_pixel_count": anomaly,
        "total_pixel_count": total,
        "coverage_pct": round(coverage_pct, 4),
    }


# --- Helpers ----------------------------------------------------------


def _read_mask(path: Path) -> np.ndarray:
    """Read the first band of a single-band GT raster."""
    import rasterio
    with rasterio.open(str(path)) as src:
        return np.asarray(src.read(1))


