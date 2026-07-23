"""Pre-rendered scene visualizations.

Called from scene_onboard after the vendable is built. Produces a fixed
set of static artifacts under
    allotrope_artifacts/scenes/<scene_id>/visualizations/

Layout:
    color.png        — sensor-aware: thermal colormap (Landsat) or true-
                       colour RGB composite (PRISMA / EnMAP). Replaces
                       the grayscale `thumbnail.png` from 7c-extended.
    nir.png          — NIR / Red / Green false-colour (hyperspectral only)
    swir.png         — SWIR2 / SWIR1 / Red composite (hyperspectral only)
    ndvi.png         — coloured (NIR-Red)/(NIR+Red) heatmap (hyperspectral)
    band_mosaic.png  — every band as a small tile (hyperspectral). Reuses
                       app.utils.visualization.hyperspectral_visualizer.
    histogram.json   — { sensor_type, kind, bins[], counts[], stats{...} }
                       Driven by uplot on the frontend (interactive).

The Scene's `thumbnail_path` column points at color.png so the Library
table picks up the colour version for free.

Convention-based discovery: the api lists whatever PNGs exist in the
visualizations dir at request time, no DB column per kind.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Matplotlib needs a non-interactive backend in a headless container.
# Import-time side effect — must run before pyplot is touched anywhere.
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from app.utils.visualization.hyperspectral_visualizer import plot_band_mosaic

logger = logging.getLogger("allotrope.worker.visualizations")

# Canonical wavelengths (nm) for hyperspectral compositing. Picked from
# common Landsat / Sentinel band definitions; PRISMA + EnMAP both
# straddle these comfortably.
_VISIBLE_RGB_NM = {"R": 660.0, "G": 550.0, "B": 470.0}
_NIR_FALSECOLOR_NM = {"R": 860.0, "G": 660.0, "B": 550.0}   # NIR / Red / Green
_SWIR_FALSECOLOR_NM = {"R": 2200.0, "G": 1610.0, "B": 660.0}  # SWIR2 / SWIR1 / Red
_NDVI_BANDS_NM = {"red": 660.0, "nir": 860.0}

# Render at native cube resolution by default — gives the frontend's
# zoom/pan something to zoom into. We still cap at a safe ceiling so a
# pathological future scene (think 30k × 30k EnMAP variant) doesn't
# allocate more than ~50 MB of pixel buffer.
_RENDER_MAX_DIM = 4096
_BAND_MOSAIC_TARGET_DIM = 1400


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _band_index_at(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    """Index of the band closest to `target_nm`."""
    return int(np.argmin(np.abs(wavelengths_nm - target_nm)))


def _stretch_uint8(
    plane: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> np.ndarray:
    """Robust percentile stretch → uint8.

    Without a mask the percentiles get pulled toward the rotated
    zero-fill border that all our test scenes have, collapsing the
    visible dynamic range. Pass `valid_mask` (1 = valid) to compute
    lo/hi only over real pixels — the result is still a full-size
    array but the invalid region renders as 0 (black) which looks
    correct against any colormap.
    """
    arr = np.asarray(plane, dtype=np.float32)
    if valid_mask is not None:
        sample = arr[(valid_mask != 0) & np.isfinite(arr)]
    else:
        sample = arr[np.isfinite(arr)]
    if sample.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out = (norm * 255.0).astype(np.uint8)
    if valid_mask is not None:
        # Force invalid pixels to 0 — keeps the rotated border crisp
        # instead of bleeding stretched-low values into it.
        out = np.where(valid_mask != 0, out, 0).astype(np.uint8)
    return out


def _hyperspectral_validity_2d(vendable) -> np.ndarray | None:
    """Collapse the 3-D validity_cube to a 2-D pixel mask (1 = any valid band).

    The validity_cube is per-pixel-per-band (B, H, W) — for compositing
    we only need a 2-D footprint. ANY-axis-0 reduction is a generous
    definition that still excludes the rotated zero-fill border.
    """
    vc = getattr(vendable, "validity_cube", None)
    if vc is None:
        return None
    arr = np.asarray(vc)
    if arr.ndim == 2:
        return (arr != 0).astype(np.uint8)
    if arr.ndim == 3:
        return np.any(arr != 0, axis=0).astype(np.uint8)
    return None


def _apply_colormap_uint8(norm01: np.ndarray, cmap_name: str) -> np.ndarray:
    """Apply a matplotlib colormap via a 256-entry uint8 LUT.

    Why not just `cm.<name>(norm01)`? Matplotlib's colormap call returns
    `(H, W, 4)` float64 RGBA. At 7700×7700 (Landsat thermal) that's
    ~1.9 GB and OOM-kills the worker. The LUT path materialises the
    output as `(H, W, 3)` uint8 only — ~178 MB at that size.

    `norm01` is expected in `[0, 1]`; values outside that range are
    clipped to the LUT endpoints.
    """
    cmap = plt.get_cmap(cmap_name)
    lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)
    idx = np.clip(np.rint(norm01 * 255.0).astype(np.int32), 0, 255)
    return lut[idx]  # (H, W) → (H, W, 3) uint8


def _downsample_for_thumb(arr: np.ndarray, *, max_dim: int) -> np.ndarray:
    """Stride-slice down to ~max_dim before any colormap step.

    Without this we materialise an `H × W × 4 × 8` RGBA array on a
    7721×7571 Landsat scene → ~1.9 GB, OOM-kills the worker. By slicing
    first the colormap runs on a ~max_dim × max_dim plane (~1 MB).
    Quality difference vs. resizing afterwards is invisible at 512 px.
    """
    if arr.ndim < 2:
        return arr
    h = arr.shape[-2]
    w = arr.shape[-1]
    stride = max(1, max(h, w) // max_dim)
    if stride == 1:
        return arr
    if arr.ndim == 2:
        return arr[::stride, ::stride]
    if arr.ndim == 3:
        return arr[..., ::stride, ::stride]
    return arr


def _save_png(img: Image.Image, dest: Path, *, max_dim: int) -> None:
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG", optimize=True)


def _thermal_units_hint(vendable) -> str:
    """Best-guess units label for the thermal colorbar legend."""
    units = getattr(vendable, "units", None)
    if isinstance(units, str):
        if units.lower().startswith("celsius"):
            return "°C"
        if units.lower().startswith("kelvin"):
            return "K"
        if units.startswith("DN_"):
            return "DN"
    return ""


def _append_colorbar_strip(
    rgb: np.ndarray,
    *,
    cmap_name: str,
    units_hint: str,
    value_range: tuple[float, float],
    bar_height_frac: float = 0.025,
    label_band_px: int = 18,
) -> np.ndarray:
    """Append a small colorbar strip + min/max labels to the bottom of
    a thumbnail. Keeps the rendered file self-explanatory without the
    frontend having to compose its own legend.

    Bar fills the full width of the image; label band sits below it with
    ``<min> <units>``  …  ``<max> <units>`` in light text on dark.
    """
    h, w = rgb.shape[:2]
    bar_h = max(12, int(round(h * bar_height_frac)))
    cmap = plt.get_cmap(cmap_name)
    lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)
    # 1D gradient stretched to image width.
    grad_idx = np.clip(
        np.linspace(0, 255, w, endpoint=True).astype(np.int32), 0, 255
    )
    grad_row = lut[grad_idx]  # (w, 3) uint8
    bar = np.broadcast_to(grad_row[None, ...], (bar_h, w, 3)).copy()

    # Label band — dark with light text. Drawn via PIL so we don't
    # need to ship a font file; default PIL bitmap font is fine for
    # the size we're at.
    label_band = np.full((label_band_px, w, 3), 8, dtype=np.uint8)
    text_img = Image.fromarray(label_band, mode="RGB")
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(text_img)
        lo, hi = value_range
        lo_text = _fmt_value(lo, units_hint)
        hi_text = _fmt_value(hi, units_hint)
        draw.text((4, 2), lo_text, fill=(220, 220, 220))
        # Right-align hi_text.
        # PIL default font is bitmap ~6 px/char; rough estimate is fine.
        approx_w = len(hi_text) * 6
        draw.text(
            (max(4, w - approx_w - 4), 2),
            hi_text,
            fill=(220, 220, 220),
        )
    except Exception:  # noqa: BLE001 — label band is cosmetic
        pass
    label_band = np.asarray(text_img)

    return np.concatenate([rgb, bar, label_band], axis=0)


def _fmt_value(v: float, units: str) -> str:
    """Compact numeric formatting for the colorbar end labels."""
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{int(round(v))}{(' ' + units) if units else ''}"
    if abs(v) >= 10:
        return f"{v:.1f}{(' ' + units) if units else ''}"
    return f"{v:.3f}{(' ' + units) if units else ''}"


def _hyperspectral_cube(vendable) -> np.ndarray:
    """Return the cube as (bands, rows, cols) — BSQ. Raises if shape
    can't be deduced. The PRISMA + EnMAP builders both vend in this
    layout per the patcher's `cube[sort_idx]` indexing convention."""
    arr = np.asarray(vendable.normalized_hyperspectral_cube)
    if arr.ndim != 3:
        raise ValueError(
            f"expected 3-D cube, got shape {arr.shape}"
        )
    return arr


def _hyperspectral_composite(
    vendable,
    band_targets_nm: dict[str, float],
) -> Image.Image:
    """Generic three-band composite from a hyperspectral vendable.

    `band_targets_nm` is a {channel: wavelength_nm} dict. Channels are
    used in iteration order, i.e. `{"R": 660, "G": 550, "B": 470}` →
    R-channel pulled from the band closest to 660 nm.

    Each per-channel band is downsampled to ~_RENDER_MAX_DIM BEFORE
    the percentile stretch and stack. This caps memory regardless of
    scene size — a 30k×30k future EnMAP scene composites in the same
    memory budget as a 1k×1k one.
    """
    cube = _hyperspectral_cube(vendable)  # (B, H, W)
    wls = np.asarray(vendable.band_cw_order, dtype=np.float32)
    if cube.shape[0] != wls.shape[0]:
        raise ValueError(
            f"band axis size {cube.shape[0]} != wavelength count {wls.shape[0]}"
        )

    valid_mask = _hyperspectral_validity_2d(vendable)
    mask_small = (
        _downsample_for_thumb(valid_mask, max_dim=_RENDER_MAX_DIM)
        if valid_mask is not None
        else None
    )

    planes = []
    for _channel, nm in band_targets_nm.items():
        idx = _band_index_at(wls, nm)
        plane_small = _downsample_for_thumb(cube[idx], max_dim=_RENDER_MAX_DIM)
        planes.append(_stretch_uint8(plane_small, valid_mask=mask_small))
    rgb = np.stack(planes, axis=-1)  # (h, w, 3) — already small
    return Image.fromarray(rgb, mode="RGB")


# ---------------------------------------------------------------------
# Per-kind renderers
# ---------------------------------------------------------------------


def render_color_thumbnail(vendable, sensor_type: str, dest: Path) -> None:
    """Sensor-aware default preview. Replaces the older grayscale thumb."""
    if sensor_type in ("landsat9", "hotsat1"):
        # Single-band thermal — `magma` colormap. Magma keeps the
        # dark→bright thermal connotation but uses a different hue
        # ramp from `inferno` (which is reserved for anomaly score
        # rendering); this avoids visually conflating "thermal value"
        # with "anomaly probability" across the catalog.
        cube = np.asarray(vendable.normalized_thermal_cube, dtype=np.float32)
        plane = np.squeeze(cube)
        if plane.ndim != 2:
            band_axis = int(np.argmin(plane.shape))
            plane = np.take(plane, plane.shape[band_axis] // 2, axis=band_axis)

        # Build a 2-D valid mask from whichever validity-flavoured field
        # the thermal vendable carries. `pure_validity_mask` is the
        # clean version (excludes invalid border + bad pixels); fall
        # back to the cloud-aware `validity_cube` if pure isn't set.
        valid_mask: np.ndarray | None = None
        for attr in ("pure_validity_mask", "validity_cube"):
            mask_obj = getattr(vendable, attr, None)
            if mask_obj is not None:
                m = np.asarray(np.squeeze(np.asarray(mask_obj)))
                if m.ndim == 2 and m.shape == plane.shape:
                    valid_mask = (m != 0).astype(np.uint8)
                    break

        # Cap to _RENDER_MAX_DIM but keep native resolution for normal-
        # sized scenes — the frontend's pan/zoom needs pixels to zoom
        # into. The LUT colormap below produces uint8 RGB directly so
        # we don't materialise a float64 RGBA array even at native res.
        plane_small = _downsample_for_thumb(plane, max_dim=_RENDER_MAX_DIM)
        mask_small = (
            _downsample_for_thumb(valid_mask, max_dim=_RENDER_MAX_DIM)
            if valid_mask is not None
            else None
        )

        if mask_small is not None:
            sample = plane_small[(mask_small != 0) & np.isfinite(plane_small)]
        else:
            sample = plane_small[np.isfinite(plane_small)]
        if sample.size == 0:
            Image.new("RGB", (1, 1), color=(0, 0, 0)).save(dest)
            return
        lo, hi = np.percentile(sample, [2, 98])
        if hi <= lo:
            norm = np.zeros_like(plane_small, dtype=np.float32)
        else:
            norm = np.clip((plane_small - lo) / (hi - lo), 0.0, 1.0)
        rgb = _apply_colormap_uint8(norm, "magma")
        if mask_small is not None:
            rgb = np.where(mask_small[..., None] != 0, rgb, 0).astype(np.uint8)
        # Append a small magma colorbar strip along the bottom so the
        # rendered file is self-explanatory. Strip height is 2.5% of
        # the image height or 12 px, whichever is larger.
        rgb_with_bar = _append_colorbar_strip(
            rgb, cmap_name="magma", units_hint=_thermal_units_hint(vendable),
            value_range=(float(lo), float(hi)),
        )
        img = Image.fromarray(rgb_with_bar, mode="RGB")
    else:
        # Hyperspectral → visible RGB true-colour. Validity handled inside.
        img = _hyperspectral_composite(vendable, _VISIBLE_RGB_NM)
    _save_png(img, dest, max_dim=_RENDER_MAX_DIM)


def render_nir_composite(vendable, dest: Path) -> None:
    img = _hyperspectral_composite(vendable, _NIR_FALSECOLOR_NM)
    _save_png(img, dest, max_dim=_RENDER_MAX_DIM)


def render_swir_composite(vendable, dest: Path) -> None:
    img = _hyperspectral_composite(vendable, _SWIR_FALSECOLOR_NM)
    _save_png(img, dest, max_dim=_RENDER_MAX_DIM)


def render_ndvi(vendable, dest: Path) -> None:
    """Coloured NDVI = (NIR - Red) / (NIR + Red), -1..1.

    Uses the RdYlGn colormap (red bare → green vegetation), which is the
    standard NDVI palette in remote-sensing tools. Both bands are
    downsampled before the divide so the colormap step's RGBA array
    stays small regardless of scene size. Invalid pixels (the rotated
    zero-fill border) are knocked to black.
    """
    cube = _hyperspectral_cube(vendable)
    wls = np.asarray(vendable.band_cw_order, dtype=np.float32)
    red_full = cube[_band_index_at(wls, _NDVI_BANDS_NM["red"])]
    nir_full = cube[_band_index_at(wls, _NDVI_BANDS_NM["nir"])]
    red = _downsample_for_thumb(red_full, max_dim=_RENDER_MAX_DIM).astype(np.float32)
    nir = _downsample_for_thumb(nir_full, max_dim=_RENDER_MAX_DIM).astype(np.float32)

    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denom != 0, (nir - red) / denom, 0.0)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    norm = (ndvi + 1.0) / 2.0
    rgb = _apply_colormap_uint8(norm, "RdYlGn")

    valid_full = _hyperspectral_validity_2d(vendable)
    valid_small = (
        _downsample_for_thumb(valid_full, max_dim=_RENDER_MAX_DIM)
        if valid_full is not None
        else None
    )
    if valid_small is not None:
        rgb = np.where(valid_small[..., None] != 0, rgb, 0).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    _save_png(img, dest, max_dim=_RENDER_MAX_DIM)


def render_band_mosaic(vendable, dest: Path, *, sensor: str) -> None:
    """Reuse the existing plot_band_mosaic figure; save as PNG.

    `plot_band_mosaic` accepts `save_path=` directly and writes the
    figure to disk; we let it do that. The result is a multi-panel
    image (every band as a tile, wavelength labels) — useful for
    spotting noisy / atmospheric bands at a glance.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # plot_band_mosaic handles BSQ cubes natively for both PRISMA and
    # EnMAP vendables (its docstring covers both types).
    plot_band_mosaic(
        vendable,
        save_path=str(dest),
        title=None,
    )
    # Downscale post-hoc so the mosaic doesn't blow past a few MB.
    try:
        with Image.open(dest) as img:
            img.thumbnail((_BAND_MOSAIC_TARGET_DIM, _BAND_MOSAIC_TARGET_DIM), Image.Resampling.LANCZOS)
            img.save(dest, format="PNG", optimize=True)
    except Exception:
        # If post-resize fails leave the original file in place.
        logger.exception("band mosaic post-resize failed (kept original)")


def render_histogram_json(vendable, sensor_type: str, dest: Path) -> None:
    """Distribution of pixel values as JSON bins.

    Shape:
        {
          "sensor_type": "...",
          "kind": "thermal" | "hyperspectral_summary",
          "bins":  [edge0, edge1, ..., edgeN],   # length N+1
          "counts": [c0, c1, ..., c_{N-1}],
          "stats":  { "mean", "std", "min", "max", "p2", "p50", "p98", "valid_pct" }
        }

    Hyperspectral case is a *summary* histogram across all bands —
    useful as a single chart, but not a per-band breakdown. A per-band
    view can come later if needed.
    """
    if sensor_type in ("landsat9", "hotsat1"):
        plane = np.asarray(
            np.squeeze(vendable.normalized_thermal_cube), dtype=np.float32
        )
        kind = "thermal"
        values = plane[np.isfinite(plane)]
    else:
        cube = _hyperspectral_cube(vendable)
        kind = "hyperspectral_summary"
        # Subsample to keep histogram cost bounded for big PRISMA cubes.
        # 1M pixels gives a smooth-looking histogram without scanning
        # ~290 M values (PRISMA: 239 bands × 1210 × 1219).
        flat = cube.reshape(-1)
        if flat.size > 1_000_000:
            rng = np.random.default_rng(seed=0)
            idx = rng.integers(0, flat.size, size=1_000_000)
            flat = flat[idx]
        values = flat[np.isfinite(flat)]

    payload: dict[str, Any] = {
        "sensor_type": sensor_type,
        "kind": kind,
    }
    if values.size == 0:
        payload["bins"] = []
        payload["counts"] = []
        payload["stats"] = {}
    else:
        # Robust bin range: 1st–99th percentile, falls back to min/max.
        lo, hi = np.percentile(values, [1, 99])
        if hi <= lo:
            lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        counts, edges = np.histogram(values, bins=64, range=(lo, hi))
        payload["bins"] = [float(x) for x in edges.tolist()]
        payload["counts"] = [int(x) for x in counts.tolist()]
        payload["stats"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p2": float(np.percentile(values, 2)),
            "p50": float(np.percentile(values, 50)),
            "p98": float(np.percentile(values, 98)),
            "valid_pct": float(values.size) * 100.0 / float(max(1, _value_count(vendable, sensor_type))),
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")


def _value_count(vendable, sensor_type: str) -> int:
    """Total pixel-count across the cube (incl. NaN/invalid). Denominator
    for `valid_pct` in the histogram stats."""
    if sensor_type in ("landsat9", "hotsat1"):
        plane = np.asarray(np.squeeze(vendable.normalized_thermal_cube))
        return int(plane.size)
    cube = _hyperspectral_cube(vendable)
    return int(cube.size)


# ---------------------------------------------------------------------
# Top-level entry — called from scene_onboard
# ---------------------------------------------------------------------


def render_all(
    vendable,
    sensor_type: str,
    artifacts_root: Path,
    scene_id_str: str,
) -> dict[str, str]:
    """Render the full set of visualizations for a single scene.

    Returns a dict mapping `kind → relative artifact path`. Caller
    persists the dict (or its `color` entry) onto the Scene row;
    everything else is discoverable by listing the dir.

    Sensor dispatch:
      - landsat9 | hotsat1 → color.png + histogram.json. NIR/SWIR/NDVI/
        band-mosaic skipped (single-band thermal scenes).
      - prisma | enmap | aviris_ng → color.png + nir.png + swir.png +
        ndvi.png + band_mosaic.png + histogram.json.

    Each step is wrapped in its own try/except so a failure in one
    visualization (e.g. band_mosaic on a degenerate cube) doesn't take
    out the others. The caller still gets back the kinds that succeeded.
    """
    out_dir = artifacts_root / "scenes" / scene_id_str / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _try(kind: str, fn) -> None:
        try:
            fn()
            written[kind] = f"scenes/{scene_id_str}/visualizations/{kind}.{'json' if kind == 'histogram' else 'png'}"
        except Exception:
            logger.exception("visualization '%s' failed for scene %s", kind, scene_id_str)

    _try("color", lambda: render_color_thumbnail(vendable, sensor_type, out_dir / "color.png"))

    if sensor_type in ("prisma", "enmap", "aviris_ng"):
        _try("nir", lambda: render_nir_composite(vendable, out_dir / "nir.png"))
        _try("swir", lambda: render_swir_composite(vendable, out_dir / "swir.png"))
        _try("ndvi", lambda: render_ndvi(vendable, out_dir / "ndvi.png"))
        _try("band_mosaic", lambda: render_band_mosaic(vendable, out_dir / "band_mosaic.png", sensor=sensor_type))

    _try("histogram", lambda: render_histogram_json(vendable, sensor_type, out_dir / "histogram.json"))

    return written
