"""In-process compute backend for the anomaly_detection_prep preview endpoint.

Kept in its own module so the route handler in ``actions.py`` stays
focused on auth + path resolution and this file owns the math.

Per-Apply call the user is doing:

  1. Resolve the composite_score raster + (optional) ground-truth mask.
  2. Convert their slider choice (percentile or absolute) into an
     absolute threshold value over the composite.
  3. Binarise the composite: mask = composite >= threshold.
  4. Optionally dilate the binary mask with a square structuring
     element of the user-chosen kernel size.
  5. Compute precision / recall / F1 against the GT raster when present.
  6. Render the binary mask as a transparent-background PNG (white for
     anomaly, transparent for non-anomaly + invalid pixels).

The composite raster is cached in this module's process memory with a
small LRU so repeat Apply calls on the same action skip the disk read.
Cache key is the absolute composite_score.tif path; entry holds the
ndarray + the valid-pixel mask + the path's mtime so we invalidate if
the worker ever overwrites the file in place.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("allotrope.api.anomaly_detection_preview")


# ---------------------------------------------------------------------------
# In-process LRU cache for the composite raster
# ---------------------------------------------------------------------------

# Cap small — composite rasters can be tens of MB and we don't want the
# api process to balloon when a user opens many prep actions. 5 deep is
# plenty for a single user's exploration flow.
_CACHE_MAX_ENTRIES = 5


@dataclass
class _CachedComposite:
    """Cached payload per (action_id, composite_path) entry."""

    composite: "object"      # np.ndarray (H, W) float32 (NaN where invalid)
    valid_mask: "object"     # np.ndarray (H, W) bool
    finite_values: "object"  # np.ndarray (n_valid,) float32, sorted ascending
    mtime: float


_cache: "OrderedDict[str, _CachedComposite]" = OrderedDict()
_cache_lock = threading.Lock()


def _load_composite(composite_path: Path) -> _CachedComposite:
    """Load (and cache) the composite raster. Repeat calls on the same
    path within the api's lifetime go through the LRU."""
    import numpy as np
    import rasterio

    key = str(composite_path.resolve())
    mtime = composite_path.stat().st_mtime

    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None and entry.mtime == mtime:
            # Move to most-recently-used end.
            _cache.move_to_end(key)
            return entry

    # Cache miss — read from disk outside the lock so concurrent
    # readers don't serialise on the same I/O.
    with rasterio.open(composite_path) as src:
        composite = src.read(1).astype(np.float32, copy=False)
    valid_mask = np.isfinite(composite)
    finite_values = composite[valid_mask]
    # Sort once so percentile→absolute conversion is O(log N).
    finite_values = np.sort(finite_values)

    new_entry = _CachedComposite(
        composite=composite,
        valid_mask=valid_mask,
        finite_values=finite_values,
        mtime=mtime,
    )
    with _cache_lock:
        _cache[key] = new_entry
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)
    logger.info(
        "composite cache miss — loaded %s shape=%s valid=%d",
        composite_path, composite.shape, int(valid_mask.sum()),
    )
    return new_entry


# ---------------------------------------------------------------------------
# Public compute entrypoint
# ---------------------------------------------------------------------------


@dataclass
class PreviewResult:
    """What the api route returns to the frontend."""

    threshold_absolute: float
    threshold_percentile: float
    dilation_kernel: int
    n_anomalous: int
    n_kept: int
    metrics: Optional[dict]
    mask_png_bytes: bytes


def compute_preview(
    *,
    composite_path: Path,
    threshold: float,
    threshold_mode: str,
    dilation_kernel: int,
    gt_mask: Optional["object"] = None,
) -> PreviewResult:
    """Run one Apply click and return everything the viewer needs.

    Args:
        composite_path: absolute path to ``composite_score.tif`` on
            disk (under the prep action's output dir).
        threshold: slider value. Interpretation depends on
            ``threshold_mode``: in ``"percentile"`` mode it's a 0–100
            value where 95 means "the top 5% of pixels"; in
            ``"absolute"`` mode it's the raw composite value.
        threshold_mode: ``"percentile"`` or ``"absolute"``.
        dilation_kernel: 0 (no dilation) or an odd integer; the binary
            mask is dilated with a square structuring element of that
            side length before metrics are computed.
        gt_mask: optional (H, W) boolean ndarray matching the composite
            shape — ground-truth anomaly mask. When present, metrics are
            computed.
    """
    import numpy as np

    if dilation_kernel < 0 or (dilation_kernel != 0 and dilation_kernel % 2 == 0):
        raise ValueError(
            f"dilation_kernel must be 0 or an odd positive integer; got "
            f"{dilation_kernel}"
        )
    if threshold_mode not in ("percentile", "absolute"):
        raise ValueError(
            f"threshold_mode must be 'percentile' or 'absolute'; got "
            f"{threshold_mode!r}"
        )

    entry = _load_composite(composite_path)
    composite = entry.composite        # (H, W) float32
    valid_mask = entry.valid_mask      # (H, W) bool
    finite_values = entry.finite_values  # sorted ascending

    # --- 1. Resolve threshold to absolute -------------------------------
    if threshold_mode == "percentile":
        if not (0.0 <= threshold <= 100.0):
            raise ValueError(
                f"percentile threshold must be in [0, 100]; got {threshold}"
            )
        if finite_values.size == 0:
            absolute_threshold = float("inf")
            percentile = float(threshold)
        else:
            # Slider semantics: value = "flag pixels above the Xth
            # percentile of the composite distribution." So slider=0
            # flags everything above p0 (i.e. all valid pixels),
            # slider=99 flags only the top 1%. This matches the user's
            # intuition: drag left = include more pixels.
            quantile_pct = max(0.0, min(100.0, float(threshold)))
            absolute_threshold = float(
                np.percentile(finite_values, quantile_pct)
            )
            percentile = float(threshold)
    else:
        absolute_threshold = float(threshold)
        if finite_values.size == 0:
            percentile = 0.0
        else:
            # Inverse of the percentile path: derive what percentile
            # this absolute value sits at within the composite
            # distribution. ``searchsorted(..., 'left')`` returns the
            # count of finite values strictly below the threshold; that
            # count over total = the threshold's percentile rank.
            n_below = int(np.searchsorted(
                finite_values, absolute_threshold, side="left"
            ))
            percentile = float(
                max(0.0, min(100.0, n_below / finite_values.size * 100.0))
            )

    # --- 2. Binarise ----------------------------------------------------
    # NaN >= x is False in numpy — invalid pixels naturally end up False.
    raw_mask = composite >= np.float32(absolute_threshold)
    raw_mask &= valid_mask  # defensive — same effect, makes intent obvious

    # --- 3. Optional dilation -------------------------------------------
    if dilation_kernel and dilation_kernel > 1:
        binary = _square_dilate(raw_mask, dilation_kernel)
        # Constrain back to valid pixels — dilation must not bleed into
        # off-swath / no-data regions.
        binary &= valid_mask
    else:
        binary = raw_mask

    n_anomalous = int(binary.sum())
    n_kept = int(valid_mask.sum())

    # --- 4. Metrics (if GT supplied) ------------------------------------
    metrics: Optional[dict] = None
    if gt_mask is not None:
        if gt_mask.shape != composite.shape:
            raise ValueError(
                f"gt_mask shape {gt_mask.shape} does not match composite "
                f"shape {composite.shape}"
            )
        gt_bool = np.asarray(gt_mask, dtype=bool) & valid_mask
        tp = int((binary & gt_bool).sum())
        fp = int((binary & (~gt_bool) & valid_mask).sum())
        fn = int(((~binary) & gt_bool).sum())
        tn = int(((~binary) & (~gt_bool) & valid_mask).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_gt_positives": int(gt_bool.sum()),
        }

    # --- 5. Render mask PNG --------------------------------------------
    mask_png_bytes = _render_mask_png(binary, valid_mask)

    return PreviewResult(
        threshold_absolute=float(absolute_threshold),
        threshold_percentile=float(percentile),
        dilation_kernel=int(dilation_kernel),
        n_anomalous=n_anomalous,
        n_kept=n_kept,
        metrics=metrics,
        mask_png_bytes=mask_png_bytes,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


# Long-edge cap for the mask PNG. Matches the convention used by
# anomaly_scoring + the prep worker's composite preview.
_PNG_MAX_EDGE = 4096


def _square_dilate(mask: "object", kernel_size: int) -> "object":
    """Morphological dilation by a square structuring element.

    Implemented in pure numpy so the api container doesn't need to
    ship scipy (the worker image does, but the api is intentionally
    lean). For a square kernel the operation is separable: a pixel is
    ``True`` after dilation iff any pixel within ``k//2`` rows AND any
    pixel within ``k//2`` cols is ``True`` in the input.

    Result is equivalent to ``scipy.ndimage.binary_dilation`` with
    ``structure=np.ones((k, k), bool)``.

    Args:
        mask: (H, W) bool ndarray.
        kernel_size: odd positive integer (we validate this upstream).

    Returns:
        (H, W) bool ndarray.
    """
    import numpy as np

    if kernel_size <= 1:
        return np.asarray(mask, dtype=bool)
    half = kernel_size // 2
    src = np.asarray(mask, dtype=bool)

    # Row sweep: each output pixel is True if any of the (2*half+1)
    # neighbouring rows is True at that column. Implemented as a sum
    # of shifted copies — bounded total cost is k * H*W bool reads.
    rows_acc = src.copy()
    for shift in range(1, half + 1):
        # Shift up by `shift` rows. Pixels beyond the top edge are
        # filled with False, which is the morphological convention
        # for "no contribution from outside the image".
        up = np.zeros_like(src)
        up[:-shift] = src[shift:]
        # Shift down by `shift` rows.
        down = np.zeros_like(src)
        down[shift:] = src[:-shift]
        rows_acc |= up
        rows_acc |= down

    # Column sweep, same trick on rows_acc.
    out = rows_acc.copy()
    for shift in range(1, half + 1):
        left = np.zeros_like(rows_acc)
        left[:, :-shift] = rows_acc[:, shift:]
        right = np.zeros_like(rows_acc)
        right[:, shift:] = rows_acc[:, :-shift]
        out |= left
        out |= right

    return out


def _render_mask_png(binary: "object", valid_mask: "object") -> bytes:
    """Render a binary anomaly mask as a transparent-background PNG.

    - Anomalous pixels   → white, fully opaque
    - Non-anomalous valid → fully transparent
    - Invalid pixels      → fully transparent

    This lets the frontend overlay it directly on top of the RGB or
    composite panel.
    """
    import numpy as np
    from PIL import Image

    h, w = binary.shape
    scale = min(1.0, _PNG_MAX_EDGE / max(h, w))
    if scale < 1.0:
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        binary_ds = (
            np.asarray(
                Image.fromarray(binary.astype(np.uint8) * 255).resize(
                    (new_w, new_h), Image.NEAREST
                )
            )
            > 127
        )
    else:
        binary_ds = binary

    h_ds, w_ds = binary_ds.shape
    rgba = np.zeros((h_ds, w_ds, 4), dtype=np.uint8)
    rgba[..., 0:3] = 255  # white
    rgba[..., 3] = binary_ds.astype(np.uint8) * 255  # alpha = 255 where anomalous

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GT loading helper
# ---------------------------------------------------------------------------


def load_gt_mask_for_action(
    *,
    data_root: Path,
    scene_id: str,
    annotation_id: str,
    expected_shape: tuple,
) -> Optional["object"]:
    """Best-effort load of the ground-truth raster for the scene's
    bound annotation.

    Returns a (H, W) boolean ndarray matching ``expected_shape`` or
    ``None`` if no rasterised GT was materialised for the annotation.

    Path convention mirrors the worker's ``_load_annotation_gt`` in
    ``_anomaly_scoring_run.py``:

        <data_root>/scenes/<scene_id>/annotations/<annotation_uuid>/*.tif

    The first .tif under that dir is treated as the GT raster — same
    semantics the worker uses (annotation types that don't rasterise
    don't end up here at all). We re-implement the lookup here
    instead of importing from the worker module so the api package
    stays free of worker-only deps.
    """
    import numpy as np
    import rasterio

    from .wireformat import parse_prefixed_id

    try:
        raw_id = parse_prefixed_id("annotation", annotation_id)
    except Exception:
        logger.warning("invalid annotation_id wire format: %r", annotation_id)
        return None

    annotation_dir = (
        data_root / "scenes" / str(scene_id) / "annotations" / str(raw_id)
    )
    if not annotation_dir.is_dir():
        logger.info(
            "annotation %s has no rasterised .tif (annotation dir missing: %s)",
            annotation_id, annotation_dir,
        )
        return None

    tifs = sorted(annotation_dir.glob("*.tif"))
    if not tifs:
        logger.info(
            "annotation %s has no rasterised .tif under %s",
            annotation_id, annotation_dir,
        )
        return None

    with rasterio.open(tifs[0]) as src:
        raw = src.read(1)
    if raw.shape != expected_shape:
        logger.warning(
            "annotation %s raster shape %s does not match composite "
            "shape %s — skipping metrics.",
            annotation_id, raw.shape, expected_shape,
        )
        return None

    return (raw > 0)
