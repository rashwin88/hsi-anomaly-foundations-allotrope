"""Worker-side implementation of the `cloud_mask` action.

Loads the scene's onboarding vendable, fits the
`B10AdaptiveCloudMasker` GMM against the B10 (Celsius) brightness-
temperature cube, predicts a binary cloud mask, and writes both the
cloud mask and a downstream-ready `keep_mask` plus a preview overlay.

Recipe:
  1. Load thermal vendable from scenes/<scene_id>/vendable/vendable.pkl
  2. Reduce to (H, W) Celsius array; build np.ma masked array using validity
  3. Configure + train + predict B10AdaptiveCloudMasker on the masked array
  4. Write cloud_mask.tif (uint8, 1 = cloud)
  5. Write keep_mask.tif  (uint8, 1 = valid ∧ ¬cloud)
  6. Render preview.png   (B10 grayscale + cyan cloud overlay)
  7. Write summary.json + diagnostics.json
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("allotrope.action_types.cloud_mask")


_SUMMARY_FILENAME = "summary.json"
_DIAGNOSTICS_FILENAME = "diagnostics.json"
_PREVIEW_FILENAME = "preview.png"
_CLOUD_FILENAME = "cloud_mask.tif"
_KEEP_FILENAME = "keep_mask.tif"
_PNG_MAX_EDGE = 4096


def run(ctx: Any) -> None:
    import numpy as np

    from app.statistical_models.b10_adaptive_cloud_masker import (
        B10AdaptiveCloudMasker,
    )

    from ..foundation_models.resolver import sensor_family

    cfg = ctx.configuration
    sampling_ratio = float(cfg.get("sampling_ratio", 0.10))

    # --- 1. Sensor sanity --------------------------------------------
    if sensor_family(ctx.sensor_type) != "thermal":
        raise ValueError(
            f"cloud_mask is thermal-only; got sensor {ctx.sensor_type!r}"
        )

    # --- 2. Load the vendable ----------------------------------------
    ctx.on_step("load_onboarding_vendable")
    pickle_path = (
        ctx.data_dir
        / "scenes"
        / str(ctx.scene_id)
        / "vendable"
        / "vendable.pkl"
    )
    if not pickle_path.exists():
        raise FileNotFoundError(f"vendable pickle missing: {pickle_path}")
    with pickle_path.open("rb") as f:
        vendable = pickle.load(f)

    thermal = vendable.normalized_thermal_cube
    if thermal.ndim == 2:
        thermal_2d = thermal.astype(np.float32, copy=False)
    elif thermal.ndim == 3 and thermal.shape[0] == 1:
        thermal_2d = thermal[0].astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"unexpected thermal cube shape {thermal.shape}; "
            "expected (H, W) or (1, H, W)"
        )

    # IMPORTANT: the Landsat onboarding stage in
    # `app/utils/dataset_builder/landsat_dataset_builder.py` already runs
    # `B10AdaptiveCloudMasker` once and AND-folds its result into
    # `vendable.validity_cube`. So `validity_cube` is *cloud-cleaned*;
    # the GMM would see no cold pixels here and emit a no-op mask.
    #
    # `pure_validity_mask` (when present) is the raw scene validity
    # **before** the onboarding cloud step — this is what we want to
    # feed the masker so it can find clouds in the first place.
    pv = getattr(vendable, "pure_validity_mask", None)
    if pv is None:
        logger.warning(
            "vendable has no pure_validity_mask; falling back to "
            "validity_cube. The onboarding step already cloud-masked "
            "this — expect 0 cloud pixels at predict time."
        )
        v = vendable.validity_cube
    else:
        v = pv

    if v.ndim == 2:
        validity_2d = v.astype(bool, copy=False)
    elif v.ndim == 3:
        validity_2d = (v.sum(axis=0) > 0)
    else:
        raise ValueError(f"unexpected validity cube shape {v.shape}")

    h, w = thermal_2d.shape
    # The masker expects values in Celsius; the vendable already is.
    # Build a masked array so the GMM training only sees valid pixels.
    masked_b10 = np.ma.masked_array(
        thermal_2d,
        mask=~validity_2d,
        copy=False,
    )

    # --- 3. Fit + predict --------------------------------------------
    ctx.on_step("fit_gmm")
    t0 = time.time()
    masker = B10AdaptiveCloudMasker()
    masker.configure(sampling_ratio=sampling_ratio)
    masker.train(masked_b10)
    fit_seconds = time.time() - t0

    ctx.on_step("predict")
    t0 = time.time()
    response = masker.predict(masked_b10)
    predict_seconds = time.time() - t0
    cloud_mask = response.cloud_mask.astype(np.uint8, copy=False)
    if cloud_mask.shape != (h, w):
        raise RuntimeError(
            f"cloud_mask shape {cloud_mask.shape} != expected ({h}, {w})"
        )

    # --- 4. Compose keep_mask = validity ∧ ¬cloud --------------------
    keep_mask = (validity_2d & (cloud_mask == 0)).astype(np.uint8)

    # --- 5. Persist rasters + preview --------------------------------
    output_dir: Path = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx.on_step("write_rasters")
    _save_raster(cloud_mask, output_dir / _CLOUD_FILENAME)
    _save_raster(keep_mask, output_dir / _KEEP_FILENAME)

    ctx.on_step("render_preview")
    try:
        _render_preview(
            thermal_2d=thermal_2d,
            validity_2d=validity_2d,
            cloud_mask=cloud_mask,
            out_path=output_dir / _PREVIEW_FILENAME,
        )
    except Exception as exc:
        logger.warning("preview render failed (non-fatal): %s", exc)

    # --- 6. Summary + diagnostics ------------------------------------
    valid_count = int(validity_2d.sum())
    cloud_count = int(cloud_mask.sum())
    keep_count = int(keep_mask.sum())

    summary: dict[str, Any] = {
        "scene_shape": [h, w],
        "valid_pixels": valid_count,
        "cloud_pixels": cloud_count,
        "kept_pixels": keep_count,
        "cloud_pct_of_valid": round(
            100.0 * cloud_count / max(valid_count, 1), 3
        ),
        "kept_pct_of_valid": round(
            100.0 * keep_count / max(valid_count, 1), 3
        ),
        "sampling_ratio": sampling_ratio,
        "n_components": int(masker.n_comp or 0),
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
    }
    (output_dir / _SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))

    anchors = (
        masker.anchors.flatten().tolist()
        if masker.anchors is not None
        else None
    )
    probe = masker.probe.tolist() if masker.probe is not None else None
    cluster_means = (
        masker.model.means_.flatten().tolist() if masker.model is not None else None
    )
    diagnostics: dict[str, Any] = {
        **summary,
        "gmm_anchors_celsius": anchors,
        "probe_percentiles_celsius": {
            "p2": probe[0] if probe else None,
            "p8": probe[1] if probe else None,
            "p50": probe[2] if probe else None,
            "p92": probe[3] if probe else None,
            "p98": probe[4] if probe else None,
        }
        if probe
        else None,
        "gmm_cluster_means_celsius": cluster_means,
        "sample_count": int(masker.sample_count or 0),
    }
    (output_dir / _DIAGNOSTICS_FILENAME).write_text(
        json.dumps(diagnostics, indent=2)
    )

    ctx.on_step("done")


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    p = output_dir / _SUMMARY_FILENAME
    if p.exists():
        return json.loads(p.read_text())
    d = output_dir / _DIAGNOSTICS_FILENAME
    if d.exists():
        return json.loads(d.read_text())
    return {"error": "diagnostics_missing"}


def preview(ctx: Any, output_dir: Any) -> Any:
    p = output_dir / _PREVIEW_FILENAME
    return p if p.exists() else None


# --- Internals ---------------------------------------------------------


def _save_raster(arr: Any, path: Any) -> None:
    """Single-band uint8 GeoTIFF writer (no CRS / transform)."""
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    a = np.asarray(arr).astype(np.uint8, copy=False)
    if a.ndim != 2:
        raise ValueError(f"expected (H, W); got {a.shape}")
    h, w = a.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": a.dtype.name,
        "compress": "lzw",
        "transform": Affine.identity(),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a, 1)


def _render_preview(
    *,
    thermal_2d: Any,
    validity_2d: Any,
    cloud_mask: Any,
    out_path: Path,
) -> None:
    """B10 grayscale + cyan cloud overlay at scene resolution.

    Validity-masked 2/98 percentile stretch on the thermal channel,
    semi-transparent cyan tint where cloud_mask == 1, downsampled to
    the configured PNG max edge.
    """
    import numpy as np
    from PIL import Image

    valid = validity_2d.astype(bool)
    vals = thermal_2d[valid] if valid.any() else thermal_2d.ravel()
    if vals.size:
        p2, p98 = np.percentile(vals, [2, 98])
        if p98 - p2 > 1e-6:
            stretched = np.clip((thermal_2d - p2) / (p98 - p2), 0.0, 1.0)
        else:
            stretched = np.zeros_like(thermal_2d)
    else:
        stretched = np.zeros_like(thermal_2d)
    base = (stretched * 255.0).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)              # (H, W, 3)
    rgb[~valid] = 0

    # Cyan tint where cloud (R=80, G=200, B=240) blended at 55 % alpha.
    cloud = cloud_mask.astype(bool)
    if cloud.any():
        tint = np.array([80, 200, 240], dtype=np.uint8)
        alpha = 0.55
        blended = (
            (1.0 - alpha) * rgb[cloud].astype(np.float32)
            + alpha * tint.astype(np.float32)
        ).astype(np.uint8)
        rgb[cloud] = blended

    h, w = base.shape
    img = Image.fromarray(rgb)
    if max(h, w) > _PNG_MAX_EDGE:
        scale = _PNG_MAX_EDGE / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
    img.save(out_path, format="PNG", optimize=True)
