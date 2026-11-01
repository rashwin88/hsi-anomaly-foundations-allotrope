"""
scene_segmentation, method "indradhanu_v2": Indradhanu V2's segmentation head as its own step.

Called from `scene_segmentation.run` when the config's method is "indradhanu_v2"; its keep_mask
feeds anomaly_scoring, which then skips V2's internal head (`_anomaly_scoring_run.upstream_v2_segmentation`).
Design: docs/lld/segmentation-v2.md.

Why: until 2026-11-01 the head ran only inside anomaly_scoring, so the user could neither see the
classes nor choose which were excluded. Running it here with the SAME calls scoring makes
(resolve -> read_bands -> segment_full_scene on the band-filtered vendable) keeps the mask
identical to the one scoring used to compute for itself.

Gotchas:
  - class_map.tif codes follow the inferencer's CLASS_NAMES (0 land .. 4 snow), 255 outside valid
    (segment_full_scene itself returns 0 there, which reads as land).
  - Thin cloud is not a class code: with thin_cloud_is_cloud=False it is only a display layer
    (bit 0x10 in class_codes.bin) and its pixels stay in keep_mask.
  - The model needs native bands (band_filter_apply use_common_wavelength_grid=false).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from ._anomaly_scoring_render import _make_rgb, _save_png_array, _save_raster_2d

# Codes in class_map.tif. Mirrors SpectralStemInferencer.CLASS_NAMES, which run() asserts, so
# this module and its tests need no torch.
CLASS_NAMES = ("land", "water", "cloud", "shadow", "snow")
INVALID_CODE = 255
# class_codes.bin (the viewer's per-pixel data): the class code, OR'd with this bit where the
# head's thin-cloud output fires and the pixel is not cloud. 255 = outside valid.
HAZE_BIT = 0x10
# Fixed and mutually distinct; the viewer reads them from the summary's legend.
CLASS_COLORS = {
    "land": "#c8a165",
    "water": "#1e6bff",
    "cloud": "#ff2fb3",
    "shadow": "#7b2cff",
    "snow": "#00e5ff",
    "thin_cloud": "#ffd400",
}
CLASS_LABELS = {"land": "Land", "water": "Water", "cloud": "Cloud", "shadow": "Shadow",
                "snow": "Snow", "thin_cloud": "Thin cloud / haze"}
# Overlay opacity of a class colour over the RGB in class_map_overlay.png; land is not tinted.
_OVERLAY_ALPHA = 0.5
_PREVIEW_MAX_EDGE = 768


def class_map_with_invalid(cls: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(H, W) uint8 class codes with INVALID_CODE outside valid."""
    out = cls.astype(np.uint8)
    out[~valid.astype(bool)] = INVALID_CODE
    return out


def keep_mask_from_class_map(class_map: np.ndarray, exclude_classes) -> np.ndarray:
    """(H, W) uint8: 1 = valid and not in exclude_classes; 0 = excluded or outside valid."""
    codes = [CLASS_NAMES.index(c) for c in exclude_classes]
    return ((class_map != INVALID_CODE) & ~np.isin(class_map, codes)).astype(np.uint8)


def class_summary(class_map: np.ndarray, exclude_classes, haze: np.ndarray | None = None) -> dict[str, Any]:
    """Pixel counts and % of valid per class, the excluded and kept %, and the legend the viewer
    draws. `haze` (H, W) bool: thin-cloud pixels outside cloud, or None when thin cloud is cloud."""
    valid = class_map != INVALID_CODE
    n_valid = int(valid.sum())
    pct = lambda n: round(100.0 * n / n_valid, 3) if n_valid else 0.0   # noqa: E731
    counts = {name: int((class_map == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    keys = list(CLASS_NAMES)
    if haze is not None:
        counts["thin_cloud"] = int((haze & valid).sum())
        keys.append("thin_cloud")
    kept = int(keep_mask_from_class_map(class_map, exclude_classes).sum())
    legend = [{"key": k, "label": CLASS_LABELS[k], "color": CLASS_COLORS[k],
               "code": CLASS_NAMES.index(k) if k in CLASS_NAMES else None,
               "pixels": counts[k], "pct_of_valid": pct(counts[k]), "excluded": k in exclude_classes}
              for k in keys]
    return {
        "pixel_counts": {**counts, "spatial_valid": n_valid, "excluded": n_valid - kept, "kept": kept},
        "class_pct_of_valid": {k: pct(counts[k]) for k in keys},
        "excluded_pct_of_valid": pct(n_valid - kept),
        "kept_pct_of_valid": pct(kept),
        "legend": legend,
    }


def class_codes(class_map: np.ndarray, haze: np.ndarray | None) -> np.ndarray:
    """class_codes.bin's content: class_map with HAZE_BIT set on haze pixels."""
    out = class_map.copy()
    if haze is not None:
        out[haze & (class_map != INVALID_CODE)] |= HAZE_BIT
    return out


def _rgb_of(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def _save_rgba(rgba: np.ndarray, path: Path, max_edge: int = 4096) -> None:
    from PIL import Image

    img = Image.fromarray(rgba)
    if max(img.size) > max_edge:
        s = max_edge / max(img.size)
        img = img.resize((max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s))), Image.NEAREST)
    img.save(path, format="PNG", optimize=True)


def _render(out: Path, class_map: np.ndarray, rgb: np.ndarray) -> None:
    """class_map.png (colours, transparent outside valid), class_map_overlay.png and preview.png."""
    from PIL import Image

    colors = np.zeros(class_map.shape + (3,), np.float32)
    overlay = rgb.astype(np.float32).copy()
    for i, name in enumerate(CLASS_NAMES):
        sel = class_map == i
        colors[sel] = _rgb_of(CLASS_COLORS[name])
        if name != "land":
            overlay[sel] = (1 - _OVERLAY_ALPHA) * overlay[sel] + _OVERLAY_ALPHA * _rgb_of(CLASS_COLORS[name])
    alpha = np.where(class_map == INVALID_CODE, 0, 255).astype(np.uint8)[..., None]
    _save_rgba(np.concatenate([(colors * 255).astype(np.uint8), alpha], axis=-1), out / "class_map.png")
    _save_png_array(overlay, out / "class_map_overlay.png")
    small = Image.open(out / "class_map_overlay.png")
    small.thumbnail((_PREVIEW_MAX_EDGE, _PREVIEW_MAX_EDGE))
    small.save(out / "preview.png", format="PNG")


NOT_REFLECTANCE_ERROR = (
    "V2 segmentation needs reflectance; this scene is {decision} - use method 'indices' or fix the scale at upload"
)


def upstream_not_reflectance(ctx: Any) -> bool:
    """True when the upstream band filter vended a not-reflectance cube (generic_hsi DN / radiance).

    Read from its summary.json (`scene_units`), so the check costs no pickle load.
    """
    summary = ctx.resolve_action_output(ctx.configuration["input_band_filter_output_id"]) / "summary.json"
    try:
        return json.loads(summary.read_text()).get("scene_units") == "not_reflectance"
    except (OSError, ValueError):
        return False


def run(ctx: Any) -> None:
    """Segment the band-filtered scene with the model's head and write the class map, keep_mask,
    the PNGs, class_codes.bin, summary.json and diagnostics.json under ctx.output_dir."""
    import torch

    from app.foundation_models.inferencers.inferencer_factory import get_inferencer
    from app.models.training.inference_config import InferenceConfig

    from ..config import settings
    from ..foundation_models.resolver import list_catalog
    from ._anomaly_scoring_run import _check_band_grid

    cfg = ctx.configuration
    upstream_output = cfg["input_band_filter_output_id"]
    exclude = list(cfg.get("exclude_classes", ["cloud"]))
    thin_is_cloud = bool(cfg.get("thin_cloud_is_cloud", False))
    codename = cfg.get("model_codename", "Indradhanu V2")
    t_start = time.time()
    if upstream_not_reflectance(ctx):
        raise ValueError(NOT_REFLECTANCE_ERROR.format(decision="not reflectance (DN or radiance, per its pre-flight report)"))

    ctx.on_step("load_filtered_vendable")
    pickle_path = ctx.resolve_action_output(upstream_output) / "filtered_vendable.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(f"upstream band_filter_apply output is missing filtered_vendable.pkl: {pickle_path}")
    with pickle_path.open("rb") as f:
        vendable = pickle.load(f)
    if getattr(vendable, "units", None) == "not_reflectance":      # a band filter without summary.json
        raise ValueError(NOT_REFLECTANCE_ERROR.format(decision="not reflectance"))
    cube = vendable.normalized_hyperspectral_cube
    wavelengths = np.asarray(vendable.band_cw_order, dtype=np.float64)
    valid = vendable.validity_cube.sum(axis=0) > 0

    ctx.on_step(f"model={codename} · load_checkpoint")
    m = {x.codename.lower(): x for x in list_catalog(Path(settings.models_dir))}.get(codename.strip().lower())
    if m is None:
        raise ValueError(f"unknown codename at run time: {codename!r}")
    _check_band_grid(codename, m.band_grid, wavelengths)
    t0 = time.time()
    inferencer = get_inferencer(InferenceConfig(
        foundation_model_name=m.foundation_model_name, model_config=m.model_config,
        checkpoint_path=m.checkpoint_abs_path, patch_size=m.default_patch_size, stride=m.default_stride,
        inference_batch_size=m.default_batch_size, pixel_stats_path=m.pixel_stats_abs_path))
    load_s = time.time() - t0
    if tuple(inferencer.CLASS_NAMES) != CLASS_NAMES:
        raise RuntimeError(f"inferencer classes {inferencer.CLASS_NAMES} != {CLASS_NAMES}")

    ctx.on_step(f"model={codename} · segment")
    t0 = time.time()
    model_cube, model_wl, _ = inferencer.read_bands(cube, wavelengths)
    with torch.no_grad():
        cls, prob = inferencer.segment_full_scene(model_cube, valid, thin_cloud=thin_is_cloud)
    segment_s = time.time() - t0
    class_map = class_map_with_invalid(cls, valid)
    haze = None
    if not thin_is_cloud:
        haze = (prob[inferencer.SEG_CLASSES.index("thin_cloud")] > 0.5) & (class_map != CLASS_NAMES.index("cloud"))
    keep_mask = keep_mask_from_class_map(class_map, exclude)

    ctx.on_step("write_rasters")
    out: Path = ctx.output_dir
    _save_raster_2d(class_map, out / "class_map.tif", "uint8")
    _save_raster_2d(keep_mask, out / "keep_mask.tif", "uint8")
    class_codes(class_map, haze).tofile(out / "class_codes.bin")
    rgb = _make_rgb(cube, wavelengths, valid)
    _save_png_array(rgb, out / "rgb.png")
    _render(out, class_map, rgb)

    summary = {
        "method": "indradhanu_v2",
        "action_id": f"action_{ctx.action_id}",
        "upstream_band_filter_output_id": upstream_output,
        "model_codename": m.codename,
        "model_architecture": m.architecture,
        "model_checkpoint": m.checkpoint_filename,
        "exclude_classes": exclude,
        "thin_cloud_is_cloud": thin_is_cloud,
        "class_names": list(CLASS_NAMES),
        "invalid_code": INVALID_CODE,
        "haze_bit": HAZE_BIT,
        "scene_shape": [int(class_map.shape[0]), int(class_map.shape[1])],
        "bands_read": int(len(model_wl)),
        **class_summary(class_map, exclude, haze),
        "device": str(inferencer.device),
        "load_seconds": round(load_s, 3),
        "segment_seconds": round(segment_s, 3),
        "runtime_seconds": round(time.time() - t_start, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    ctx.on_step("done")
