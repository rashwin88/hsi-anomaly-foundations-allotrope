"""Worker-side spectral namer stage of ``spectral_library_match``: name the anomaly objects first.

Called from ``_spectral_library_match_run.run`` before SAM and family matching, on the same
native onboarding vendable. Loads the checkpoint the ``spectral_namer`` manifest points at
(``foundation_models.resolver.resolve_spectral_namer``), names every pixel of the committed mask
(``app.spectral_match.namer.infer``), groups them into objects
(``app.spectral_match.namer.objects``) and writes the ``namer_*`` files and the summary block.

Why it runs first: matching every anomaly pixel against man-made libraries gives natural
false alarms man-made names. The namer's level 1 beats the classical ACE call by 0.14-0.20
man-made recall at fill 0.25 (research/experiments/spectral-namer/N001b/README.md), so its
natural objects are labelled natural and only man-made objects' pixels are matched.

Never raises: a failure (no manifest, a bad checkpoint, anything in inference) comes back as
``{"status": "failed", ...}`` with no man-made mask, and the caller matches every pixel as
before. Like every action-written GeoTIFF, the maps carry the mask's identity transform; the
export recovers the CRS from the raw scene.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("allotrope.action_types.spectral_library_match.namer")

NAMER_PIXELS_PARQUET = "namer_pixels.parquet"
NAMER_OBJECTS_PARQUET = "namer_objects.parquet"
NAMER_LEVEL1_TIF = "namer_level1_map.tif"
NAMER_LEVEL1_PNG = "namer_level1_map.png"
NAMER_LEVEL1_OVERLAY_PNG = "namer_level1_overlay.png"
NAMER_CLASS_TIF = "namer_class_map.tif"
NAMER_CLASS_OVERLAY_PNG = "namer_class_overlay.png"
NAMER_LEGEND_JSON = "namer_legend.json"

MANMADE_COLOUR = "#E8743B"
NATURAL_COLOUR = "#2A9D8F"
# One colour per v1 class (app/spectral_match/namer/classes.py), chosen to stay apart from the
# level-1 orange / teal and from each other on a satellite RGB.
CLASS_COLOURS = {
    "oil_fuel": "#7C3AED", "plastic": "#06B6D4", "textile_fabric": "#EC4899",
    "wood_paper": "#A16207", "paint_coatings": "#FACC15", "metal": "#3B82F6",
    "concrete_asphalt": "#EF4444",
}
CLASS_LABELS = {
    "oil_fuel": "oil / fuel", "plastic": "plastic", "textile_fabric": "textile / fabric",
    "wood_paper": "wood / paper", "paint_coatings": "paint / coatings", "metal": "metal",
    "concrete_asphalt": "concrete / asphalt",
}
# Tentative classes are drawn at this alpha (of 255) so the confident ones stand out.
TENTATIVE_ALPHA = 105

_PNG_MAX_EDGE = 4096


def run_namer(
    ctx: Any, cfg: dict, *, cube, validity, wavelengths, pixel_mask, profile, output_dir: Path,
) -> tuple[dict[str, Any], Any]:
    """Name every pixel of ``pixel_mask`` (H, W) bool and write the namer_* files.

    Returns (summary["namer"], (H, W) bool pixels of man-made objects), or (failure block, None).
    """
    import numpy as np

    import app.spectral_match.family_match as fm
    from app.spectral_match.namer.infer import load_namer, name_pixels
    from app.spectral_match.namer.objects import class_map, level1_map, name_objects

    from ..config import settings
    from ..foundation_models.resolver import resolve_spectral_namer

    confident_p = float(cfg.get("namer_confident_p", 0.55))
    t = time.monotonic()
    try:
        model_info = resolve_spectral_namer(Path(settings.models_dir))
        model = load_namer(model_info.checkpoint_abs_path)
        rows, cols = np.nonzero(pixel_mask)
        res = name_pixels(model, cube, wavelengths, pixel_mask, fm.FAMILY_BUNDLE_DIR,
                          Path(settings.splib07_cache_dir) / "namer", valid=validity)
        objs = name_objects(pixel_mask, rows, cols, res.p_manmade, res.probs, res.fill,
                            res.classes, confident_p)
        infer_seconds = time.monotonic() - t
        level1, classes = level1_map(objs), class_map(objs)
        legend = namer_legend(res.classes, confident_p, model_info)
        _write_pixels(output_dir / NAMER_PIXELS_PARQUET, rows, cols, res, objs.pixel_obj)
        _write_objects(output_dir / NAMER_OBJECTS_PARQUET, objs.table)
        for name, raster in ((NAMER_LEVEL1_TIF, level1), (NAMER_CLASS_TIF, classes)):
            _write_int8_map(output_dir / name, profile, raster)
        render_codes_png(output_dir / NAMER_LEVEL1_PNG, level1, legend["level1"], transparent=False)
        render_codes_png(output_dir / NAMER_LEVEL1_OVERLAY_PNG, level1, legend["level1"],
                         transparent=True)
        render_codes_png(output_dir / NAMER_CLASS_OVERLAY_PNG, classes, legend["classes"],
                         transparent=True)
        (output_dir / NAMER_LEGEND_JSON).write_text(json.dumps(legend, indent=2))
    except Exception as exc:  # noqa: BLE001 — recorded in the summary; matching runs on all pixels
        logger.exception("spectral namer failed")
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, None

    block = namer_summary(objs, model_info, confident_p)
    block["timing_seconds"] = {"infer": round(infer_seconds, 3),
                               "total": round(time.monotonic() - t, 3)}
    return block, objs.manmade_pixels() & pixel_mask


def namer_summary(objs, model_info, confident_p: float) -> dict[str, Any]:
    """summary["namer"] for a successful run: object and pixel counts, per-class object counts."""
    import numpy as np

    t = objs.table
    mm = objs.manmade
    n_px = t["n_px"].astype(np.int64)
    per_class = {c: int(((t["cls"] == c) & mm).sum()) for c in objs.classes}
    confident = {c: int(((t["cls"] == c) & mm & t["confident"]).sum()) for c in objs.classes}
    return {
        "status": "ok",
        "codename": model_info.codename,
        "label": model_info.label,
        "checkpoint": model_info.checkpoint_filename,
        "version": model_info.version,
        "epoch": model_info.epoch,
        "confident_p": confident_p,
        "n_objects": int(len(mm)),
        "n_objects_manmade": int(mm.sum()),
        "n_objects_natural": int((~mm).sum()),
        "n_objects_confident": int(t["confident"].sum()),
        "n_pixels": int(n_px.sum()),
        "n_pixels_manmade": int(n_px[mm].sum()),
        "n_pixels_natural": int(n_px[~mm].sum()),
        "class_counts": {c: n for c, n in per_class.items() if n},
        "class_counts_confident": {c: n for c, n in confident.items() if n},
    }


def namer_legend(classes: tuple, confident_p: float, model_info) -> dict[str, Any]:
    """Codes of the two namer maps → label / colour / alpha, plus what produced them."""
    from app.spectral_match.namer.objects import TENTATIVE_OFFSET

    level1 = {
        "0": {"label": "Natural", "colour": NATURAL_COLOUR, "alpha": 255},
        "1": {"label": "Man-made", "colour": MANMADE_COLOUR, "alpha": 255},
    }
    cls_codes: dict[str, dict] = {}
    for i, c in enumerate(classes):
        item = {"cls": c, "label": CLASS_LABELS.get(c, c), "colour": CLASS_COLOURS.get(c, "#e2e8f0")}
        cls_codes[str(i + 1)] = {**item, "tentative": False, "alpha": 255}
        cls_codes[str(i + 1 + TENTATIVE_OFFSET)] = {**item, "tentative": True,
                                                    "alpha": TENTATIVE_ALPHA}
    return {"level1": level1, "classes": cls_codes, "confident_p": confident_p,
            "codename": model_info.codename, "checkpoint": model_info.checkpoint_filename}


def render_codes_png(out_path: Path, code_map, codes: dict, *, transparent: bool) -> None:
    """One colour per code (codes: str(code) → {colour, alpha}); other codes are not drawn.

    Dark backing for the thumbnail, transparent for the overlay; the overlay grows each pixel
    by a dimmer one-pixel halo, as the family overlay does, so single pixels read on any RGB.
    """
    import numpy as np
    from PIL import Image

    H, W = code_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if not transparent:
        rgba[...] = (12, 14, 18, 255)
    drawn = np.zeros((H, W), dtype=bool)
    for code, item in codes.items():
        sel = code_map == int(code)
        if sel.any():
            h = item["colour"].lstrip("#")
            rgba[sel] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), item.get("alpha", 255))
            drawn |= sel
    if transparent and drawn.any():
        r, c = np.nonzero(drawn)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = np.clip(r + dr, 0, H - 1), np.clip(c + dc, 0, W - 1)
                empty = rgba[rr, cc, 3] == 0
                rgba[rr[empty], cc[empty], :3] = rgba[r[empty], c[empty], :3].astype(np.uint16) * 3 // 4
                rgba[rr[empty], cc[empty], 3] = (rgba[r[empty], c[empty], 3].astype(np.uint16) * 7 // 8)
    img = Image.fromarray(rgba, mode="RGBA")
    if max(H, W) > _PNG_MAX_EDGE:
        scale = _PNG_MAX_EDGE / max(H, W)
        img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.NEAREST)
    img.save(out_path, format="PNG", optimize=True)


def _write_pixels(out_path: Path, rows, cols, res, obj) -> None:
    """One row per named pixel. int32 / float32 for the browser's parquet reader."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = np.asarray(res.classes, dtype=object)
    table = pa.table({
        "row": rows.astype("int32"), "col": cols.astype("int32"),
        "p_manmade": res.p_manmade.astype("float32"),
        "cls": pa.array(names[res.cls].astype(str) if len(rows) else [], pa.string()),
        "p_cls": res.p_cls.astype("float32"), "fill": res.fill.astype("float32"),
        "obj_id": obj.astype("int32"),
    })
    pq.write_table(table, out_path, compression="snappy")


def _write_objects(out_path: Path, table: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {k: (pa.array(v.astype(str), pa.string()) if v.dtype == object else v)
            for k, v in table.items()}
    pq.write_table(pa.table(cols), out_path, compression="snappy")


def _write_int8_map(out_path: Path, profile, raster) -> None:
    import rasterio

    from app.spectral_match.namer.objects import OUTSIDE

    profile = dict(profile)
    profile.update(dtype="int8", count=1, nodata=OUTSIDE, compress="deflate")
    profile.pop("predictor", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster, 1)
