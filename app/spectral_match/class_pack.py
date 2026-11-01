"""
The class-driven output pack of a spectral_library_match action: only the objects of the classes picked.

Built by ``POST /actions/{id}/export?classes=...`` (backend/allotrope/api/action_class_pack.py), which resolves the
scene georef and the chain of upstream actions and hands them in; the selection itself is
app/spectral_match/class_selection.py. Design: docs/proposals/class-driven-flow.md (gap 2, additions A and D).

Why: the submission is a boolean anomaly map plus points, of the anomalies the analyst decided are of interest -
"oil + plastic", "all man-made", "natural" - at the man-made sensitivity they chose. The full bundle
(app/spectral_match/export.py) carries every anomaly pixel and the coded rasters; this pack carries the selection
only, with a manifest that records every step that produced it. There is no heatmap in it.

    hyper_<scene>_<action>_<sel>/
        hyper_anomaly_mask_<sel>.tif        uint8 0/1, 1 on the selected objects' pixels; scene CRS and affine
        hyper_anomaly_pixels_<sel>.shp ...  one point per selected pixel (+ .shx .dbf .prj .cpg)
        hyper_anomaly_objects_<sel>.shp ... one point per selected object, at its centroid
        hyper_anomaly_objects_<sel>.csv     obj_id, call, class, confident, p_manmade, p_class, n_px, lat, lon
        hyper_namer_level1.tif              int8 -1 outside the mask, 0 natural, 1 man-made, at manmade_p
        hyper_namer_class.tif               int8 -1 outside, 0 natural, class codes (+10 tentative), at manmade_p
        hyper_quicklook_<sel>.png           dimmed RGB, the selected pixels as yellow points (squares man-made,
                                            circles natural, dark outline, a minimum size)
        MANIFEST.json                       the chain (scene ... match), the selection, the counts

Gotchas: every raster gets the georef the api resolved from the raw scene (the action's own TIFFs carry an identity
transform). Shapefile geometry is in the scene CRS; lat / lon columns are EPSG:4326. The two namer maps are
recomputed at manmade_p so they agree with the selection, not with the action's stored maps.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .class_selection import UNKNOWN_CLASS, Selection, SelectionError, select

SCHEMA = "allotrope-export-hyper-classes/1"
# hyper_namer_class.tif code of a man-made object whose class could not be computed (UNKNOWN_CLASS).
UNKNOWN_CODE = 99
# Quick-look rendering (addition D): one bright colour unused by the class colours, a dark outline, and markers
# that keep a minimum size so single pixels stay visible on the whole scene.
POINT_RGB = (255, 230, 0)
OUTLINE_RGB = (20, 20, 20)
_QUICKLOOK_EDGE = 1600          # long edge of the PNG, px
_MIN_MARKER_HALF = 2            # marker half-size at PNG scale, px (5 px squares / circles)
_RGB_DIM = 0.45                 # the RGB under the points is desaturated and dimmed to this brightness


@dataclass(frozen=True)
class ClassPackSpec:
    action_id: str                  # wire ids
    scene_id: str
    sensor_type: str
    artifact_dir: Path
    software_version: str
    crs: Any                        # rasterio CRS of the scene grid
    transform: Any                  # its affine
    tokens: tuple[str, ...]
    confident_only: bool = False
    manmade_p: float | None = None  # None = the action's own cut
    chain: dict = field(default_factory=dict)   # MANIFEST "chain", resolved by the api


def selection_of(spec: ClassPackSpec) -> Selection:
    """The spec's selection (the same call selection_counts makes)."""
    legend = namer_legend(spec.artifact_dir)
    return select(spec.artifact_dir, spec.tokens, confident_only=spec.confident_only,
                  manmade_p=spec.manmade_p, confident_p=float(legend.get("confident_p", 0.55)))


def namer_legend(artifact_dir: Path) -> dict:
    path = artifact_dir / "namer_legend.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def build_class_pack(spec: ClassPackSpec) -> tuple[bytes, str, Selection]:
    """The pack as (zip bytes, zip file name, the selection). Raises SelectionError without namer data."""
    art = spec.artifact_dir
    for f in ("namer_pixels.parquet", "namer_objects.parquet", "namer_level1_map.tif"):
        if not (art / f).is_file():
            raise SelectionError(f"this action has no spectral-namer results ({f} is missing), so it cannot be "
                                 "exported by class")
    sel = selection_of(spec)
    legend = namer_legend(art)
    obj_map = _object_map(art)
    ids = np.asarray(sel.objects["obj_id"], np.int64)
    lut_size = int(max(ids.max(initial=0), obj_map.max(initial=0))) + 1
    selected_lut = np.zeros(lut_size, bool)
    selected_lut[ids[sel.selected]] = True
    mask = (obj_map > 0) & selected_lut[np.maximum(obj_map, 0)]
    # namer_pixels covers the whole committed mask, so the map holds every selected object's pixels.
    if int(mask.sum()) != sel.n_pixels:
        raise RuntimeError(f"namer_pixels.parquet has {int(mask.sum())} pixels of the selection, "
                           f"namer_objects.parquet says {sel.n_pixels}")

    slug = sel.slug()
    folder = f"hyper_{_short(spec.scene_id)}_{_short(spec.action_id)}_{slug}"
    level1, classes = _namer_maps(obj_map, sel, legend)
    members: dict[str, bytes] = {
        f"hyper_anomaly_mask_{slug}.tif": _geotiff(mask.astype(np.uint8), spec, "uint8", None,
                                                   {"ANOMALY_MASK": f"1 = anomaly pixel of the selection {slug}"}),
        "hyper_namer_level1.tif": _geotiff(level1, spec, "int8", -1, {"NAMER_LEVEL1_CODES": json.dumps(
            {"-1": "outside the anomaly mask", "0": "natural", "1": "man-made"})}),
        "hyper_namer_class.tif": _geotiff(classes, spec, "int8", -1, {"NAMER_CLASS_CODES": json.dumps(
            _class_codes(legend))}),
        f"hyper_quicklook_{slug}.png": _quicklook(art, obj_map, mask, sel),
    }
    pixel_gdf, object_gdf, object_csv = _points(art, obj_map, mask, sel, spec)
    for name, gdf in ((f"hyper_anomaly_pixels_{slug}", pixel_gdf), (f"hyper_anomaly_objects_{slug}", object_gdf)):
        for ext, blob in _shapefile(gdf).items():
            members[f"{name}.{ext}"] = blob
    members[f"hyper_anomaly_objects_{slug}.csv"] = object_csv
    members["MANIFEST.json"] = json.dumps(_manifest(spec, sel, legend, mask.shape, sorted(members)),
                                          indent=2).encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in members.items():
            zf.writestr(f"{folder}/{name}", blob)
    return buf.getvalue(), f"{folder}.zip", sel


# ---------------------------------------------------------------------------


def _short(wire_id: str) -> str:
    return wire_id.split("_", 1)[-1].replace("-", "")[:8]


def _object_map(art: Path) -> np.ndarray:
    """(H, W) int32 object id per named pixel, 0 elsewhere, painted from namer_pixels.parquet."""
    import pyarrow.parquet as pq
    import rasterio

    with rasterio.open(art / "namer_level1_map.tif") as src:
        shape = (src.height, src.width)
    px = pq.read_table(art / "namer_pixels.parquet", columns=["row", "col", "obj_id"]).to_pydict()
    out = np.zeros(shape, np.int32)
    out[np.asarray(px["row"], np.int64), np.asarray(px["col"], np.int64)] = np.asarray(px["obj_id"], np.int32)
    return out


def _class_codes(legend: dict) -> dict[str, str]:
    """hyper_namer_class.tif code -> label, from the action's namer_legend.json."""
    codes = {"-1": "outside the anomaly mask", "0": "natural", str(UNKNOWN_CODE): "man-made, class unknown"}
    for code, item in (legend.get("classes") or {}).items():
        codes[code] = f"{item.get('label', item.get('cls'))}{' (tentative)' if item.get('tentative') else ''}"
    return codes


def _namer_maps(obj_map: np.ndarray, sel: Selection, legend: dict) -> tuple[np.ndarray, np.ndarray]:
    """Level-1 and class maps at the selection's cut (codes as namer/objects.py writes them)."""
    from app.spectral_match.namer.objects import OUTSIDE, TENTATIVE_OFFSET

    o = sel.objects
    code_of = {item["cls"]: int(code) for code, item in (legend.get("classes") or {}).items()
               if not item.get("tentative")}
    mm = o["call"] == "manmade"
    size = int(max(np.max(o["obj_id"], initial=0), obj_map.max(initial=0))) + 1
    l1_lut = np.full(size, OUTSIDE, np.int8)
    cls_lut = np.full(size, OUTSIDE, np.int8)
    for oid, is_mm, cls, conf in zip(o["obj_id"], mm, o["cls"], o["confident"]):
        l1_lut[int(oid)] = 1 if is_mm else 0
        if not is_mm:
            cls_lut[int(oid)] = 0
        elif cls in code_of:
            cls_lut[int(oid)] = code_of[cls] + (0 if conf else TENTATIVE_OFFSET)
        else:
            cls_lut[int(oid)] = UNKNOWN_CODE
    l1_lut[0] = cls_lut[0] = OUTSIDE
    return l1_lut[obj_map], cls_lut[obj_map]


def _geotiff(raster: np.ndarray, spec: ClassPackSpec, dtype: str, nodata, tags: dict) -> bytes:
    from rasterio.io import MemoryFile

    profile = {"driver": "GTiff", "height": raster.shape[0], "width": raster.shape[1], "count": 1,
               "dtype": dtype, "crs": spec.crs, "transform": spec.transform, "compress": "lzw", "nodata": nodata}
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(raster.astype(dtype), 1)
            dst.update_tags(**tags)
        return memfile.read()


def _points(art: Path, obj_map: np.ndarray, mask: np.ndarray, sel: Selection, spec: ClassPackSpec):
    """(pixel GeoDataFrame, object GeoDataFrame, object CSV bytes). Geometry in the scene CRS, lat / lon in 4326."""
    import csv

    import geopandas as gpd
    import rasterio
    from rasterio.warp import transform as warp

    o = sel.objects
    row_of = {int(i): k for k, i in enumerate(o["obj_id"])}
    rows, cols = np.nonzero(mask)
    xs, ys = rasterio.transform.xy(spec.transform, rows, cols, offset="center") if len(rows) else ([], [])
    xs, ys = np.atleast_1d(xs).astype(float), np.atleast_1d(ys).astype(float)
    lons, lats = warp(spec.crs, "EPSG:4326", list(xs), list(ys)) if len(rows) else ([], [])
    fam, sam = _matches_by_pixel(art)
    k = [row_of[int(i)] for i in obj_map[rows, cols]]
    pixel_cols = {
        "row": rows.astype(int), "col": cols.astype(int), "lat": np.round(lats, 7), "lon": np.round(lons, 7),
        "nm_obj": o["obj_id"][k].astype(int) if k else [], "nm_call": o["call"][k].astype(str) if k else [],
        "nm_cls": o["cls"][k].astype(str) if k else [],
        "nm_p": np.round(o["p_cls"][k].astype(float), 4) if k else [],
        "nm_mm": np.round(o["p_manmade"][k].astype(float), 4) if k else [],
        "nm_conf": o["confident"][k].astype(int) if k else [],
        "family": [fam.get((int(r), int(c)), "") for r, c in zip(rows, cols)],
        "mat1": [sam.get((int(r), int(c)), ("", None))[0] for r, c in zip(rows, cols)],
        "scr1": [sam.get((int(r), int(c)), ("", None))[1] for r, c in zip(rows, cols)],
    }
    pixel_gdf = gpd.GeoDataFrame(pixel_cols, geometry=gpd.points_from_xy(xs, ys), crs=spec.crs)

    picked = np.nonzero(sel.selected)[0]
    oxs, oys = (rasterio.transform.xy(spec.transform, o["row"][picked].astype(float),
                                      o["col"][picked].astype(float), offset="center")
                if len(picked) else ([], []))
    oxs, oys = np.atleast_1d(oxs).astype(float), np.atleast_1d(oys).astype(float)
    olons, olats = warp(spec.crs, "EPSG:4326", list(oxs), list(oys)) if len(picked) else ([], [])
    obj_cols = {
        "obj_id": o["obj_id"][picked].astype(int), "call": o["call"][picked].astype(str),
        "class": [c if c else "natural" for c in o["cls"][picked].astype(str)] if len(picked) else [],
        "confident": o["confident"][picked].astype(int),
        "p_manmade": np.round(o["p_manmade"][picked].astype(float), 4),
        "p_class": np.round(o["p_cls"][picked].astype(float), 4),
        "n_px": o["n_px"][picked].astype(int), "lat": np.round(olats, 7), "lon": np.round(olons, 7),
    }
    object_gdf = gpd.GeoDataFrame(obj_cols, geometry=gpd.points_from_xy(oxs, oys), crs=spec.crs)

    buf = io.StringIO()
    keys = list(obj_cols)
    writer = csv.writer(buf)
    writer.writerow(keys)
    for i in range(len(picked)):
        writer.writerow(["" if (isinstance(v, float) and np.isnan(v)) else v
                         for v in (obj_cols[c][i] for c in keys)])
    return pixel_gdf, object_gdf, buf.getvalue().encode("utf-8")


def _matches_by_pixel(art: Path) -> tuple[dict, dict]:
    """(family name, (top SAM material, angle)) per (row, col), where the action matched it."""
    import pyarrow.parquet as pq

    fam: dict = {}
    path = art / "family_matches.parquet"
    legend_path = art / "family_legend.json"
    if path.is_file() and legend_path.is_file():
        codes = json.loads(legend_path.read_text()).get("codes", {})
        t = pq.read_table(path, columns=["row", "col", "family_code"]).to_pydict()
        fam = {(int(r), int(c)): codes.get(str(int(f)), {}).get("family", "")
               for r, c, f in zip(t["row"], t["col"], t["family_code"])}
    sam: dict = {}
    path = art / "matches.parquet"
    if path.is_file():
        t = pq.read_table(path, columns=["row", "col", "rank", "name", "angle_deg"]).to_pydict()
        top = min(t["rank"], default=0)
        sam = {(int(r), int(c)): (str(n), round(float(a), 4))
               for r, c, k, n, a in zip(t["row"], t["col"], t["rank"], t["name"], t["angle_deg"]) if k == top}
    return fam, sam


def _shapefile(gdf) -> dict[str, bytes]:
    """Shapefile + sidecars of ``gdf`` as {extension: bytes}. An empty selection still gets its files."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "shape.shp"
        gdf.to_file(out, driver="ESRI Shapefile")
        return {ext: (Path(td) / f"shape.{ext}").read_bytes()
                for ext in ("shp", "shx", "dbf", "prj", "cpg") if (Path(td) / f"shape.{ext}").is_file()}


def _quicklook(art: Path, obj_map: np.ndarray, mask: np.ndarray, sel: Selection) -> bytes:
    """Dimmed, desaturated RGB with the selected pixels as yellow markers: squares man-made, circles natural."""
    from PIL import Image, ImageDraw

    H, W = mask.shape
    rgb_path = art / "rgb.png"
    if rgb_path.is_file():
        rgb = np.asarray(Image.open(rgb_path).convert("RGB").resize((W, H), Image.BILINEAR), np.float32)
    else:
        rgb = np.full((H, W, 3), 128, np.float32)
    grey = rgb.mean(axis=2, keepdims=True)
    base = (_RGB_DIM * (0.35 * rgb + 0.65 * grey)).clip(0, 255).astype(np.uint8)
    scale = min(1.0 * _QUICKLOOK_EDGE / max(H, W), 8.0)
    size = (max(1, round(W * scale)), max(1, round(H * scale)))
    img = Image.fromarray(base).resize(size, Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    half = max(_MIN_MARKER_HALF, round(scale / 2))
    mm_ids = set(int(i) for i, c in zip(sel.objects["obj_id"], sel.objects["call"]) if c == "manmade")
    rows, cols = np.nonzero(mask)
    pts = [((c + 0.5) * scale, (r + 0.5) * scale, int(obj_map[r, c]) in mm_ids) for r, c in zip(rows, cols)]
    # Outlines first, then fills, so a neighbour's outline never covers a point.
    for grow, colour in ((1, OUTLINE_RGB), (0, POINT_RGB)):
        h = half + grow
        for x, y, is_mm in pts:
            box = (x - h, y - h, x + h, y + h)
            (draw.rectangle if is_mm else draw.ellipse)(box, fill=colour)
    n_mm = sum(1 for p in pts if p[2])
    _legend(draw, n_mm, len(pts) - n_mm)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _legend(draw, n_manmade: int, n_natural: int) -> None:
    """Top-left key: the two marker shapes with their pixel counts."""
    draw.rectangle((6, 6, 210, 46), fill=(0, 0, 0))
    for i, (label, n, square) in enumerate((("man-made", n_manmade, True), ("natural", n_natural, False))):
        y = 12 + 18 * i
        box = (12, y, 22, y + 10)
        (draw.rectangle if square else draw.ellipse)(box, fill=POINT_RGB, outline=OUTLINE_RGB)
        draw.text((30, y - 1), f"{label}  {n:,} px", fill=(235, 235, 235))


def _manifest(spec: ClassPackSpec, sel: Selection, legend: dict, shape, files: list[str]) -> dict:
    return {
        "schema": SCHEMA,
        "software_version": spec.software_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action_id": spec.action_id,
        "scene_id": spec.scene_id,
        "sensor_type": spec.sensor_type,
        "grid": {"height": int(shape[0]), "width": int(shape[1]), "crs_wkt": spec.crs.to_wkt() if spec.crs else None,
                 "transform": list(spec.transform)[:6] if spec.transform is not None else None},
        "chain": spec.chain,
        "selection": {
            "classes": list(sel.tokens),
            "confident_only": sel.confident_only,
            "manmade_p": sel.manmade_p,
            "manmade_p_default": sel.default_p,
            "slug": sel.slug(),
            "newly_manmade_class_source": sel.class_source,
            "n_objects": sel.n_objects,
            "n_pixels": sel.n_pixels,
            "per_class": sel.per_class(),
            "rule": ("natural = objects called natural; manmade = all man-made objects; a material class = "
                     "man-made objects whose closest class it is; the union of the tokens; confident_only drops "
                     "tentative man-made objects. Calls re-decided at manmade_p on the object's mean P(man-made)."),
        },
        "namer": {"codename": legend.get("codename"), "checkpoint": legend.get("checkpoint"),
                  "confident_p": legend.get("confident_p")},
        "unknown_class": UNKNOWN_CLASS,
        "files": files,
        "report_required": True,
    }
