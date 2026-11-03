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
        MANIFEST.json                       the chain (scene ... match), the selection, the counts

Nothing else (user, 2026-11-01: v1.0.1 dropped the objects shapefile / csv, the two namer maps and the quick-look).
Each pixel carries its object's ``material``: the family-library spectrum closest to the OBJECT's mean spectrum
within the object's class ("natural" entries for a natural object), computed exactly as the match screen's object
card does (frontend/src/components/spectralMatch/data.ts meanSpectrum + nearestInClass), so the two agree; ``mat_lib``
is its library (USGS / ECOSTRESS) and ``mat_ang`` the spectral angle in degrees.

Gotchas: every raster gets the georef the api resolved from the raw scene (the action's own TIFFs carry an identity
transform). Shapefile geometry is in the scene CRS; lat / lon columns are EPSG:4326. A scene with no georeference
gets the pixel grid (crs None): geometry x = col + 0.5, y = -(row + 0.5), no .prj, lat / lon blank, and the
MANIFEST "georeference" says so.
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
# Library of a family-library entry, from its id prefix ("usgs:..." / "ecostress:..."; data/family_library).
_LIBRARY_OF_PREFIX = {"usgs": "USGS", "ecostress": "ECOSTRESS"}


@dataclass(frozen=True)
class ClassPackSpec:
    action_id: str                  # wire ids
    scene_id: str
    sensor_type: str
    artifact_dir: Path
    software_version: str
    crs: Any                        # rasterio CRS of the scene grid; None = pixel grid (no georeference)
    transform: Any                  # its affine
    tokens: tuple[str, ...]
    confident_only: bool = False
    manmade_p: float | None = None  # None = the action's own cut
    chain: dict = field(default_factory=dict)   # MANIFEST "chain", resolved by the api
    georef_note: str | None = None  # MANIFEST "georeference" when the scene has none


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
    members: dict[str, bytes] = {
        f"hyper_anomaly_mask_{slug}.tif": _geotiff(mask.astype(np.uint8), spec, "uint8", None,
                                                   {"ANOMALY_MASK": f"1 = anomaly pixel of the selection {slug}"}),
    }
    for ext, blob in _shapefile(_points(art, obj_map, mask, sel, spec)).items():
        members[f"hyper_anomaly_pixels_{slug}.{ext}"] = blob
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
    """GeoDataFrame, one point per selected pixel. Geometry in the scene CRS, lat / lon in 4326 (blank on the
    pixel grid)."""
    import geopandas as gpd
    import rasterio
    from rasterio.warp import transform as warp

    o = sel.objects
    row_of = {int(i): k for k, i in enumerate(o["obj_id"])}
    rows, cols = np.nonzero(mask)
    xs, ys = rasterio.transform.xy(spec.transform, rows, cols, offset="center") if len(rows) else ([], [])
    xs, ys = np.atleast_1d(xs).astype(float), np.atleast_1d(ys).astype(float)
    if spec.crs is None:
        lons = lats = np.full(len(rows), np.nan)
    else:
        lons, lats = warp(spec.crs, "EPSG:4326", list(xs), list(ys)) if len(rows) else ([], [])
    fam, sam = _matches_by_pixel(art)
    k = [row_of[int(i)] for i in obj_map[rows, cols]]
    natural = (o["call"][k].astype(str) != "manmade") if k else np.zeros(0, bool)
    mat = _object_materials(art, obj_map, sel)
    pix_obj = o["obj_id"][k].astype(int) if k else np.zeros(0, int)
    pixel_cols = {
        "row": rows.astype(int), "col": cols.astype(int), "lat": np.round(lats, 7), "lon": np.round(lons, 7),
        "nm_obj": pix_obj, "nm_call": o["call"][k].astype(str) if k else [],
        "nm_cls": np.where(natural, "natural", o["cls"][k].astype(str)) if k else [],
        "nm_p": np.round(o["p_cls"][k].astype(float), 4) if k else [],
        "nm_mm": np.round(o["p_manmade"][k].astype(float), 4) if k else [],
        "nm_conf": o["confident"][k].astype(int) if k else [],
        "family": [fam.get((int(r), int(c)), "") for r, c in zip(rows, cols)],
        "mat1": [sam.get((int(r), int(c)), ("", None))[0] for r, c in zip(rows, cols)],
        "scr1": [sam.get((int(r), int(c)), ("", None))[1] for r, c in zip(rows, cols)],
        "material": [mat.get(int(i), ("", "", None))[0] for i in pix_obj],
        "mat_lib": [mat.get(int(i), ("", "", None))[1] for i in pix_obj],
        "mat_ang": np.array([mat.get(int(i), ("", "", np.nan))[2] for i in pix_obj], float),
    }
    return gpd.GeoDataFrame(pixel_cols, geometry=gpd.points_from_xy(xs, ys), crs=spec.crs)


def _object_materials(art: Path, obj_map: np.ndarray, sel: Selection) -> dict[int, tuple[str, str, float]]:
    """obj_id -> (closest family-library label, its library, angle in degrees, 2 dp) for each selected object.

    Mirrors the match screen exactly (data.ts meanSpectrum + nearestInClass): the object's mean over its pixels,
    band by band over valid finite values; then, among library rows of the object's class ("natural" for a natural
    object), the largest cosine on the bands both have finite, the first on a tie. Empty when the action has no
    family library or pixel spectra, the band counts differ (the screen then shows no reference either), or the
    class has no entry (class unknown, none covering this sensor, or a legend from before library[].cls)."""
    paths = [art / f for f in ("family_legend.json", "family_library_refl.npz", "anomaly_pixel_spectra.npz")]
    if not all(p.is_file() for p in paths):
        return {}
    entries = json.loads(paths[0].read_text()).get("library") or []
    lib, px = np.load(paths[1]), np.load(paths[2])
    ref = np.where(lib["valid"].astype(bool), lib["refl"].astype(np.float64), np.nan)
    spectra = np.where(px["valid"].astype(bool), px["spectra"].astype(np.float64), np.nan)
    if len(entries) != ref.shape[0] or ref.shape[1] != spectra.shape[1]:
        return {}
    ref = np.where(np.isfinite(ref), ref, np.nan)
    spec_ix = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(px["rows"], px["cols"]))}
    lib_cls = np.array([e.get("cls") or "" for e in entries], dtype=object)
    o = sel.objects
    wanted = {int(i): ("natural" if call != "manmade" else str(cls or ""))
              for i, call, cls, on in zip(o["obj_id"], o["call"], o["cls"], sel.selected) if on}
    rows, cols = np.nonzero(np.isin(obj_map, list(wanted)) & (obj_map > 0))
    members: dict[int, list[int]] = {}
    for r, c in zip(rows, cols):
        i = spec_ix.get((int(r), int(c)))
        if i is not None:
            members.setdefault(int(obj_map[r, c]), []).append(i)
    out: dict[int, tuple[str, str, float]] = {}
    for oid, ix in members.items():
        cand = np.flatnonzero(lib_cls == wanted[oid]) if wanted[oid] else np.zeros(0, int)
        if not len(cand):
            continue
        block = spectra[ix]
        n = np.isfinite(block).sum(axis=0)
        mean = np.where(n > 0, np.nansum(block, axis=0) / np.maximum(n, 1), np.nan)
        cands = ref[cand]                                           # (k, B)
        ok = np.isfinite(mean)[None, :] & np.isfinite(cands)
        a, b = np.where(ok, mean[None, :], 0.0), np.where(ok, cands, 0.0)
        dot, na, nb = (a * b).sum(axis=1), (a * a).sum(axis=1), (b * b).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where((na > 0) & (nb > 0), dot / np.sqrt(na * nb), -np.inf)
        if not np.isfinite(cos).any():
            continue
        best_cos = float(cos.max())
        entry = entries[int(cand[int(np.argmax(cos))])]             # argmax: the first on a tie, as the screen
        angle = float(np.degrees(np.arccos(np.clip(best_cos, -1.0, 1.0))))
        out[oid] = (str(entry.get("label", "")), _LIBRARY_OF_PREFIX.get(str(entry.get("id", "")).split(":", 1)[0], ""),
                    round(angle, 2))
    return out


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
        "material_rule": ("material / mat_lib / mat_ang: the family-library spectrum (USGS or ECOSTRESS) closest by "
                          "spectral angle (degrees) to the pixel's OBJECT mean spectrum, within the object's class "
                          "(natural entries for natural objects); mat1 / scr1 are the pixel's own USGS splib07 SAM top-1"),
        **({"georeference": spec.georef_note} if spec.georef_note else {}),
        "files": files,
        "report_required": True,
    }
