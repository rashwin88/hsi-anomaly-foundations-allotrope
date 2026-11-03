"""Submission-ready export bundle for `spectral_library_match` results.

Produces a zip with everything an evaluation script (or a human downstream
analyst) needs: a georeferenced GeoTIFF, a shapefile with sidecars, a
flat CSV, and a manifest. Every path inside the zip is prefixed with
``hyper_`` per the submission rules (2026-05-14):

    hyper_<scene_id>_<action_id>/
        hyper_anomaly_materials.tif         # int16 GeoTIFF, top-1 lib_ix
        hyper_anomaly_materials.shp         # Point per matched pixel
        hyper_anomaly_materials.shx
        hyper_anomaly_materials.dbf
        hyper_anomaly_materials.prj
        hyper_anomaly_materials.cpg
        hyper_material_legend.json          # id → name/chapter/asd_subtype
        hyper_match_summary.csv             # flat table, no geometry
        hyper_namer_level1.tif              # int8 GeoTIFF, man-made vs natural (namer actions)
        hyper_namer_objects.csv             # one row per object (namer actions)
        MANIFEST.json                       # audit trail

Hard requirements from the submission rules:
  - GeoTIFF carries the scene CRS. A scene with none exports in pixel coordinates
    (2026-11-01): the api passes ``app.georef.pixel_grid()`` with crs None and a
    ``georef_note``; no .prj, lat / lon blank, MANIFEST "georeference" says so.
    ``MissingCRSError`` (a 422) only when no override transform is given at all.
  - Filenames AND the folder path must literally start with ``hyper_``.
  - Shapefile sidecar set is complete (geopandas writes all five files).

Materials-as-metadata channels (belt-and-suspenders):
  - ``MATERIAL_IDS`` GeoTIFF tag — JSON id→name embedded in the raster's
    own metadata directory. Travels with the file. Read by ArcGIS, QGIS,
    gdalinfo.
  - Shapefile DBF columns ``mat1/scr1, mat2/scr2, mat3/scr3`` — top-3 per
    pixel. (Top-K survives in the CSV.)
  - ``hyper_material_legend.json`` — separate file, redundant with the
    TIFF tag for humans who don't use GDAL.

Three bundle layouts, chosen by what the action wrote (schema in MANIFEST.json):

  - ``allotrope-export-hyper/3`` — actions where the spectral namer ran
    (``namer_pixels.parquet``). One point per anomaly pixel, natural ones included. Leading
    DBF / CSV columns ``nm_mm`` (the object's P(man-made)), ``nm_call`` (manmade / natural),
    ``nm_cls`` (closest known class, man-made only), ``nm_p`` (its P(class | man-made)),
    ``nm_conf`` (1 confident, 0 tentative), ``nm_obj`` (object id), then the /2 family columns
    (natural pixels: family ``namer_natural``, not matched) and SAM's. Adds
    ``hyper_namer_level1.tif`` (-1 outside the mask, 0 natural, 1 man-made, same georef as the
    main raster) and ``hyper_namer_objects.csv`` (centroid lat / lon per object).
  - ``allotrope-export-hyper/2`` — actions with family matching
    (``family_matches.parquet``). The raster holds family codes (-2 outside the mask,
    -1 not scored, 0 man-made type unconfirmed, 1..F a family) and the legend maps them.
    One point per anomaly pixel; DBF ``family, group, fconf, named, best_ent`` then SAM's
    ``mat1/scr1 … mat3/scr3`` (scr = SAM angle, degrees). ``hyper_sam_legend.json`` keeps
    the SAM library legend.
  - ``allotrope-export-hyper/1`` — SAM-only actions from before family matching: top-1
    library index raster, ``conf = angle < confidence_threshold_deg``. Unchanged.

Subset bundles (``ExportSpec.subset`` "manmade" / "natural", namer actions only) keep only the
pixels of objects the namer gave that call, and replace every coded raster with one boolean
map - the submission shape (boolean GeoTIFF + shapefile), no heatmaps:

    hyper_<scene_id>_<action_id>_<subset>/
        hyper_anomaly_mask_<subset>.tif     # uint8 0/1 on the subset's pixels, scene CRS
        hyper_anomaly_materials_<subset>.*  # one point per subset pixel, /3 columns
        hyper_match_summary_<subset>.csv
        hyper_namer_objects_<subset>.csv
        MANIFEST.json                       # schema /3, "subset" names the call
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MissingCRSError(Exception):
    """Raised when the source GeoTIFF has no CRS — we refuse to silently
    ship a bundle that the evaluation script will disqualify."""


class SubsetUnavailableError(Exception):
    """A man-made / natural subset was asked of an action the spectral namer did not run on."""


# ExportSpec.subset values. "all" is the whole bundle; the others are the namer's object calls.
SUBSETS = ("all", "manmade", "natural")


# Default confidence threshold (degrees). SAM angles below this are
# flagged ``conf=1`` in the shapefile/CSV. Caller can override.
DEFAULT_CONFIDENCE_THRESHOLD_DEG = 15.0

# Shapefile column-name limit is 10 chars (ESRI spec, hard).
# Field count limit is 255 — we use 10 (well under).
_SHP_COLUMN_RENAME = {
    "latitude":  "lat",
    "longitude": "lon",
    "material_1": "mat1",
    "material_2": "mat2",
    "material_3": "mat3",
    "score_1": "scr1",
    "score_2": "scr2",
    "score_3": "scr3",
    "confident": "conf",
    "family_group": "group",
    "family_confidence": "fconf",
    "family_named": "named",
    "best_entry": "best_ent",
}

# Family bundles: CSV / DBF columns ahead of SAM's material_k / score_k.
_FAMILY_KEYS = ["row", "col", "latitude", "longitude", "family", "family_group",
                "family_confidence", "family_named", "best_entry"]
# Namer bundles: the object's namer result, per pixel, after row / col / lat / lon. Already
# within the shapefile's 10-character limit.
_NAMER_KEYS = ["nm_mm", "nm_call", "nm_cls", "nm_p", "nm_conf", "nm_obj"]
# Family columns of an anomaly pixel the namer called natural (not matched).
_NAMER_NATURAL_FAMILY = {"family": "namer_natural", "family_group": "natural",
                         "family_confidence": 0.0, "family_named": 0, "best_entry": ""}
_OBJECT_KEYS = ["obj_id", "n_px", "latitude", "longitude", "row", "col", "p_manmade", "call",
                "cls", "p_cls", "confident", "fill"]


@dataclass(frozen=True)
class ExportSpec:
    """Inputs the api endpoint resolves and hands to ``build_hyper_bundle``."""

    action_id: str          # wire id (`action_<uuid>`)
    scene_id: str           # wire id (`scene_<uuid>`)
    sensor_type: str
    artifact_dir: Path      # action's output dir (must contain match_map.tif, …)
    splib_version: str | None
    software_version: str
    confidence_threshold_deg: float = DEFAULT_CONFIDENCE_THRESHOLD_DEG
    # Caller-supplied georef override. Used when the action's match_map.tif
    # was written with an identity transform (because the vendable doesn't
    # carry CRS). The api resolves this from the raw scene file via
    # ``app.georef.resolve_scene_georef`` and threads it in here.
    override_transform: Any = None     # rasterio.transform.Affine
    override_crs: Any = None           # rasterio.crs.CRS; None with a transform = pixel grid
    subset: str = "all"                # "manmade" / "natural": only those objects' pixels
    georef_note: str | None = None     # set when the scene has no georeference


def build_hyper_bundle(spec: ExportSpec) -> tuple[bytes, str]:
    """Build the submission zip in-memory. Returns (zip_bytes, filename).

    Raises ``MissingCRSError`` if the source GeoTIFF has no CRS.
    Raises ``FileNotFoundError`` if required input artifacts are missing.
    Raises ``SubsetUnavailableError`` for a man-made / natural subset without namer files.
    """
    import numpy as np
    import rasterio
    import geopandas as gpd
    from shapely.geometry import Point
    import pyarrow.parquet as pq

    art = spec.artifact_dir
    match_map_tif = art / "match_map.tif"
    matches_parquet = art / "matches.parquet"
    legend_json = art / "match_map_legend.json"
    if not match_map_tif.is_file():
        raise FileNotFoundError(f"missing {match_map_tif}")
    if not matches_parquet.is_file():
        raise FileNotFoundError(f"missing {matches_parquet}")
    if not legend_json.is_file():
        raise FileNotFoundError(f"missing {legend_json}")
    family_files = [art / f for f in ("family_matches.parquet", "family_map.tif",
                                      "family_legend.json")]
    is_family = all(p.is_file() for p in family_files)
    raster_tif = family_files[1] if is_family else match_map_tif
    namer_files = [art / f for f in ("namer_pixels.parquet", "namer_objects.parquet",
                                     "namer_level1_map.tif")]
    is_namer = all(p.is_file() for p in namer_files)
    if spec.subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {spec.subset!r}")
    subset = None if spec.subset == "all" else spec.subset
    if subset and not is_namer:
        raise SubsetUnavailableError(
            "this action has no spectral-namer results, so it cannot be split into "
            "man-made and natural objects; export it without a subset")

    # ---- 1. Read the match map and (if needed) override georef -----
    with rasterio.open(raster_tif) as src:
        raster = src.read(1)
        profile = src.profile
        src_crs = src.crs
        src_transform = src.transform

    # If the source TIFF has no CRS — common today because the vendable
    # model doesn't thread spatial reference — accept a caller-supplied
    # override (resolved at endpoint time from the raw scene file).
    if src_crs is None:
        if spec.override_transform is not None:
            src_crs = spec.override_crs
            src_transform = spec.override_transform
            profile = dict(profile)
            profile["crs"] = src_crs
            profile["transform"] = src_transform
        else:
            raise MissingCRSError(
                f"match_map.tif at {match_map_tif} has no CRS and no "
                f"override was provided; the source vendable for scene "
                f"{spec.scene_id} was onboarded without georeference. "
                f"Submission would be disqualified."
            )

    # ---- 2. Read the matches table and build per-pixel row records --
    table = pq.read_table(matches_parquet)
    rows = table.to_pylist()

    # Group by (row, col) so each pixel becomes one feature with up to K ranks.
    by_pixel: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        key = (int(r["row"]), int(r["col"]))
        by_pixel.setdefault(key, []).append(r)
    for bucket in by_pixel.values():
        bucket.sort(key=lambda x: int(x["rank"]))

    # Build legend map: library_ix → {name, chapter, asd_subtype}
    sam_legend = json.loads(legend_json.read_text())  # already keyed by library_ix
    family_legend = json.loads(family_files[2].read_text()) if is_family else None
    legend = family_legend["codes"] if is_family else sam_legend

    # ---- 3. Build features ------------------------------------------
    # Namer bundles: one point per anomaly pixel; family / SAM attached where they matched.
    # Family bundles: one point per anomaly pixel, SAM ranks attached where SAM matched.
    # SAM bundles: one point per SAM-matched pixel.
    fam_by_px = ({(int(f["row"]), int(f["col"])): f
                  for f in pq.read_table(family_files[0]).to_pylist()} if is_family else {})
    objects = pq.read_table(namer_files[1]).to_pylist() if is_namer else []
    obj_by_id = {int(o["obj_id"]): o for o in objects}
    if is_namer:
        pixels = [(int(n["row"]), int(n["col"]), fam_by_px.get((int(n["row"]), int(n["col"]))),
                   obj_by_id[int(n["obj_id"])])
                  for n in pq.read_table(namer_files[0], columns=["row", "col", "obj_id"]).to_pylist()]
    elif is_family:
        pixels = [(r, c, f, None) for (r, c), f in fam_by_px.items()]
    else:
        pixels = [(r, c, None, None) for (r, c) in by_pixel]
    if subset:
        pixels = [p for p in pixels if p[3]["call"] == subset]
        objects = [o for o in objects if o["call"] == subset]
    records: list[dict[str, Any]] = []
    for r, c, fam, obj in pixels:
        # Pixel-centre lon/lat from the source raster's affine.
        lon, lat = rasterio.transform.xy(src_transform, r, c, offset="center")
        rec: dict[str, Any] = {
            "row": int(r),
            "col": int(c),
            # Pixel-grid x / y are not degrees: no CRS, no lat / lon.
            "latitude": float(lat) if src_crs is not None else float("nan"),
            "longitude": float(lon) if src_crs is not None else float("nan"),
        }
        if obj is not None:
            rec.update(_namer_fields(obj))
        if fam is not None:
            rec.update(_family_fields(fam, family_legend["codes"]))
        elif is_family:
            rec.update(_NAMER_NATURAL_FAMILY)
        ranks = by_pixel.get((r, c), [])
        for k, hit in enumerate(ranks):
            rec[f"material_{k+1}"] = hit["name"]
            rec[f"score_{k+1}"] = round(float(hit["angle_deg"]), 5)
        if not is_family:
            rec["confident"] = int(
                bool(ranks) and float(ranks[0]["angle_deg"]) < spec.confidence_threshold_deg
            )
        rec["geometry"] = Point(lon, lat)
        records.append(rec)

    # An empty subset (no man-made objects, say) still ships its all-zero mask and manifest.
    gdf = (gpd.GeoDataFrame(records, geometry="geometry", crs=src_crs) if records
           else gpd.GeoDataFrame(geometry=[], crs=src_crs))

    # Reproject to EPSG:4326 if we're not already there — common GIS
    # convention for shapefile/CSV, and matches what `latitude/longitude`
    # field names imply.
    if records and gdf.crs is not None and str(gdf.crs).lower() not in ("epsg:4326",):
        # Compute lat/lon in EPSG:4326 explicitly for the attribute table,
        # but keep the geometry in the native CRS so it overlays the
        # source raster correctly in QGIS/ArcGIS.
        # Decision: geometry stays in source CRS (matches the GeoTIFF).
        # Attribute lat/lon are reported in 4326 for human use.
        gdf_4326 = gdf.to_crs("EPSG:4326")
        gdf["latitude"] = gdf_4326.geometry.y
        gdf["longitude"] = gdf_4326.geometry.x

    # ---- 4. Assemble the zip in-memory -------------------------------
    # A subset bundle names the folder and every member after the subset.
    sfx = f"_{subset}" if subset else ""
    folder = f"hyper_{_short(spec.scene_id)}_{_short(spec.action_id)}{sfx}"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # 4a. GeoTIFF: the subset's boolean mask, or the coded raster with its MATERIAL_IDS tag.
        if subset:
            zf.writestr(f"{folder}/hyper_anomaly_mask_{subset}.tif",
                        _write_subset_mask_bytes(namer_files[2], subset, crs=src_crs,
                                                 transform=src_transform))
        else:
            tif_bytes = _write_geotiff_bytes(
                raster=raster, profile=profile, legend=legend,
                nodata=-2 if is_family else -1,
            )
            zf.writestr(f"{folder}/hyper_anomaly_materials.tif", tif_bytes)

        # 4b. Shapefile + sidecars (geopandas writes them to a tmpdir;
        # we read each back as bytes and stuff into the zip).
        shp_files = _write_shapefile_sidecars(
            gdf=gdf,
            top_k_for_shapefile=3,
            lead_cols=(_NAMER_KEYS if is_namer else []) + (_FAMILY_KEYS[4:] if is_family else []),
        ) if records else {}
        for fname, blob in shp_files.items():
            zf.writestr(f"{folder}/hyper_anomaly_materials{sfx}.{fname}", blob)

        # 4c. Legend JSON: the codes of the coded raster, which a subset bundle does not carry.
        if not subset:
            zf.writestr(
                f"{folder}/hyper_material_legend.json",
                json.dumps(family_legend if is_family else legend, indent=2),
            )
        if is_family and not subset:
            zf.writestr(f"{folder}/hyper_sam_legend.json", json.dumps(sam_legend, indent=2))

        # 4d. Flat CSV (no geometry, all top-K)
        base_keys = _FAMILY_KEYS if is_family else ["row", "col", "latitude", "longitude",
                                                     "confident"]
        if is_namer:
            base_keys = base_keys[:4] + _NAMER_KEYS + base_keys[4:]
        csv_bytes = _records_to_csv_bytes(
            records, base_keys=base_keys if (is_family or is_namer) else None)
        zf.writestr(f"{folder}/hyper_match_summary{sfx}.csv", csv_bytes)

        # 4d'. The namer's level-1 map (same georef as the main raster) and its objects.
        if is_namer and not subset:
            zf.writestr(f"{folder}/hyper_namer_level1.tif",
                        _write_level1_geotiff_bytes(namer_files[2], crs=src_crs,
                                                    transform=src_transform))
        if is_namer:
            zf.writestr(f"{folder}/hyper_namer_objects{sfx}.csv",
                        _objects_csv_bytes(objects, src_transform, src_crs))

        # 4e. Manifest
        manifest = {
            "schema": "allotrope-export-hyper/2" if is_family else "allotrope-export-hyper/1",
            "action_id": spec.action_id,
            "scene_id": spec.scene_id,
            "sensor_type": spec.sensor_type,
            "splib07_version": spec.splib_version,
            "software_version": spec.software_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "crs_wkt": src_crs.to_wkt() if src_crs else None,
            "n_matched_pixels": len(records),
            "n_unique_materials": len({rec.get("material_1") for rec in records} - {None}),
            "report_required": True,
            "report_note": (
                "Submission requires a Word-doc report describing experiments, "
                "experiences, and approaches (30% of marks). Author manually."
            ),
        }
        if is_namer:
            manifest["schema"] = "allotrope-export-hyper/3"
            manifest["namer"] = _namer_manifest(art, objects)
            manifest["namer"]["n_pixels_natural"] = sum(1 for rec in records
                                                       if rec["nm_call"] == "natural")
            manifest["subset"] = subset or "all"
        if subset:
            manifest["subset_note"] = (
                f"only the pixels of objects the spectral namer called {subset}; "
                f"hyper_anomaly_mask_{subset}.tif is 1 on exactly those pixels, 0 elsewhere")
        if is_family:
            manifest.update({
                "matcher": "family_v1 (calibrated Simplex ACE) + SAM top-K alongside",
                "family_alpha": family_legend["alpha"],
                "family_p_star": family_legend["p_star"],
                "n_manmade_families": family_legend["n_manmade_families"],
                "n_named_manmade": sum(1 for rec in records if rec["family_named"]
                                       and rec["family_group"] == "manmade"),
                "family_counts": _counts(rec["family"] for rec in records),
                "score_k_meaning": "SAM angle in degrees to material_k (lower = closer)",
            })
        else:
            manifest["confidence_threshold_deg"] = spec.confidence_threshold_deg
        if spec.georef_note:
            manifest["georeference"] = spec.georef_note
        zf.writestr(f"{folder}/MANIFEST.json", json.dumps(manifest, indent=2))

    zip_filename = f"{folder}.zip"
    return zip_buf.getvalue(), zip_filename


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _short(wire_id: str) -> str:
    """First 8 chars of the uuid half of a wire id (e.g. 'action_abcd1234')."""
    if "_" in wire_id:
        wire_id = wire_id.split("_", 1)[1]
    return wire_id.replace("-", "")[:8]


def _family_fields(fam: dict, codes: dict) -> dict[str, Any]:
    """CSV / DBF family columns for one family_matches.parquet row.

    ``family`` is what the viewer shows: the family when named or natural, "unconfirmed" for
    a man-made pixel under p*, "not_scored" when nodata kept it out.
    """
    return {
        "family": codes[str(int(fam["family_code"]))]["family"],
        "family_group": fam["family_group"],
        "family_confidence": round(float(fam["confidence"]), 5),
        "family_named": int(bool(fam["answered"])),
        "best_entry": fam["best_entry"],
    }


def _namer_fields(obj: dict) -> dict[str, Any]:
    """CSV / DBF namer columns for a pixel: its object's call, class and confidence."""
    manmade = obj["call"] == "manmade"
    return {
        "nm_mm": round(float(obj["p_manmade"]), 4),
        "nm_call": obj["call"],
        "nm_cls": obj["cls"] if manmade else "",
        "nm_p": round(float(obj["p_cls"]), 4) if manmade else None,
        "nm_conf": int(bool(obj["confident"])),
        "nm_obj": int(obj["obj_id"]),
    }


def _namer_manifest(art: Path, objects: list[dict]) -> dict[str, Any]:
    """MANIFEST.json "namer" block: what named the objects and how many of each."""
    legend_path = art / "namer_legend.json"
    legend = json.loads(legend_path.read_text()) if legend_path.is_file() else {}
    manmade = [o for o in objects if o["call"] == "manmade"]
    return {
        "codename": legend.get("codename"),
        "checkpoint": legend.get("checkpoint"),
        "confident_p": legend.get("confident_p"),
        "n_objects": len(objects),
        "n_objects_manmade": len(manmade),
        "n_objects_natural": len(objects) - len(manmade),
        "class_counts": _counts(o["cls"] for o in manmade),
        "n_objects_confident": sum(1 for o in manmade if o["confident"]),
        "nm_meaning": ("per-pixel columns carry the pixel's OBJECT result; nm_cls is the closest "
                       "known class (nm_conf 0 = tentative), not a closed-world answer"),
    }


def _write_level1_geotiff_bytes(path: Path, *, crs, transform) -> bytes:
    """namer_level1_map.tif with the scene's georef (the action wrote an identity transform)."""
    import rasterio
    from rasterio.io import MemoryFile

    with rasterio.open(path) as src:
        raster = src.read(1)
        profile = dict(src.profile)
    profile.update(driver="GTiff", count=1, dtype="int8", nodata=-1, compress="lzw",
                   crs=crs, transform=transform)
    profile.pop("predictor", None)
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(raster.astype("int8"), 1)
            dst.update_tags(NAMER_LEVEL1_CODES=json.dumps(
                {"-1": "outside the anomaly mask", "0": "natural", "1": "man-made"}))
        return memfile.read()


def _write_subset_mask_bytes(level1_path: Path, subset: str, *, crs, transform) -> bytes:
    """uint8 0/1 GeoTIFF, 1 on the pixels of the subset's objects (from namer_level1_map.tif).

    No nodata: 0 is an answer ("not a man-made anomaly"), which is what a boolean map means.
    """
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile

    with rasterio.open(level1_path) as src:
        level1 = src.read(1)
        profile = dict(src.profile)
    mask = (level1 == (1 if subset == "manmade" else 0)).astype(np.uint8)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=None, compress="lzw",
                   crs=crs, transform=transform)
    profile.pop("predictor", None)
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(mask, 1)
            dst.update_tags(ANOMALY_MASK=f"1 = anomaly pixel of a {subset} object, 0 = not")
        return memfile.read()


def _objects_csv_bytes(objects: list[dict], transform, crs) -> bytes:
    """One row per object, centroid lat / lon in EPSG:4326."""
    import csv

    import rasterio
    from rasterio.warp import transform as warp

    if not objects:
        return b""
    xs, ys = zip(*(rasterio.transform.xy(transform, float(o["row"]), float(o["col"]),
                                         offset="center") for o in objects))
    # No CRS (pixel grid): nothing to project, lat / lon stay blank.
    lons, lats = (warp(crs, "EPSG:4326", list(xs), list(ys)) if crs is not None
                  else ([float("nan")] * len(xs), [float("nan")] * len(xs)))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_OBJECT_KEYS, extrasaction="ignore")
    writer.writeheader()
    for o, lon, lat in zip(objects, lons, lats):
        writer.writerow({**o, "latitude": round(lat, 7), "longitude": round(lon, 7),
                         "row": round(float(o["row"]), 2), "col": round(float(o["col"]), 2),
                         "p_manmade": round(float(o["p_manmade"]), 4),
                         "p_cls": "" if o["call"] != "manmade" else round(float(o["p_cls"]), 4),
                         "confident": int(bool(o["confident"])),
                         "fill": round(float(o["fill"]), 4)})
    return buf.getvalue().encode("utf-8")


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _write_geotiff_bytes(*, raster, profile, legend: dict, nodata: int = -1) -> bytes:
    """Write the raster to an in-memory GeoTIFF with MATERIAL_IDS tag.

    nodata: -1 for SAM's library-index raster (no match), -2 for the family-code raster,
    where -1 is a real code (not scored).
    """
    import rasterio
    from rasterio.io import MemoryFile

    out_profile = dict(profile)
    # Source is int32; downgrade to int16 if values fit (library_ix usually
    # ≤ a few thousand, well within int16 range). int16 halves the file.
    use_int16 = bool(raster.max() <= 32767 and raster.min() >= -32768)
    if use_int16:
        raster = raster.astype("int16")
        out_profile.update(dtype="int16", nodata=nodata)
    out_profile.update(driver="GTiff", count=1, compress="lzw", predictor=2)

    with MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(raster, 1)
            # Materials-as-metadata: full JSON map in a single tag. ArcGIS,
            # QGIS, and gdalinfo all surface this.
            dst.update_tags(MATERIAL_IDS=json.dumps(legend))
        return memfile.read()


def _write_shapefile_sidecars(
    *, gdf, top_k_for_shapefile: int, lead_cols: list[str] | None = None,
) -> dict[str, bytes]:
    """Write the GeoDataFrame to a temp dir as a shapefile + sidecars,
    read each file back as bytes, return a dict keyed by extension.

    lead_cols (e.g. the family columns) go right after row/col/lat/lon."""
    import tempfile

    # Trim wide attribute tables down to top-K for the shapefile (column
    # name limit is 10 chars; full top-K still lives in the CSV).
    keep_cols = ["row", "col", "latitude", "longitude", "geometry"] + list(lead_cols or [])
    for k in range(1, top_k_for_shapefile + 1):
        if f"material_{k}" in gdf.columns:
            keep_cols.append(f"material_{k}")
            keep_cols.append(f"score_{k}")
    if "confident" in gdf.columns:
        keep_cols.append("confident")
    shp_gdf = gdf[keep_cols].rename(
        columns={k: v for k, v in _SHP_COLUMN_RENAME.items() if k in keep_cols}
    )

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "shape.shp"
        shp_gdf.to_file(out_path, driver="ESRI Shapefile")
        return {
            ext: (Path(td) / f"shape.{ext}").read_bytes()
            for ext in ("shp", "shx", "dbf", "prj", "cpg")
            if (Path(td) / f"shape.{ext}").is_file()
        }


def _records_to_csv_bytes(records: list[dict], base_keys: list[str] | None = None) -> bytes:
    """All records, no geometry, every top-K column preserved.

    base_keys: leading columns; default is the SAM bundle's (thermal export shares this)."""
    import csv

    if not records:
        return b""
    base_keys = base_keys or ["row", "col", "latitude", "longitude", "confident"]
    rank_keys: list[str] = []
    max_k = 0
    for rec in records:
        for k in range(1, 20):
            if f"material_{k}" in rec and k > max_k:
                max_k = k
    for k in range(1, max_k + 1):
        rank_keys.append(f"material_{k}")
        rank_keys.append(f"score_{k}")
    keys = base_keys + rank_keys

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        rrec = {k: rec.get(k) for k in keys}
        writer.writerow(rrec)
    return buf.getvalue().encode("utf-8")
