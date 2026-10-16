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
        MANIFEST.json                       # audit trail

Hard requirements from the submission rules:
  - GeoTIFF must carry a valid CRS. We refuse to ship a CRS-less bundle
    (raises ``MissingCRSError`` — the api translates that to a 422).
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

Two bundle layouts, chosen by what the action wrote (schema in MANIFEST.json):

  - ``allotrope-export-hyper/2`` — actions with family matching
    (``family_matches.parquet``). The raster holds family codes (-2 outside the mask,
    -1 not scored, 0 man-made type unconfirmed, 1..F a family) and the legend maps them.
    One point per anomaly pixel; DBF ``family, group, fconf, named, best_ent`` then SAM's
    ``mat1/scr1 … mat3/scr3`` (scr = SAM angle, degrees). ``hyper_sam_legend.json`` keeps
    the SAM library legend.
  - ``allotrope-export-hyper/1`` — SAM-only actions from before family matching: top-1
    library index raster, ``conf = angle < confidence_threshold_deg``. Unchanged.
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
    override_crs: Any = None           # rasterio.crs.CRS


def build_hyper_bundle(spec: ExportSpec) -> tuple[bytes, str]:
    """Build the submission zip in-memory. Returns (zip_bytes, filename).

    Raises ``MissingCRSError`` if the source GeoTIFF has no CRS.
    Raises ``FileNotFoundError`` if required input artifacts are missing.
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
        if spec.override_crs is not None and spec.override_transform is not None:
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
    # Family bundles: one point per anomaly pixel, SAM ranks attached where SAM matched.
    # SAM bundles: one point per SAM-matched pixel.
    if is_family:
        pixels = [(int(f["row"]), int(f["col"]), f)
                  for f in pq.read_table(family_files[0]).to_pylist()]
    else:
        pixels = [(r, c, None) for (r, c) in by_pixel]
    records: list[dict[str, Any]] = []
    for r, c, fam in pixels:
        # Pixel-centre lon/lat from the source raster's affine.
        lon, lat = rasterio.transform.xy(src_transform, r, c, offset="center")
        rec: dict[str, Any] = {
            "row": int(r),
            "col": int(c),
            "latitude": float(lat),
            "longitude": float(lon),
        }
        if fam is not None:
            rec.update(_family_fields(fam, family_legend["codes"]))
        ranks = by_pixel.get((r, c), [])
        for k, hit in enumerate(ranks):
            rec[f"material_{k+1}"] = hit["name"]
            rec[f"score_{k+1}"] = round(float(hit["angle_deg"]), 5)
        if fam is None:
            rec["confident"] = int(
                float(ranks[0]["angle_deg"]) < spec.confidence_threshold_deg
            )
        rec["geometry"] = Point(lon, lat)
        records.append(rec)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=src_crs)

    # Reproject to EPSG:4326 if we're not already there — common GIS
    # convention for shapefile/CSV, and matches what `latitude/longitude`
    # field names imply.
    if str(gdf.crs).lower() not in ("epsg:4326",):
        # Compute lat/lon in EPSG:4326 explicitly for the attribute table,
        # but keep the geometry in the native CRS so it overlays the
        # source raster correctly in QGIS/ArcGIS.
        # Decision: geometry stays in source CRS (matches the GeoTIFF).
        # Attribute lat/lon are reported in 4326 for human use.
        gdf_4326 = gdf.to_crs("EPSG:4326")
        gdf["latitude"] = gdf_4326.geometry.y
        gdf["longitude"] = gdf_4326.geometry.x

    # ---- 4. Assemble the zip in-memory -------------------------------
    folder = f"hyper_{_short(spec.scene_id)}_{_short(spec.action_id)}"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # 4a. GeoTIFF with MATERIAL_IDS tag
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
            lead_cols=_FAMILY_KEYS[4:] if is_family else [],
        )
        for fname, blob in shp_files.items():
            zf.writestr(f"{folder}/hyper_anomaly_materials.{fname}", blob)

        # 4c. Legend JSON
        zf.writestr(
            f"{folder}/hyper_material_legend.json",
            json.dumps(family_legend if is_family else legend, indent=2),
        )
        if is_family:
            zf.writestr(f"{folder}/hyper_sam_legend.json", json.dumps(sam_legend, indent=2))

        # 4d. Flat CSV (no geometry, all top-K)
        csv_bytes = _records_to_csv_bytes(
            records, base_keys=_FAMILY_KEYS if is_family else None)
        zf.writestr(f"{folder}/hyper_match_summary.csv", csv_bytes)

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
