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
}


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

    # ---- 1. Read the match map and (if needed) override georef -----
    with rasterio.open(match_map_tif) as src:
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
    legend = json.loads(legend_json.read_text())  # already keyed by library_ix

    # ---- 3. Build features ------------------------------------------
    records: list[dict[str, Any]] = []
    for (r, c), ranks in by_pixel.items():
        # Pixel-centre lon/lat from the source raster's affine.
        lon, lat = rasterio.transform.xy(src_transform, r, c, offset="center")
        rec: dict[str, Any] = {
            "row": int(r),
            "col": int(c),
            "latitude": float(lat),
            "longitude": float(lon),
        }
        for k, hit in enumerate(ranks):
            rec[f"material_{k+1}"] = hit["name"]
            rec[f"score_{k+1}"] = round(float(hit["angle_deg"]), 5)
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
        )
        zf.writestr(f"{folder}/hyper_anomaly_materials.tif", tif_bytes)

        # 4b. Shapefile + sidecars (geopandas writes them to a tmpdir;
        # we read each back as bytes and stuff into the zip).
        shp_files = _write_shapefile_sidecars(
            gdf=gdf,
            top_k_for_shapefile=3,
        )
        for fname, blob in shp_files.items():
            zf.writestr(f"{folder}/hyper_anomaly_materials.{fname}", blob)

        # 4c. Legend JSON
        zf.writestr(
            f"{folder}/hyper_material_legend.json",
            json.dumps(legend, indent=2),
        )

        # 4d. Flat CSV (no geometry, all top-K)
        csv_bytes = _records_to_csv_bytes(records)
        zf.writestr(f"{folder}/hyper_match_summary.csv", csv_bytes)

        # 4e. Manifest
        manifest = {
            "schema": "allotrope-export-hyper/1",
            "action_id": spec.action_id,
            "scene_id": spec.scene_id,
            "sensor_type": spec.sensor_type,
            "splib07_version": spec.splib_version,
            "software_version": spec.software_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "crs_wkt": src_crs.to_wkt() if src_crs else None,
            "confidence_threshold_deg": spec.confidence_threshold_deg,
            "n_matched_pixels": len(records),
            "n_unique_materials": len({rec["material_1"] for rec in records}),
            "report_required": True,
            "report_note": (
                "Submission requires a Word-doc report describing experiments, "
                "experiences, and approaches (30% of marks). Author manually."
            ),
        }
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


def _write_geotiff_bytes(*, raster, profile, legend: dict) -> bytes:
    """Write the raster to an in-memory GeoTIFF with MATERIAL_IDS tag."""
    import rasterio
    from rasterio.io import MemoryFile

    out_profile = dict(profile)
    # Source is int32; downgrade to int16 if values fit (library_ix usually
    # ≤ a few thousand, well within int16 range). int16 halves the file.
    use_int16 = bool(raster.max() <= 32767 and raster.min() >= -32768)
    if use_int16:
        raster = raster.astype("int16")
        out_profile.update(dtype="int16", nodata=-1)
    out_profile.update(driver="GTiff", count=1, compress="lzw", predictor=2)

    with MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(raster, 1)
            # Materials-as-metadata: full JSON map in a single tag. ArcGIS,
            # QGIS, and gdalinfo all surface this.
            dst.update_tags(MATERIAL_IDS=json.dumps(legend))
        return memfile.read()


def _write_shapefile_sidecars(
    *, gdf, top_k_for_shapefile: int,
) -> dict[str, bytes]:
    """Write the GeoDataFrame to a temp dir as a shapefile + sidecars,
    read each file back as bytes, return a dict keyed by extension."""
    import tempfile

    # Trim wide attribute tables down to top-K for the shapefile (column
    # name limit is 10 chars; full top-K still lives in the CSV).
    keep_cols = ["row", "col", "latitude", "longitude", "geometry"]
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


def _records_to_csv_bytes(records: list[dict]) -> bytes:
    """All records, no geometry, every top-K column preserved."""
    import csv

    if not records:
        return b""
    base_keys = ["row", "col", "latitude", "longitude", "confident"]
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
