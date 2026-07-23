"""Submission-ready export bundle for committed ``anomaly_detection_prep``.

Mirror of ``app/spectral_match/export.py`` but for the thermal case: there's
no material match, just an anomaly mask + per-pixel point cloud. Layout:

    thermal_<scene_id>_<action_id>/
        thermal_anomaly_mask.tif          # uint8 binary GeoTIFF
        thermal_anomaly_pixels.shp        # Point per anomaly pixel
        thermal_anomaly_pixels.shx
        thermal_anomaly_pixels.dbf
        thermal_anomaly_pixels.prj
        thermal_anomaly_pixels.cpg
        thermal_anomaly_summary.csv       # flat table
        MANIFEST.json                     # audit trail

Hard requirements (2026-05-14 submission rules):
  - GeoTIFF with valid CRS (refuse to ship CRS-less bundles).
  - Every path inside the zip starts with the literal ``thermal_``.
  - Shapefile sidecar set is complete.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.spectral_match.export import (
    MissingCRSError,
    _short,
    _records_to_csv_bytes,
)


# Shapefile column-name rename.
_SHP_COLUMN_RENAME = {
    "latitude":  "lat",
    "longitude": "lon",
}


@dataclass(frozen=True)
class ThermalExportSpec:
    action_id: str
    scene_id: str
    sensor_type: str
    artifact_dir: Path     # committed prep's output dir; must contain anomaly_mask.tif
    software_version: str
    # Caller-supplied georef override (resolved from raw scene file) when
    # the action's anomaly_mask.tif itself has no CRS. Mirrors the hyper path.
    override_transform: Any = None     # rasterio.transform.Affine
    override_crs: Any = None           # rasterio.crs.CRS


def build_thermal_bundle(spec: ThermalExportSpec) -> tuple[bytes, str]:
    """Build the thermal submission zip in-memory. Returns (bytes, filename).

    The prep action has these output files we rely on (under spec.artifact_dir):
      anomaly_mask.tif       — uint8 binary GeoTIFF (written at commit time)
      composite_score.tif    — float32 score raster (used for per-pixel score)
      metrics.json           — threshold + dilation + P/R/F1 (if GT)
    """
    import numpy as np
    import rasterio
    import geopandas as gpd
    from shapely.geometry import Point

    art = spec.artifact_dir
    mask_path = art / "anomaly_mask.tif"
    composite_path = art / "composite_score.tif"
    metrics_path = art / "metrics.json"

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"missing {mask_path}; the prep action must be committed "
            f"before export."
        )

    # ---- 1. Read mask + apply georef override if source has none ----
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask_profile = src.profile
        src_crs = src.crs
        src_transform = src.transform

    if src_crs is None:
        if spec.override_crs is not None and spec.override_transform is not None:
            src_crs = spec.override_crs
            src_transform = spec.override_transform
            # Rewrite the mask with the resolved georef before shipping it.
            from rasterio.io import MemoryFile
            out_profile = dict(mask_profile)
            out_profile.update(crs=src_crs, transform=src_transform)
            with MemoryFile() as memfile:
                with memfile.open(**out_profile) as dst:
                    dst.write(mask, 1)
                mask_bytes = memfile.read()
        else:
            raise MissingCRSError(
                f"anomaly_mask.tif at {mask_path} has no CRS and no "
                f"override was provided; the source vendable for scene "
                f"{spec.scene_id} was onboarded without georeference. "
                f"Submission would be disqualified."
            )
    else:
        # Source already has CRS — copy verbatim.
        mask_bytes = mask_path.read_bytes()

    # ---- 2. Read the composite score (so each pixel gets a score) ---
    composite: np.ndarray | None = None
    if composite_path.is_file():
        with rasterio.open(composite_path) as src2:
            composite = src2.read(1)
            if composite.shape != mask.shape:
                composite = None  # mismatched; ignore rather than misalign

    # ---- 3. Per-pixel records ---------------------------------------
    anomaly_yx = np.argwhere(mask.astype(bool))   # rows, cols of True
    records: list[dict[str, Any]] = []
    for r, c in anomaly_yx:
        lon, lat = rasterio.transform.xy(
            src_transform, int(r), int(c), offset="center",
        )
        rec: dict[str, Any] = {
            "row": int(r),
            "col": int(c),
            "latitude": float(lat),
            "longitude": float(lon),
            "score": (
                float(composite[int(r), int(c)])
                if composite is not None else None
            ),
            "geometry": Point(lon, lat),
        }
        records.append(rec)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=src_crs)

    # Lat/lon attributes always in 4326 for the table; geometry stays native.
    if str(gdf.crs).lower() not in ("epsg:4326",):
        gdf_4326 = gdf.to_crs("EPSG:4326")
        gdf["latitude"] = gdf_4326.geometry.y
        gdf["longitude"] = gdf_4326.geometry.x

    # ---- 4. Build the zip --------------------------------------------
    folder = f"thermal_{_short(spec.scene_id)}_{_short(spec.action_id)}"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 4a. Mask GeoTIFF — copied verbatim. Already georef'd + compressed
        # by the commit step.
        zf.writestr(f"{folder}/thermal_anomaly_mask.tif", mask_bytes)

        # 4b. Shapefile + sidecars.
        shp_files = _write_thermal_shapefile_sidecars(gdf)
        for ext, blob in shp_files.items():
            zf.writestr(f"{folder}/thermal_anomaly_pixels.{ext}", blob)

        # 4c. CSV.
        zf.writestr(
            f"{folder}/thermal_anomaly_summary.csv",
            _records_to_csv_bytes(
                [{k: v for k, v in r.items() if k != "geometry"} for r in records],
            ),
        )

        # 4d. Manifest.
        metrics: dict | None = None
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text())
            except (OSError, ValueError):
                metrics = None

        manifest = {
            "schema": "allotrope-export-thermal/1",
            "action_id": spec.action_id,
            "scene_id": spec.scene_id,
            "sensor_type": spec.sensor_type,
            "software_version": spec.software_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "crs_wkt": src_crs.to_wkt(),
            "n_anomalous_pixels": len(records),
            "metrics": metrics,
            "report_required": True,
            "report_note": (
                "Submission requires a Word-doc report describing experiments, "
                "experiences, and approaches (30% of marks). Author manually."
            ),
        }
        zf.writestr(f"{folder}/MANIFEST.json", json.dumps(manifest, indent=2))

    return zip_buf.getvalue(), f"{folder}.zip"


def _write_thermal_shapefile_sidecars(gdf) -> dict[str, bytes]:
    import tempfile

    keep_cols = ["row", "col", "latitude", "longitude", "score", "geometry"]
    keep_cols = [c for c in keep_cols if c in gdf.columns]
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
