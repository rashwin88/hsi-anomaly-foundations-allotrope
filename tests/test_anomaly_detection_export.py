"""Prep submission bundle (app/anomaly_detection/export.py): modality prefix, 0/1 mask with the resolved grid, and a
shapefile with no score attribute (AIGC evaluation rules, 2026-10-31)."""

import io
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.anomaly_detection.export import ThermalExportSpec, build_thermal_bundle

GRID = Affine(30.0, 0.0, 545007.75, 0.0, -30.0, 2501437.5)


def _write(path, arr, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype) as dst:                      # no CRS, identity transform: as the commit writes
        dst.write(arr.astype(dtype), 1)


@pytest.mark.parametrize("sensor,prefix", [("prisma", "hyper"), ("enmap", "hyper"), ("tanager", "hyper"), ("landsat9", "thermal")])
def test_bundle_is_named_by_modality_and_ships_a_boolean_mask(tmp_path, sensor, prefix):
    mask = np.zeros((6, 5), np.uint8); mask[2, 3] = mask[4, 1] = 1
    _write(tmp_path / "anomaly_mask.tif", mask, "uint8")
    _write(tmp_path / "composite_score.tif", np.random.rand(6, 5), "float32")
    blob, name = build_thermal_bundle(ThermalExportSpec("action_x", "scene_y", sensor, tmp_path, "t",
                                                        GRID, CRS.from_epsg(32642)))
    assert name.startswith(f"{prefix}_")
    z = zipfile.ZipFile(io.BytesIO(blob))
    assert all(n.split("/")[-1] == "MANIFEST.json" or n.split("/")[-1].startswith(f"{prefix}_") for n in z.namelist())
    tif = next(n for n in z.namelist() if n.endswith("_anomaly_mask.tif"))
    with rasterio.open(io.BytesIO(z.read(tif))) as src:
        assert src.crs.to_epsg() == 32642 and src.transform == GRID
        assert set(np.unique(src.read(1))) == {0, 1}
    dbf = next(n for n in z.namelist() if n.endswith(".dbf"))
    assert b"SCORE" not in z.read(dbf).upper()[:600]                # field descriptors live in the header
