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


def test_bundle_without_georeference_is_in_pixel_coordinates(tmp_path):
    """v1.0.1: a scene with no georeference exports on the pixel grid, no CRS and no .prj, noted in MANIFEST."""
    import json

    import geopandas as gpd

    from app.anomaly_detection.export import MissingCRSError
    from app.georef import NO_GEOREF_NOTE, pixel_grid

    mask = np.zeros((6, 5), np.uint8); mask[2, 3] = mask[4, 1] = 1
    _write(tmp_path / "anomaly_mask.tif", mask, "uint8")
    with pytest.raises(MissingCRSError):                            # no override at all: still refused
        build_thermal_bundle(ThermalExportSpec("action_x", "scene_y", "landsat9", tmp_path, "t"))
    blob, name = build_thermal_bundle(ThermalExportSpec("action_x", "scene_y", "landsat9", tmp_path, "t",
                                                        pixel_grid(), None, NO_GEOREF_NOTE))
    z = zipfile.ZipFile(io.BytesIO(blob))
    assert not any(n.endswith(".prj") for n in z.namelist())
    tif = next(n for n in z.namelist() if n.endswith("_anomaly_mask.tif"))
    with rasterio.open(io.BytesIO(z.read(tif))) as src:
        assert src.crs is None and tuple(src.transform)[:6] == (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        assert int(src.read(1).sum()) == 2
    assert json.loads(z.read(f"{name[:-4]}/MANIFEST.json"))["georeference"] == NO_GEOREF_NOTE
    for n in z.namelist():
        if "_anomaly_pixels." in n:
            (tmp_path / f"p.{n.rsplit('.', 1)[1]}").write_bytes(z.read(n))
    pts = gpd.read_file(tmp_path / "p.shp")
    assert pts.crs is None and sorted(zip(pts["row"], pts["col"])) == [(2, 3), (4, 1)]
    assert np.allclose(pts.geometry.x, pts["col"] + 0.5) and np.allclose(pts.geometry.y, -(pts["row"] + 0.5))
