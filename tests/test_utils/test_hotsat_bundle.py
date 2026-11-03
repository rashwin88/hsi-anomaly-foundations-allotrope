"""A HotSat-1 scene folder through FileSourceConfig, the export georef and onboarding's pre-flight id.

The upload failed at FileSourceConfig (no category for the bundle folder), the georef could pick
the stacked raster, whose grid differs from the ortho the cube is built from, and the pre-flight
id kept the ortho's ``_frame025`` in it. Synthetic bundles mirror SatVu's L2 Visual layout.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")

from app.georef.from_scene_file import resolve_scene_georef  # noqa: E402
from app.models.file_processing.file_categories import FileCategory  # noqa: E402
from app.models.file_processing.sources import FileSourceConfig  # noqa: E402
from allotrope_worker.scene_onboard import _filename_for_id  # noqa: E402

SCENE_ID = "20231024T042615000_visual_30_hotsat1"
ORTHO = Affine(3.665687, 0.0, 702923.04, 0.0, -3.665687, 8632314.25)
STACKED = Affine(3.5, 0.0, 702900.0, 0.0, -3.5, 8632400.0)
ORTHO_SHAPE, STACKED_SHAPE = (12, 10), (14, 11)


def _tiff(path: Path, shape, transform):
    with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                       dtype="uint16", crs=CRS.from_epsg(32752), transform=transform) as dst:
        dst.write(np.full(shape, 5000, dtype=np.uint16), 1)


def _bundle(folder: Path, with_metadata: bool = True) -> Path:
    folder.mkdir(parents=True)
    _tiff(folder / f"{SCENE_ID}_frame025_ortho.tiff", ORTHO_SHAPE, ORTHO)
    _tiff(folder / f"{SCENE_ID}_stacked.tiff", STACKED_SHAPE, STACKED)
    _tiff(folder / f"{SCENE_ID}_udm.tiff", ORTHO_SHAPE, ORTHO)
    if with_metadata:
        (folder / "metadata.json").write_text(json.dumps({
            "id": SCENE_ID, "bbox": [130.0, -12.4, 130.1, -12.3],
            "properties": {"proj:epsg": 32752, "proj:shape": list(ORTHO_SHAPE),
                           "proj:transform": list(ORTHO)[:6], "proj:bbox": [0, 0, 1, 1]},
            "assets": {"visual": {"href": f"./{SCENE_ID}_frame025_ortho.tiff"},
                       "experimental_stacked": {"href": f"./{SCENE_ID}_stacked.tiff"},
                       "udm": {"href": f"./{SCENE_ID}_udm.tiff"}},
        }))
    return folder


def test_a_hotsat_folder_is_a_hotsat_folder(tmp_path):
    folder = _bundle(tmp_path / SCENE_ID)
    assert FileSourceConfig(source_path=str(folder)).file_category == FileCategory.HOTSAT_FOLDER


def test_metadata_json_without_a_hotsat_raster_is_not_a_hotsat_folder(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "scene.tiff").write_bytes(b"")
    with pytest.raises(ValueError, match="Unrecognized"):
        FileSourceConfig(source_path=str(tmp_path))


def test_envi_and_enmap_folders_keep_their_categories(tmp_path):
    envi = tmp_path / "ang20180101t000000"
    envi.mkdir()
    (envi / "metadata.json").write_text("{}")
    (envi / "ang20180101t000000_corr_v2m2_img.hdr").write_text("ENVI\n")
    assert FileSourceConfig(source_path=str(envi)).file_category == FileCategory.ENVI_FOLDER
    enmap = tmp_path / "ENMAP01-____L2A-DT0000001_20230101T000000Z_001_V010400_20230102T000000Z"
    enmap.mkdir()
    (enmap / f"{enmap.name}-METADATA.XML").write_text("<x/>")
    assert FileSourceConfig(source_path=str(enmap)).file_category == FileCategory.ENMAP_FOLDER


@pytest.mark.parametrize("with_metadata", [True, False])
def test_georef_is_the_ortho_grid_never_the_stacked_one(tmp_path, with_metadata):
    raw = tmp_path / "raw"
    _bundle(raw / SCENE_ID, with_metadata=with_metadata)
    transform, crs = resolve_scene_georef(scene_dir=raw, sensor_type="hotsat1", target_shape=ORTHO_SHAPE)
    assert crs.to_epsg() == 32752
    assert transform.almost_equals(ORTHO)


def test_preflight_id_is_the_builders_id(tmp_path):
    folder = _bundle(tmp_path / "upload")
    files = [p.relative_to(tmp_path) for p in sorted(folder.iterdir())]
    assert _filename_for_id("hotsat1", files, root=tmp_path) == SCENE_ID


@pytest.mark.parametrize("name", [f"{SCENE_ID}_frame025_ortho.tiff", f"{SCENE_ID}_stacked.tiff",
                                  f"{SCENE_ID}_udm.tiff"])
def test_preflight_id_without_metadata_strips_the_raster_suffix(name):
    assert _filename_for_id("hotsat1", [Path(name)]) == SCENE_ID
