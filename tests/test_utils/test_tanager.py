"""Planet Tanager: TanagerHelper, TanagerDatasetBuilder and the georef resolver, on synthetic HDF-EOS5 files.

The synthetic files copy the real product's groups, attributes and StructMetadata.0 (item
20250205_175845_00_4001, docs/lld/tanager.md) at a few pixels, so nothing large is committed.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from app.georef.from_scene_file import GeorefUnavailable, resolve_scene_georef
from app.models.dataset.vendables import BandFilterConfig
from app.models.file_processing.sources import FileSourceConfig
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.utils.dataset_builder.tanager_dataset_builder import TanagerDatasetBuilder
from app.utils.files.tanager_helper import TanagerHelper, grid_epsg, parse_struct_metadata, pick_tanager_file

# scene_onboard imports allotrope.config, whose Settings require these (as in test_spectral_family_outputs.py).
os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")

FILL = -9999.0
STRIP = "20250205_175845_00_4001_strip"
REAL_FILE = Path("C:/es/tanager/dl/ortho_sr.h5")
REAL_TRANSFORM = (30.0, 0.0, 589050.0, 0.0, -30.0, 3653460.0)


def _struct(x_dim, y_dim, ul=(589050.0, 3653460.0), zone=13, pixel=30.0):
    lr = (ul[0] + x_dim * pixel, ul[1] - y_dim * pixel)
    return (
        'GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n\tGROUP=GRID_1\n'
        f'\t\tGridName="HYP"\n\t\tBand=4\n\t\tXDim={x_dim}\n\t\tYDim={y_dim}\n'
        f'\t\tUpperLeftPointMtrs=({ul[0]:.2f},{ul[1]:.2f})\n\t\tLowerRightMtrs=({lr[0]:.2f},{lr[1]:.2f})\n'
        f'\t\tProjection=HE5_GCTP_UTM\n\t\tZoneCode={zone}\n\t\tSphereCode=12\n'
        '\t\tPixelRegistration=HE5_HDFE_CORNER\n\t\tGridOrigin=HE5_HDFE_GD_UL\n\tEND_GROUP=GRID_1\n'
        'END_GROUP=GridStructure\n'
    ).encode()


def write_tanager(path, cube, wavelengths, fwhm, good, nodata, cloud=None, cirrus=None,
                  epsg=32613, zone=13, strip_id=STRIP, struct_shape=None):
    """A Tanager-shaped HDF-EOS5 file. `cube` is (C, H, W) in FILE band order."""
    c, h, w = cube.shape
    with h5py.File(path, "w") as f:
        grid = f.create_group("HDFEOS/GRIDS/HYP")
        if epsg is not None:
            grid.attrs["epsg_code"] = np.int64(epsg)
        if strip_id is not None:
            grid.attrs["strip_id"] = strip_id
        fields = grid.create_group("Data Fields")
        sr = fields.create_dataset("surface_reflectance", data=cube.astype(np.float32),
                                   chunks=(min(c, 3), h, w), compression="gzip")
        sr.attrs["_FillValue"] = np.float32(FILL)
        sr.attrs["wavelengths"] = np.asarray(wavelengths, dtype=np.float32)
        sr.attrs["fwhm"] = np.asarray(fwhm, dtype=np.float32)
        sr.attrs["good_wavelengths"] = np.asarray(good, dtype=np.uint8)
        sr.attrs["wavelengths_units"] = "nm"
        for name, arr in (("nodata_pixels", nodata), ("beta_cloud_mask", cloud), ("beta_cirrus_mask", cirrus)):
            arr = np.zeros((h, w), np.uint8) if arr is None else arr
            fields.create_dataset(name, data=arr.astype(np.uint8)).attrs["_FillValue"] = np.uint8(255)
        sh, sw = struct_shape or (h, w)
        f.create_group("HDFEOS INFORMATION").create_dataset(
            "StructMetadata.0", data=np.bytes_(_struct(sw, sh, zone=zone)))
    return Path(path)


# Band 1 sits out of order in the file (366 nm between 381 and 391), as one real scene did.
FILE_WL = [376.0, 366.0, 391.0, 1200.0, 1400.0, 2200.0]
FILE_FWHM = [5.4, 5.3, 5.5, 6.0, 0.0, 6.5]          # 1400 nm: no usable width
FILE_GOOD = [1, 1, 1, 1, 1, 0]                       # 2200 nm: flagged by Planet


@pytest.fixture
def scene(tmp_path):
    h, w = 4, 5
    # Each band holds its own wavelength / 10000, so ordering is checkable from the values.
    cube = np.stack([np.full((h, w), wl / 10000.0, np.float32) for wl in FILE_WL])
    nodata = np.zeros((h, w), np.uint8)
    nodata[0, 0] = 1
    cube[:, 0, 0] = FILL
    cube[2, 1, 1] = np.nan                           # file band 2 = 391 nm
    cube[3, 1, 2] = FILL                             # file band 3 = 1200 nm
    cloud = np.zeros((h, w), np.uint8)
    cloud[3, 4] = 1
    cloud[3, 3] = 255                                # fill: not cloud
    cirrus = np.zeros((h, w), np.uint8)
    cirrus[2, 4] = 1
    return write_tanager(tmp_path / "ortho_sr.h5", cube, FILE_WL, FILE_FWHM, FILE_GOOD, nodata, cloud, cirrus)


def test_file_source_config_accepts_h5():
    assert FileSourceConfig(source_path="x/ortho_sr.h5").file_category is not None


def test_helper_sorts_bands_and_cube_follows(scene):
    helper = TanagerHelper(FileSourceConfig(source_path=str(scene)))
    md = helper.file_metadata
    assert md.wavelengths_nm == sorted(FILE_WL)
    assert md.file_band_index == [1, 0, 2, 3, 4, 5]
    cube = helper.extract_specific_bands(bands=[], mode="all")
    assert np.allclose(cube[:, 3, 3], np.asarray(sorted(FILE_WL)) / 10000.0)
    assert np.allclose(helper.extract_specific_bands([0])[0, 2, 2], 0.0366)


def test_helper_identity_and_bad_fwhm(scene):
    md = TanagerHelper(FileSourceConfig(source_path=str(scene))).file_metadata
    assert md.scene_id == "20250205_175845_00_4001"
    assert md.acquisition_at.isoformat() == "2025-02-05T17:58:45+00:00"
    assert all(f > 0 for f in md.fwhm_nm)
    i1400 = md.wavelengths_nm.index(1400.0)
    assert md.good_band[i1400] == 0                  # FWHM 0 -> flagged bad
    assert md.good_band[md.wavelengths_nm.index(2200.0)] == 0


def test_vendable_validity(scene):
    builder = TanagerDatasetBuilder(FileSourceConfig(source_path=str(scene)))
    v = builder.vend_dataset()
    wl = v.band_cw_order
    assert wl == sorted(FILE_WL) and all(np.diff(wl) > 0)
    assert v.validity_cube.shape == v.normalized_hyperspectral_cube.shape == (6, 4, 5)
    assert not v.validity_cube[:, 0, 0].any()                       # nodata pixel
    assert v.validity_cube[wl.index(391.0), 1, 1] == 0                # NaN is invalid
    assert v.validity_cube[wl.index(1200.0), 1, 2] == 0               # fill is invalid
    assert not v.validity_cube[wl.index(2200.0)].any()                # bad band
    assert v.validity_cube[wl.index(391.0), 3, 3] == 1
    assert np.isfinite(v.normalized_hyperspectral_cube).all()
    assert v.normalized_hyperspectral_cube[:, 0, 0].max() == 0.0      # no -9999 left behind
    assert all(f > 0 for f in v.band_fwhm_order)
    assert v.spectral_family_order[wl.index(391.0)] == SpectralFamily.VNIR
    assert v.spectral_family_order[wl.index(1200.0)] == SpectralFamily.SWIR
    assert v.band_validity_by_position == [1, 1, 1, 1, 0, 0]


def test_band_filter_applies_cloud_and_cirrus(scene):
    builder = TanagerDatasetBuilder(FileSourceConfig(source_path=str(scene)))
    cfg = BandFilterConfig(exclusion_ranges=[], edge_bands_to_trim=0, min_valid_pixel_pct=0.0,
                           max_invalid_voxel_fraction=1.0, quality_masks_to_apply=["cloud", "cirrus"])
    v = builder.vend_dataset(band_filter_config=cfg)
    assert v.band_cw_order == [366.0, 376.0, 391.0, 1200.0]          # flagged bands dropped
    spatial = v.validity_cube.any(axis=0)
    assert not spatial[3, 4] and not spatial[2, 4]                    # cloud, cirrus
    assert spatial[3, 3]                                              # mask fill 255 is not cloud
    assert v.validity_cube[2, 1, 1] == 1                              # NaN voxel PCHIP-filled
    assert np.isfinite(v.normalized_hyperspectral_cube).all()


def test_stac_item_and_cloud_cover(scene):
    builder = TanagerDatasetBuilder(FileSourceConfig(source_path=str(scene)))
    item = builder.stac_item
    assert item.id == "20250205_175845_00_4001"
    assert item.properties["proj:epsg"] == 32613
    lon0, lat0, lon1, lat1 = item.bbox
    assert -105 < lon0 < lon1 < -103 and 32 < lat0 < lat1 < 34
    assert builder.cloud_cover_pct() == pytest.approx(100.0 / 19)     # 1 cloud pixel of 19 with data


def test_georef_pins_the_real_grid(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cube = np.zeros((2, 1020, 895), np.float32)
    write_tanager(raw / "ortho_sr.h5", cube, [500.0, 600.0], [5.5, 5.5], [1, 1], np.zeros((1020, 895), np.uint8))
    transform, crs = resolve_scene_georef(scene_dir=tmp_path, sensor_type="tanager", target_shape=(1020, 895))
    assert np.allclose(tuple(transform)[:6], REAL_TRANSFORM)
    assert crs.to_epsg() == 32613
    with pytest.raises(GeorefUnavailable):
        resolve_scene_georef(scene_dir=tmp_path, sensor_type="tanager", target_shape=(1019, 895))


def test_epsg_from_zone_when_attribute_missing():
    assert grid_epsg(parse_struct_metadata(_struct(5, 4, zone=13).decode())) == 32613
    assert grid_epsg(parse_struct_metadata(_struct(5, 4, zone=-33).decode())) == 32733


def test_grid_cube_mismatch_is_refused(tmp_path):
    path = write_tanager(tmp_path / "x.h5", np.zeros((2, 4, 5)), [500.0, 600.0], [5.5, 5.5], [1, 1],
                         np.zeros((4, 5), np.uint8), struct_shape=(4, 6))
    with pytest.raises(ValueError):
        TanagerHelper(FileSourceConfig(source_path=str(path)))


def test_pick_prefers_the_sr_file():
    files = [Path("a/ortho_radiance.h5"), Path("a/ortho_sr.h5"), Path("a/notes.json")]
    assert pick_tanager_file(files) == Path("a/ortho_sr.h5")
    assert pick_tanager_file([Path("a/scene.hdf5")]) == Path("a/scene.hdf5")
    assert pick_tanager_file([Path("a/notes.json")]) is None


def test_backend_resolves_and_names_the_scene(scene):
    from allotrope.sensors.source_path import resolve_source_path
    from allotrope_worker.scene_onboard import _filename_for_id

    assert resolve_source_path("tanager", scene.parent) == scene
    assert _filename_for_id("tanager", [Path(scene.name)], root=scene.parent) == "20250205_175845_00_4001"


def test_band_filter_default_validates_for_tanager():
    from allotrope.action_types import band_filter_apply

    cfg = dict(band_filter_apply.META.default_config_per_sensor["tanager"],
               input_scene_id="scene_00000000-0000-0000-0000-000000000000")
    out = band_filter_apply.validate_config(cfg, "tanager")
    assert out["quality_masks_to_apply"] == ["cloud", "cirrus"]
    assert out["use_common_wavelength_grid"] is False


@pytest.mark.large_files
@pytest.mark.skipif(not REAL_FILE.exists(), reason="real Tanager file not on this machine")
def test_real_file_georef():
    transform, crs = resolve_scene_georef(scene_dir=REAL_FILE.parent, sensor_type="tanager", target_shape=(1020, 895))
    assert np.allclose(tuple(transform)[:6], REAL_TRANSFORM) and crs.to_epsg() == 32613
