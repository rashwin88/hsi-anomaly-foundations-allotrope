"""Planet Tanager delivered as ENVI (.hdr + data): reader, builder, georef and source picking.

Each scene is a hand-written header plus raw bytes from numpy, so the interleave really is
what the header says and GDAL has to untangle it (the AVIRIS-NG reader misreads BSQ).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from app.georef.from_scene_file import GeorefUnavailable, resolve_scene_georef
from app.models.dataset.vendables import BandFilterConfig
from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.tanager_dataset_builder import TanagerDatasetBuilder
from app.utils.files.tanager_helper import TanagerHelper, pick_tanager_file, read_tanager_metadata
from tests.test_utils.test_tanager import FILE_WL, write_tanager

os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")

NAME = "20250205_175845_00_4001_ortho_sr"
MAP_INFO = "{UTM, 1, 1, 589050, 3653460, 30, 30, 13, North, WGS-84, units=Meters}"
C, H, W = 6, 4, 5


def _cube():
    """(C, H, W) where every voxel is unique: band b, row r, col c -> 0.01*b + 0.001*r + 0.0001*c + 0.05."""
    b, r, c = np.meshgrid(np.arange(C), np.arange(H), np.arange(W), indexing="ij")
    return (0.05 + 0.01 * b + 0.001 * r + 0.0001 * c).astype(np.float32)


def write_envi(folder: Path, cube: np.ndarray, interleave="bsq", stem=NAME, ext=".img", wavelengths=None,
               units="Nanometers", fwhm=True, bbl=None, ignore=-9999.0, map_info=True, scale=None,
               description="Tanager ortho_sr"):
    """Header + raw little-endian float32 data. `cube` is (C, H, W) in file band order."""
    folder.mkdir(parents=True, exist_ok=True)
    c, h, w = cube.shape
    wl = list(FILE_WL[:c] if wavelengths is None else wavelengths)
    lines = ["ENVI", f"description = {{{description}}}", f"samples = {w}", f"lines = {h}", f"bands = {c}",
             "header offset = 0", "file type = ENVI Standard", "data type = 4", f"interleave = {interleave}",
             "byte order = 0", f"wavelength units = {units}",
             "wavelength = {" + ", ".join(f"{v:g}" for v in wl) + "}",
             "band names = {" + ", ".join(f"b{i}" for i in range(c)) + "}"]
    if fwhm:
        lines.append("fwhm = {" + ", ".join("5.5" if units.startswith("N") else "0.0055" for _ in wl) + "}")
    if bbl is not None:
        lines.append("bbl = {" + ", ".join(str(int(v)) for v in bbl) + "}")
    if ignore is not None:
        lines.append(f"data ignore value = {ignore:g}")
    if scale is not None:
        lines.append(f"reflectance scale factor = {scale:g}")
    if map_info:
        lines.append(f"map info = {MAP_INFO}")
    (folder / f"{stem}.hdr").write_text("\n".join(lines) + "\n")
    order = {"bsq": (0, 1, 2), "bil": (1, 0, 2), "bip": (1, 2, 0)}[interleave]
    np.ascontiguousarray(cube.astype("<f4").transpose(order)).tofile(folder / f"{stem}{ext}")
    return folder / f"{stem}.hdr"


def _helper(hdr):
    return TanagerHelper(FileSourceConfig(source_path=str(hdr)))


@pytest.mark.parametrize("interleave", ["bsq", "bil", "bip"])
def test_every_interleave_reads_the_same_cube(tmp_path, interleave):
    cube = _cube()
    hdr = write_envi(tmp_path / interleave, cube, interleave=interleave, ext=f".{interleave}")
    out = _helper(hdr).extract_specific_bands(bands=[], mode="all")
    order = np.argsort(FILE_WL[:C])                       # the helper returns ascending wavelength
    assert out.shape == (C, H, W)
    assert np.array_equal(out, cube[order])


def test_micrometres_convert_to_nm(tmp_path):
    hdr = write_envi(tmp_path, _cube(), units="Micrometers", wavelengths=[w / 1000 for w in FILE_WL])
    md = read_tanager_metadata(hdr)
    assert np.allclose(md.wavelengths_nm, sorted(FILE_WL))
    assert np.allclose(md.fwhm_nm, 5.5)
    assert md.file_format == "envi" and md.interleave == "bsq"
    assert md.band_names[0] == "b1"                       # 366 nm sits second in the file


def test_bbl_and_fill_and_nan_validity(tmp_path):
    cube = _cube()
    cube[:, 0, 0] = -9999.0                               # a nodata pixel, ENVI style
    cube[2, 1, 1] = np.nan                                # file band 2 = 391 nm
    hdr = write_envi(tmp_path, cube, bbl=[1, 1, 1, 1, 1, 0])   # 2200 nm bad
    v = TanagerDatasetBuilder(FileSourceConfig(source_path=str(hdr))).vend_dataset()
    wl = v.band_cw_order
    assert not v.validity_cube[:, 0, 0].any()
    assert v.validity_cube[wl.index(391.0), 1, 1] == 0
    assert not v.validity_cube[wl.index(2200.0)].any()
    assert v.validity_cube[wl.index(1200.0), 2, 2] == 1
    assert np.isfinite(v.normalized_hyperspectral_cube).all()
    assert v.normalized_hyperspectral_cube[:, 0, 0].max() == 0.0
    assert v.band_validity_by_position == [1, 1, 1, 1, 1, 0]


def test_no_bbl_means_all_good(tmp_path):
    md = read_tanager_metadata(write_envi(tmp_path, _cube()))
    assert md.good_band == [1] * C


def test_fwhm_present_vs_derived(tmp_path):
    present = read_tanager_metadata(write_envi(tmp_path / "a", _cube()))
    derived = read_tanager_metadata(write_envi(tmp_path / "b", _cube(), fwhm=False))
    assert not present.fwhm_derived and np.allclose(present.fwhm_nm, 5.5)
    assert derived.fwhm_derived
    assert np.allclose(derived.fwhm_nm, np.gradient(np.asarray(sorted(FILE_WL))))
    assert all(f > 0 for f in derived.fwhm_nm)


def test_reflectance_scale_header_and_detected(tmp_path):
    scaled = _cube() * 10000.0
    by_header = _helper(write_envi(tmp_path / "h", scaled, scale=10000))
    detected = _helper(write_envi(tmp_path / "d", scaled))
    unit = read_tanager_metadata(write_envi(tmp_path / "u", _cube()))
    assert by_header.file_metadata.reflectance_scale_source == "header"
    assert detected.file_metadata.reflectance_scale_source == "detected_0_10000"
    assert unit.reflectance_scale_source == "detected_0_1" and unit.reflectance_scale_factor == 1.0
    order = np.argsort(FILE_WL[:C])
    for helper in (by_header, detected):
        assert np.allclose(helper.extract_specific_bands(bands=[], mode="all"), _cube()[order], atol=1e-6)


def test_quality_masks_skip_with_a_record(tmp_path):
    builder = TanagerDatasetBuilder(FileSourceConfig(source_path=str(write_envi(tmp_path, _cube()))))
    cfg = BandFilterConfig(exclusion_ranges=[], edge_bands_to_trim=0, min_valid_pixel_pct=0.0,
                           max_invalid_voxel_fraction=1.0, quality_masks_to_apply=["cloud", "cirrus"])
    v = builder.vend_dataset(band_filter_config=cfg)
    assert builder.quality_masks_skipped == ["cloud", "cirrus"]
    assert v.validity_cube.all()
    assert builder.cloud_cover_pct() is None


def test_map_info_georef_and_stac(tmp_path):
    write_envi(tmp_path / "raw", _cube())
    transform, crs = resolve_scene_georef(scene_dir=tmp_path, sensor_type="tanager", target_shape=(H, W))
    assert np.allclose(tuple(transform)[:6], (30.0, 0.0, 589050.0, 0.0, -30.0, 3653460.0))
    assert crs.to_epsg() == 32613
    builder = TanagerDatasetBuilder(FileSourceConfig(source_path=str(tmp_path / "raw" / f"{NAME}.hdr")))
    assert builder.stac_item.properties["proj:epsg"] == 32613
    assert -105 < builder.stac_item.bbox[0] < -103
    with pytest.raises(GeorefUnavailable):
        resolve_scene_georef(scene_dir=tmp_path, sensor_type="tanager", target_shape=(H + 1, W))


def test_missing_map_info_refuses_georef(tmp_path):
    write_envi(tmp_path / "raw", _cube(), map_info=False)
    with pytest.raises(GeorefUnavailable, match="no map info"):
        resolve_scene_georef(scene_dir=tmp_path, sensor_type="tanager", target_shape=(H, W))
    md = read_tanager_metadata(tmp_path / "raw" / f"{NAME}.hdr")
    assert md.grid is None
    builder = TanagerDatasetBuilder(FileSourceConfig(source_path=md.source_path))
    assert builder.stac_item.bbox is None


def test_scene_id_from_name_then_description_then_stem(tmp_path):
    by_name = read_tanager_metadata(write_envi(tmp_path / "a", _cube()))
    by_desc = read_tanager_metadata(write_envi(tmp_path / "b", _cube(), stem="export",
                                               description="Planet 20240101_120000_01_abcd ortho"))
    by_stem = read_tanager_metadata(write_envi(tmp_path / "c", _cube(), stem="my_scene", ext=""))
    assert by_name.scene_id == "20250205_175845_00_4001"
    assert by_name.acquisition_at.isoformat() == "2025-02-05T17:58:45+00:00"
    assert by_desc.scene_id == "20240101_120000_01_abcd"
    assert by_stem.scene_id == "my_scene" and by_stem.acquisition_at is None
    assert Path(by_stem.data_path).name == "my_scene"                 # data file without extension
    assert TanagerDatasetBuilder(FileSourceConfig(source_path=by_stem.source_path)).stac_item.datetime is None


def test_pick_and_source_path_prefer_hdf5(tmp_path):
    from allotrope.sensors.source_path import resolve_source_path
    from allotrope_worker.scene_onboard import _filename_for_id, _validate_for_sensor

    hdr = write_envi(tmp_path, _cube())
    (tmp_path / "orphan.hdr").write_text("ENVI\n")                    # no data file: never picked
    assert pick_tanager_file(sorted(tmp_path.iterdir())) == hdr
    assert resolve_source_path("tanager", tmp_path) == hdr
    files = sorted(p.relative_to(tmp_path) for p in tmp_path.iterdir())
    _validate_for_sensor("tanager", files)
    assert _filename_for_id("tanager", files, root=tmp_path) == "20250205_175845_00_4001"

    h5 = write_tanager(tmp_path / "ortho_sr.h5", np.zeros((2, H, W)), [500.0, 600.0], [5.5, 5.5], [1, 1],
                       np.zeros((H, W), np.uint8))
    assert resolve_source_path("tanager", tmp_path) == h5


def test_file_source_config_accepts_hdr():
    assert FileSourceConfig(source_path="x/scene.hdr").file_category is not None
