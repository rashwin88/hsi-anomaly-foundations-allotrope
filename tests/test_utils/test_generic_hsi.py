"""
The generic_hsi "surprise" suite: every file shape the evaluation might hand us, built in tmp_path.

Each case asserts what the vendable must satisfy whatever the input: wavelengths
ascending, FWHM > 0, the scale decision, and voxel validity. Formats (GeoTIFF,
COG, ENVI BSQ/BIL/BIP incl. OSK's .hsi, HDF5 subdatasets), band tables (um,
shuffled, descending, misspelt key, descriptions only, sidecar only, the
GHOSt dataset 3 label/position mismatch), FWHM (missing, GHOSt hint), scales
(0-1 float32/64, uint16 x10000, DN, radiance, header factor, operator
override), no-data (0, -9999, NaN, all-zero) and georef (EPSG:4326, UTM, none,
IGM swath). docs/lld/generic-hsi.md.
"""

import json

import h5py
import numpy as np
import pytest

from app.georef.from_scene_file import GeorefUnavailable, resolve_scene_georef
from app.models.dataset.vendables import BandFilterConfig
from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.generic_hsi_dataset_builder import GenericHsiDatasetBuilder
from app.utils.files.generic_hsi_georef import NO_MAP_MESSAGE
from app.utils.files.generic_hsi_helper import locate_data, pick_generic_file
from app.utils.files.generic_hsi_scale import FWHM_PER_SPACING, GHOST_FWHM_NM

from generic_hsi_files import (GEO_TRANSFORM, H, UTM_TRANSFORM, W, WL, cube_for, swath, wl_tags, write_envi,
                               write_gtiff, write_igm)


def build(path):
    return GenericHsiDatasetBuilder(FileSourceConfig(source_path=str(path)))


def check(builder, *, decision, reflectance=True, factor=None):
    """The invariants every case shares; returns the onboarding vendable."""
    md = builder.file_helper.file_metadata
    v = builder.vend_dataset()
    wl = np.asarray(v.band_cw_order)
    assert np.all(np.diff(wl) > 0), "wavelengths must be ascending"
    assert all(f > 0 for f in v.band_fwhm_order), "every FWHM > 0"
    assert md.scale_decision == decision, md.preflight["scale"]
    assert md.reflectance is reflectance
    assert v.units == (None if reflectance else "not_reflectance")
    if factor is not None:
        assert md.scale_factor == pytest.approx(factor)
    assert v.validity_cube.shape == v.normalized_hyperspectral_cube.shape
    assert md.preflight["schema"].startswith("allotrope.generic_hsi.preflight")
    return v


def assert_band_values(v, wl_sorted, scale=1.0, atol=0.012):
    """Sorted band i still carries wl_sorted[i] / 10000 (the cube_for encoding)."""
    cube = v.normalized_hyperspectral_cube
    valid = v.validity_cube.astype(bool)
    for i, w in enumerate(wl_sorted):
        vals = cube[i][valid[i]]
        assert vals.size and np.allclose(vals, w / 10000.0 * scale, atol=atol * scale), (i, w, vals[:3])


# --- formats --------------------------------------------------------------

def test_geotiff_pixel_interleave_ghost_layout(tmp_path):
    """GHOSt dataset 1: float32 0-1, per-band wavelength + misspelt units, nodata 0, a UTM grid."""
    cube = cube_for(WL)
    cube[:, 0, 0] = 0.0
    p = write_gtiff(tmp_path / "dataset1.tif", cube, band_tags=wl_tags(WL), nodata=0,
                    crs="EPSG:32613", transform=UTM_TRANSFORM)
    b = build(p)
    v = check(b, decision="reflectance_0_1", factor=1.0)
    assert_band_values(v, WL)
    assert not v.validity_cube[:, 0, 0].any() and v.validity_cube[:, 1, 1].all()
    md = b.file_helper.file_metadata
    assert md.preflight["wavelengths"]["source"] == "band_metadata"
    assert md.preflight["interleave"] == "pixel"
    assert (md.georef_source, md.epsg) == ("file", 32613)
    assert md.preflight["nodata"] == {"declared": 0.0, "source": "gdal_nodata",
                                      "also_invalid": ["NaN / inf", "-9999", "all-zero spectrum"]}
    assert b.stac_item.id == "dataset1"


def test_cog_geographic(tmp_path):
    p = write_gtiff(tmp_path / "scene.tif", cube_for(WL), band_tags=wl_tags(WL), nodata=0,
                    crs="EPSG:4326", transform=GEO_TRANSFORM, cog=True)
    b = build(p)
    v = check(b, decision="reflectance_0_1")
    assert_band_values(v, WL)
    md = b.file_helper.file_metadata
    assert (md.georef_source, md.epsg) == ("file", 4326)
    lon0, lat_top = GEO_TRANSFORM.c, GEO_TRANSFORM.f
    assert md.bbox == pytest.approx([lon0, lat_top - H * 0.0003, lon0 + W * 0.0003, lat_top])


@pytest.mark.parametrize("interleave", ["bsq", "bil", "bip"])
def test_envi_osk_hsi_every_interleave(tmp_path, interleave):
    """OSK native: `.hsi` + `.hdr` with wavelength, fwhm 8.7 and map info. GDAL reads every interleave alike."""
    hdr = write_envi(tmp_path, "osk_l2a", cube_for(WL), interleave=interleave, wl=WL, fwhm=[8.7] * WL.size,
                     header={"wavelength units": "Nanometers", "sensor type": "GHOSt"},
                     map_info="UTM, 1.0, 1.0, 589050.0, 3653460.0, 30.0, 30.0, 13, North, WGS-84")
    assert pick_generic_file(list(tmp_path.iterdir())) == hdr
    b = build(hdr)
    v = check(b, decision="reflectance_0_1")
    assert_band_values(v, WL)
    md = b.file_helper.file_metadata
    assert md.fwhm_nm == pytest.approx([8.7] * WL.size) and not md.preflight["fwhm"]["derived"]
    assert md.data_path.endswith("osk_l2a.hsi")
    assert md.epsg == 32613 and md.platform == "GHOSt"


@pytest.mark.parametrize("ext", [".img", ".bsq", ""])
def test_envi_other_data_extensions(tmp_path, ext):
    hdr = write_envi(tmp_path, "scene", cube_for(WL), ext=ext, wl=WL)
    check(build(hdr), decision="reflectance_0_1")


def test_hdf5_picks_the_cube_among_subdatasets(tmp_path):
    p = tmp_path / "product.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("Data Fields")
        rfl = g.create_dataset("surface_reflectance", data=cube_for(WL))
        rfl.attrs["wavelengths"] = WL.astype(np.float32)
        rfl.attrs["fwhm"] = np.full(WL.size, 6.0, np.float32)
        g.create_dataset("surface_reflectance_uncertainty", data=cube_for(WL) * 0.01)
        g.create_dataset("angles", data=np.zeros((3, H, W), np.float32))
        g.create_dataset("cloud_mask", data=np.zeros((H, W), np.uint8))
    b = build(p)
    v = check(b, decision="reflectance_0_1")
    assert_band_values(v, WL)
    md = b.file_helper.file_metadata
    assert "surface_reflectance" in md.data_path and "uncertainty" not in md.data_path
    assert md.fwhm_nm == pytest.approx([6.0] * WL.size)
    assert md.georef_source == "none"


def test_hdf5_wavelengths_as_a_separate_dataset(tmp_path):
    """The band table in a 1-D `/wavelength` dataset (um), not in attributes: the h5py fallback."""
    p = tmp_path / "cube.nc"
    with h5py.File(p, "w") as f:
        f.create_dataset("radiance_free_reflectance", data=cube_for(WL))
        wl = f.create_dataset("wavelength", data=WL / 1000.0)
        wl.attrs["units"] = "micrometers"
    v = check(build(p), decision="reflectance_0_1")
    assert np.allclose(v.band_cw_order, WL)


# --- band tables ----------------------------------------------------------

def test_micrometres_converted(tmp_path):
    p = write_gtiff(tmp_path / "um.tif", cube_for(WL), band_tags=wl_tags(WL / 1000.0, units="Micrometers"))
    b = build(p)
    v = check(b, decision="reflectance_0_1")
    assert np.allclose(v.band_cw_order, WL)
    assert any("micrometres" in a for a in b.file_helper.file_metadata.preflight["assumptions"])


@pytest.mark.parametrize("order", ["shuffled", "descending"])
def test_band_order_follows_wavelength_not_position(tmp_path, order):
    perm = np.random.default_rng(3).permutation(WL.size) if order == "shuffled" else np.arange(WL.size)[::-1]
    file_wl = WL[perm]
    p = write_gtiff(tmp_path / "x.tif", cube_for(file_wl), band_tags=wl_tags(file_wl))
    v = check(build(p), decision="reflectance_0_1")
    assert np.allclose(v.band_cw_order, WL)
    assert_band_values(v, WL)


def test_descriptions_only(tmp_path):
    desc = [f"Band {i + 1} ({w:.1f} nm)" for i, w in enumerate(WL)]
    desc[5] = f"Band 6 ({WL[5]:.1f})"            # GHOSt dataset 2 drops the unit on some
    b = build(write_gtiff(tmp_path / "d.tif", cube_for(WL), descriptions=desc))
    v = check(b, decision="reflectance_0_1")
    assert np.allclose(v.band_cw_order, np.round(WL, 1))
    assert b.file_helper.file_metadata.preflight["wavelengths"]["source"] == "band_descriptions"


@pytest.mark.parametrize("kind", ["csv", "json"])
def test_wavelength_sidecar_only(tmp_path, kind):
    """No band table anywhere in the cube: the uploaded wavelength file is the escape hatch."""
    p = write_gtiff(tmp_path / "bare.tif", cube_for(WL))
    if kind == "csv":
        rows = "\n".join(f"{i + 1},{w / 1000:.5f},{0.006:.4f}" for i, w in enumerate(WL))
        (tmp_path / "wavelengths.csv").write_text("band,wavelength_um,fwhm_um\n" + rows)
    else:
        (tmp_path / "bands.json").write_text(json.dumps({"wavelength": WL.tolist(), "units": "nm"}))
    b = build(p)
    v = check(b, decision="reflectance_0_1")
    assert np.allclose(v.band_cw_order, WL, atol=0.01)
    md = b.file_helper.file_metadata
    assert md.preflight["wavelengths"]["source"].startswith("wavelength_file:")
    if kind == "csv":
        assert md.fwhm_nm == pytest.approx([6.0] * WL.size)


def test_no_wavelengths_anywhere_says_what_to_upload(tmp_path):
    p = write_gtiff(tmp_path / "bare.tif", cube_for(WL))
    with pytest.raises(ValueError, match="wavelength file"):
        build(p)


def test_ghost_dataset3_label_position_mismatch(tmp_path):
    """447 of 472 bands, so position 218 is labelled `Band 224 (1396 nm)`. The metadata wavelength
    is the authority and the cube value at 1396 nm is the one stored at that position."""
    full = np.round(np.linspace(396.8, 2507.2, 472), 1)
    keep = np.r_[0:217, 223:472][:447]
    wl = full[keep]
    tags = [{"wavelength": f"{w:.6f}", "wavelenght_units": "nm"} for w in wl]
    desc = [f"Band {k + 1} ({full[k]:.6f} nm)" for k in keep]
    cube = (cube_for(wl) * 10000).astype(np.uint16)
    b = build(write_gtiff(tmp_path / "dataset3.tif", cube, band_tags=tags, descriptions=desc, nodata=0,
                          crs="EPSG:4326", transform=GEO_TRANSFORM))
    v = check(b, decision="reflectance_x10000", factor=1e-4)
    assert desc[217].startswith("Band 224")
    i = int(np.argmin(np.abs(np.asarray(v.band_cw_order) - wl[217])))
    assert v.band_cw_order[i] == pytest.approx(wl[217])
    assert np.allclose(v.normalized_hyperspectral_cube[i], cube[217] / 10000.0, atol=1e-6)


# --- FWHM -----------------------------------------------------------------

def test_missing_fwhm_is_derived_from_spacing(tmp_path):
    b = build(write_gtiff(tmp_path / "x.tif", cube_for(WL), band_tags=wl_tags(WL)))
    md = b.file_helper.file_metadata
    assert md.fwhm_nm == pytest.approx(FWHM_PER_SPACING * np.gradient(WL))
    assert md.preflight["fwhm"]["derived"] is True


def test_missing_fwhm_ghost_hint(tmp_path):
    p = write_gtiff(tmp_path / "x.tif", cube_for(WL), band_tags=wl_tags(WL),
                    root_tags={"PROVIDER": "Orbital Sidekick GHOSt L2A"})
    md = build(p).file_helper.file_metadata
    assert md.fwhm_nm == pytest.approx([GHOST_FWHM_NM] * WL.size)
    assert "GHOSt" in md.fwhm_source and md.platform == "GHOSt"


# --- scale ----------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scale_0_1(tmp_path, dtype):
    p = write_gtiff(tmp_path / "x.tif", cube_for(WL, dtype=dtype), band_tags=wl_tags(WL))
    v = check(build(p), decision="reflectance_0_1", factor=1.0)
    assert v.normalized_hyperspectral_cube.dtype == np.float32


def test_scale_uint16_x10000(tmp_path):
    p = write_gtiff(tmp_path / "x.tif", (cube_for(WL) * 10000).astype(np.uint16), band_tags=wl_tags(WL), nodata=0)
    v = check(build(p), decision="reflectance_x10000", factor=1e-4)
    assert_band_values(v, WL)


def test_scale_uint16_dn_is_not_reflectance(tmp_path):
    p = write_gtiff(tmp_path / "x.tif", (cube_for(WL) * 200000 + 15000).astype(np.uint16), band_tags=wl_tags(WL))
    v = check(build(p), decision="not_reflectance", reflectance=False)
    assert v.normalized_hyperspectral_cube.max() > 1000      # left unscaled


def test_radiance_units_are_not_reflectance(tmp_path):
    tags = [{**t, "units": "uW/(cm^2 sr nm)"} for t in wl_tags(WL)]
    b = build(write_gtiff(tmp_path / "rdn.tif", cube_for(WL), band_tags=tags))
    check(b, decision="not_reflectance", reflectance=False)
    md = b.file_helper.file_metadata
    assert md.preflight["scale"]["source"] == "metadata_units" and md.preflight["warnings"]


def test_envi_reflectance_scale_factor_wins(tmp_path):
    hdr = write_envi(tmp_path, "s", (cube_for(WL) * 1000).astype(np.int16), wl=WL,
                     header={"reflectance scale factor": "1000", "data ignore value": "-9999"})
    v = check(build(hdr), decision="reflectance_scaled", factor=1e-3)
    assert_band_values(v, WL)


@pytest.mark.parametrize("override, decision, factor, refl", [
    ("x1", "reflectance_0_1", 1.0, True),
    ("x0.0001", "reflectance_scaled", 1e-4, True),
    ("radiance", "not_reflectance", 1.0, False),
])
def test_operator_override(tmp_path, override, decision, factor, refl):
    p = write_gtiff(tmp_path / "x.tif", (cube_for(WL) * 10000).astype(np.uint16), band_tags=wl_tags(WL))
    (tmp_path / "allotrope_ingest.json").write_text(json.dumps({"scale": override}))
    b = build(p)
    check(b, decision=decision, factor=factor, reflectance=refl)
    md = b.file_helper.file_metadata
    assert md.preflight["scale"]["source"] == "operator"
    assert md.scene_id == f"x@{override}"


# --- no-data --------------------------------------------------------------

@pytest.mark.parametrize("fill, declared", [(0.0, True), (-9999.0, True), (-9999.0, False), (np.nan, False)])
def test_nodata_forms(tmp_path, fill, declared):
    cube = cube_for(WL)
    cube[:, 2, 3] = fill              # a whole pixel
    cube[7, 5, 5] = fill              # one voxel
    p = write_gtiff(tmp_path / "x.tif", cube, band_tags=wl_tags(WL), nodata=fill if declared else None)
    v = check(build(p), decision="reflectance_0_1")
    ok = v.validity_cube.astype(bool)
    assert not ok[:, 2, 3].any() and not ok[7, 5, 5]
    assert ok[:, 0, 0].all() and ok[6, 5, 5]
    assert np.isfinite(v.normalized_hyperspectral_cube).all()


def test_undeclared_all_zero_spectrum_is_nodata_but_a_lone_zero_is_not(tmp_path):
    cube = cube_for(WL)
    cube[:, 4, 4] = 0.0
    cube[3, 6, 6] = 0.0
    v = check(build(write_gtiff(tmp_path / "x.tif", cube, band_tags=wl_tags(WL))), decision="reflectance_0_1")
    assert not v.validity_cube[:, 4, 4].any() and v.validity_cube[3, 6, 6]


# --- georef ---------------------------------------------------------------

@pytest.mark.parametrize("crs, transform", [("EPSG:4326", GEO_TRANSFORM), ("EPSG:32613", UTM_TRANSFORM)])
def test_georef_from_the_file(tmp_path, crs, transform):
    write_gtiff(tmp_path / "x.tif", cube_for(WL), band_tags=wl_tags(WL), crs=crs, transform=transform)
    t, c = resolve_scene_georef(scene_dir=tmp_path, sensor_type="generic_hsi", target_shape=(H, W))
    assert t == transform and c.to_string() == crs


def test_no_georef_onboards_but_export_refuses(tmp_path):
    b = build(write_gtiff(tmp_path / "x.tif", cube_for(WL), band_tags=wl_tags(WL)))
    md = b.file_helper.file_metadata
    assert md.georef_source == "none" and md.bbox is None and b.stac_item.bbox is None
    assert md.preflight["georef"]["exportable"] is False
    with pytest.raises(GeorefUnavailable, match="no map coordinates"):
        resolve_scene_georef(scene_dir=tmp_path, sensor_type="generic_hsi", target_shape=(H, W))
    assert "no map coordinates" in NO_MAP_MESSAGE


@pytest.mark.parametrize("names", [True, False])
def test_igm_swath_georef(tmp_path, names):
    """Unrectified OSK L2A: no map info in the header, per-pixel lat/lon in `.igm`."""
    hdr = write_envi(tmp_path, "osk", cube_for(WL), wl=WL, fwhm=[8.7] * WL.size)
    lat, lon = swath()
    write_igm(tmp_path, "osk", lat, lon, names=names)
    assert pick_generic_file(list(tmp_path.iterdir())) == hdr
    md = build(hdr).file_helper.file_metadata
    assert md.georef_source == "igm" and md.epsg == 4326
    assert md.bbox == pytest.approx([lon.min(), lat.min(), lon.max(), lat.max()])
    t, c = resolve_scene_georef(scene_dir=tmp_path, sensor_type="generic_hsi", target_shape=(H, W))
    assert c.to_epsg() == 4326
    assert (t.c, t.f) == pytest.approx((lon.min(), lat.max()))


# --- band filter and the upload picker --------------------------------------

def test_band_filter_pipeline_and_skipped_masks(tmp_path):
    b = build(write_gtiff(tmp_path / "x.tif", cube_for(WL), band_tags=wl_tags(WL)))
    v = b.vend_dataset(BandFilterConfig(quality_masks_to_apply=["cloud"]))
    wl = np.asarray(v.band_cw_order)
    assert np.all(np.diff(wl) > 0) and wl.size < WL.size
    assert not ((wl >= 1350) & (wl <= 1450)).any()
    assert b.quality_masks_skipped == ["cloud"]
    assert len(v.band_fwhm_order) == wl.size


def test_locate_data_message_and_largest_cube(tmp_path):
    with pytest.raises(ValueError, match="no hyperspectral cube"):
        locate_data([write_igm(tmp_path, "g", *swath())])
    small = write_gtiff(tmp_path / "quicklook.tif", cube_for(WL[:3]))
    big = write_gtiff(tmp_path / "cube.tif", cube_for(WL))
    assert pick_generic_file([small, big]) == big


@pytest.mark.large_files
def test_real_tanager_hdf5_through_the_generic_reader():
    """Planet's ortho_sr.h5 (docs/lld/tanager.md) read as an unknown HDF5: the reflectance subdataset, Planet's
    per-band wavelengths and FWHM, -9999 fill and the HDF-EOS grid, with no Tanager code involved."""
    from pathlib import Path

    real = Path("C:/es/tanager/dl/ortho_sr.h5")
    if not real.is_file():
        pytest.skip("real Tanager file not on this machine")
    from app.utils.files.generic_hsi_helper import read_generic_hsi_metadata

    md = read_generic_hsi_metadata(real, root=real.parent)
    assert md.data_path.endswith("surface_reflectance") and md.bands == 426
    assert md.wavelengths_nm[0] == pytest.approx(376.44, abs=0.01) and np.all(np.diff(md.wavelengths_nm) > 0)
    assert md.fwhm_source == "band_metadata" and min(md.fwhm_nm) > 0
    assert (md.fill_value, md.scale_decision, md.epsg) == (-9999.0, "reflectance_0_1", 32613)
