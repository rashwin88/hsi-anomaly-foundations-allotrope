"""readers.cube (ENVI / GeoTIFF access, interleave check, band order) and readers.geometry."""
from datetime import datetime, timezone

import numpy as np
import pytest

from conftest import FWHM, WL, radiance, write_envi, write_tif
from readers import cube, geometry


@pytest.mark.parametrize("inter", ["bil", "bsq", "bip"])
def test_open_envi_each_layout(tmp_path, rho, inter):
    stem = write_envi(tmp_path / f"c_{inter}", rho, interleave=inter)
    info = cube.open_envi(str(stem) + ".hdr")
    assert info.interleave == inter and info.shape == rho.shape
    np.testing.assert_allclose(info.read_rows(5, 9), rho[5:9])
    assert info.transform is None and not info.notes


def test_wrong_interleave_in_header_is_caught(tmp_path, rho):
    stem = write_envi(tmp_path / "c", rho, interleave="bsq")
    hdr = (tmp_path / "c.hdr")
    hdr.write_text(hdr.read_text().replace("interleave = bsq", "interleave = bil"))
    info = cube.open_envi(str(hdr))
    assert info.interleave == "bsq"
    assert any("header says interleave" in n for n in info.notes)
    np.testing.assert_allclose(info.read_rows(0, 3), rho[0:3])


def test_envi_ignore_value_and_size_check(tmp_path, rho):
    x = rho.copy()
    x[0, :, 0] = -9999
    stem = write_envi(tmp_path / "c", x)
    assert np.isnan(cube.open_envi(str(stem) + ".hdr").read_rows(0, 1)[0, :, 0]).all()
    (tmp_path / "c").write_bytes(b"\0" * 10)
    with pytest.raises(ValueError, match="header implies"):
        cube.open_envi(str(stem) + ".hdr")


def test_envi_without_wavelengths_needs_a_file(tmp_path, rho):
    stem = write_envi(tmp_path / "c", rho)
    hdr = tmp_path / "c.hdr"
    hdr.write_text("\n".join(l for l in hdr.read_text().splitlines() if not l.startswith(("wavelength =", "fwhm"))))
    with pytest.raises(ValueError, match="wavelength"):
        cube.open_envi(str(hdr))
    wf = tmp_path / "wl.txt"
    np.savetxt(wf, np.c_[WL / 1000, FWHM / 1000])   # um, two columns
    info = cube.open_envi(str(hdr), str(wf))
    np.testing.assert_allclose(info.wl, WL)
    np.testing.assert_allclose(info.fwhm, FWHM)
    with pytest.raises(ValueError, match="rows"):
        cube.load_wl_file(str(wf), 5)


def test_open_geotiff_with_tags(tmp_path, rho):
    p = write_tif(tmp_path / "c.tif", rho, nodata=-9999.0)
    info = cube.open_geotiff(str(p))
    np.testing.assert_allclose(info.wl, WL)
    np.testing.assert_allclose(info.fwhm, FWHM)
    assert info.crs.to_epsg() == 32644
    np.testing.assert_allclose(info.read_rows(0, 2), rho[0:2])
    lon, lat = cube.lonlat_reader(info.transform, info.crs)(0, 2, rho.shape[2])
    assert lon.shape == (2, rho.shape[2]) and 20 < lat.mean() < 23 and 80 < lon.mean() < 82


def test_geotiff_wavelength_from_descriptions(tmp_path, rho):
    import rasterio
    p = tmp_path / "d.tif"
    L, B, S = rho.shape
    with rasterio.open(p, "w", driver="GTiff", width=S, height=L, count=B, dtype="float32") as d:
        d.write(rho.transpose(1, 0, 2))
        for i in range(B):
            d.set_band_description(i + 1, f"Band {i + 1} ({WL[i] / 1000:.3f} um)")
    info = cube.open_geotiff(str(p))
    np.testing.assert_allclose(info.wl, WL, atol=0.01)
    assert info.fwhm is None and info.crs is None


def test_sort_bands_and_typo_repair(tmp_path, rho):
    p = write_tif(tmp_path / "rev.tif", rho[:, ::-1, :].copy(), wl=WL[::-1], fwhm=FWHM)
    info = cube.sort_bands(cube.open_geotiff(str(p)))
    np.testing.assert_allclose(info.wl, WL)
    np.testing.assert_allclose(info.read_rows(0, 1), rho[0:1])
    notes = []
    wl = WL.copy()
    wl[10] = 1.0
    fixed = cube.fix_order(wl, notes)
    assert fixed[10] == pytest.approx(WL[10]) and notes


def test_derive_fwhm_keeps_gap_edges():
    wl = np.r_[np.arange(400, 500, 10.0), np.arange(600, 700, 10.0)]
    f = cube.derive_fwhm(wl)
    assert np.allclose(f, 10.0)


def test_find_hdr(tmp_path, rho):
    stem = write_envi(tmp_path / "cube", rho)
    assert cube.find_hdr(str(stem)) == str(stem) + ".hdr"
    assert cube.find_hdr(str(tmp_path / "missing")) is None


def test_sun_position_noon_equinox():
    t = datetime(2025, 3, 20, 12, 0, tzinfo=timezone.utc)
    z, a = geometry.sun_position(t, np.array([0.0]), np.array([0.0]))
    assert z[0] < 3.0
    z, a = geometry.sun_position(t, np.array([45.0]), np.array([0.0]))
    assert 43 < z[0] < 47 and 150 < a[0] < 210


def test_earth_sun_distance():
    assert geometry.earth_sun_au(None) == 1.0
    assert geometry.earth_sun_au(datetime(2025, 1, 4)) == pytest.approx(0.9833, abs=1e-3)
    assert geometry.earth_sun_au(datetime(2025, 7, 4)) == pytest.approx(1.0167, abs=1e-3)


def test_solar_irradiance(irradiance):
    from conftest import solar
    e = geometry.solar_irradiance(WL, FWHM)
    np.testing.assert_allclose(e, solar(WL), rtol=0.02)


def test_solar_irradiance_missing(monkeypatch):
    monkeypatch.setattr(geometry, "ISOFIT_IRR", "/nonexistent/irr.dat")
    monkeypatch.setattr(geometry.glob, "glob", lambda *a, **k: [])
    with pytest.raises(FileNotFoundError):
        geometry.solar_irradiance(WL, FWHM)


def test_sample_dem(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    lon, lat = np.meshgrid(np.linspace(81.0, 81.1, 5), np.linspace(21.7, 21.6, 4))
    z, note = geometry.sample_dem(None, lon, lat)
    assert not z.any() and "no DEM" in note
    p = tmp_path / "dem.tif"
    with rasterio.open(p, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(80.95, 21.75, 0.02, 0.02), nodata=-32768) as d:
        d.write(np.full((10, 10), 250.0, np.float32), 1)
    z, note = geometry.sample_dem(str(p), lon, lat)
    assert np.allclose(z, 250.0) and note.startswith("DEM")
    z, note = geometry.sample_dem(str(p), lon + 10, lat)
    assert not z.any() and "outside" in note


def test_slope_aspect_flat_and_tilted():
    lon, lat = np.meshgrid(np.linspace(81, 81.01, 20), np.linspace(21.71, 21.70, 20))
    s, a, ci = geometry.slope_aspect(lat, lon, np.zeros(lon.shape), 30.0, 180.0)
    assert np.allclose(s, 0, atol=1e-6) and np.allclose(ci, np.cos(np.radians(30)))
    z = (lon - 81) * 111000 * 0.1   # rises to the east: faces west
    s, a, ci = geometry.slope_aspect(lat, lon, z, 30.0, 180.0)
    assert 4 < np.median(s) < 8 and 260 < np.median(a) < 280


def test_obs_bands_order():
    one = np.ones((2, 3))
    b = geometry.obs_bands(one * 5e5, 0, 5, 180, 30, one * 0, one * 0, one, 6.0)
    assert len(b) == len(geometry.OBS_NAMES) == 10
    assert np.allclose(b[5], 35.0, atol=0.5)   # phase = sza + vza for opposite azimuths


def test_radiance_fixture_is_plausible(rho):
    r = radiance(rho)
    assert 1 < np.median(r[:, np.argmin(abs(WL - 860)), :]) < 20
