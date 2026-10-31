"""readers.prep end to end on synthetic deliveries: what ISOFIT gets, scene.json, every stop and its override."""
from pathlib import Path

import numpy as np
import pytest
from spectral.io import envi

from conftest import WL, WHEN, radiance, read_json, toa_reflectance, write_envi, write_tanager, write_tif
from readers import prep

TAGS = {"ACQUISITION_TIME": WHEN.isoformat()}


def _run(inp, w, *extra):
    assert prep.main(["--input", str(inp), "--workdir", str(w), *extra]) == 0
    return read_json(w / "scene.json")


def _cube(w, fid, kind):
    return envi.open(str(w / f"{fid}_{kind}.hdr"), str(w / f"{fid}_{kind}")).load()


def test_radiance_geotiff(tmp_path, rho):
    rad = radiance(rho)
    p = write_tif(tmp_path / "scene_rad.tif", rad, tags={**TAGS, "UNITS": "uW/(cm^2 sr nm)"})
    s = _run(p, tmp_path / "w")
    w = tmp_path / "w"
    assert s["level"] == "radiance" and not s["pass_through"]
    assert s["format"] == "geotiff" and s["isofit_sensor"] == "NA-20250601" and s["rule"] == "generic"
    assert s["grade"] == "C"   # view nadir assumed, no DEM; time, location and units delivered
    assert 20 < s["centre_lat"] < 23 and s["season"] == "summer" and s["pixel_m"] == 30.0
    items = {r["item"]: r["source"] for r in s["checklist"]}
    assert items["time"] == "metadata" and items["location"] == "metadata" and items["view"] == "assumed"
    fid = s["fid"]
    r = _cube(w, fid, "rdn")
    assert r.shape == (rad.shape[0], rad.shape[2], rad.shape[1])
    np.testing.assert_allclose(np.asarray(r)[3, 4], rad[3, :, 4], rtol=1e-5)
    loc, obs = _cube(w, fid, "loc"), _cube(w, fid, "obs")
    assert loc.shape[2] == 3 and obs.shape[2] == 10
    assert np.asarray(loc)[0, 0, 2] == prep.WIDEN_M   # no DEM: the elevation grid is widened
    sza = np.asarray(obs)[..., 4]
    assert 0 < sza.min() and sza.max() < 60
    assert (w / "wavelengths.txt").read_text().count("\n") == len(WL)


def test_toa_reflectance_is_converted_to_radiance(tmp_path, rho, irradiance):
    toa = toa_reflectance(rho)
    p = write_tif(tmp_path / "scene_toa.tif", toa, tags=TAGS)
    s = _run(p, tmp_path / "w")
    assert s["level"] == "toa_reflectance" and not s["pass_through"]
    r = np.asarray(_cube(tmp_path / "w", s["fid"], "rdn"))
    i = int(np.argmin(abs(WL - 860)))
    assert 1 < np.median(r[..., i]) < 30   # uW/cm^2/sr/nm


def test_surface_reflectance_passes_through(tmp_path, rho):
    p = write_tif(tmp_path / "scene_sr.tif", rho, tags=TAGS)
    s = _run(p, tmp_path / "w")
    assert s["level"] == "surface_reflectance" and s["pass_through"]
    assert (tmp_path / "w" / f"{s['fid']}_rfl").exists()
    s2 = _run(p, tmp_path / "w2", "--input-level", "toa")   # the override corrects it anyway
    assert s2["level"] == "toa_reflectance" and not s2["pass_through"]
    note = next(r["note"] for r in s2["checklist"] if r["item"] == "level")
    assert "set by the operator" in note


def test_raw_counts_stop_and_override(tmp_path, rho):
    dn = np.rint(radiance(rho) * 50).astype(np.uint16)
    p = write_tif(tmp_path / "scene_dn.tif", dn, tags=TAGS)
    with pytest.raises(SystemExit, match="radiance-scale"):
        prep.main(["--input", str(p), "--workdir", str(tmp_path / "w")])
    s = _run(p, tmp_path / "w2", "--radiance-scale", "0.02")
    assert s["level"] == "radiance"
    r = np.asarray(_cube(tmp_path / "w2", s["fid"], "rdn"))
    # radiance is floored at 1e-4 (a zero count in a saturated water band would break ISOFIT's log terms)
    np.testing.assert_allclose(r[2, 2], np.maximum(dn[2, :, 2] * 0.02, 1e-4), rtol=1e-5)


def test_no_location_no_time(tmp_path, rho):
    stem = write_envi(tmp_path / "nometa", radiance(rho), extra=["radiance units = uW/(cm^2 sr nm)"])
    s = _run(str(stem) + ".hdr", tmp_path / "w")
    assert s["grade"] == "D" and s["centre_lat"] is None
    assert any("pseudo lat/lon" in n for n in s["notes"])
    loc = np.asarray(_cube(tmp_path / "w", s["fid"], "loc"))
    assert np.ptp(loc[..., 0]) > 0 and np.ptp(loc[..., 1]) > 0   # not one constant coordinate


def test_tanager_hdf5(tmp_path, rho):
    p = write_tanager(tmp_path / "20250601_060000_00_4001_basic_radiance_hdf5.h5", rho)
    s = _run(p, tmp_path / "w")
    assert s["format"] == "tanager_h5" and s["isofit_sensor"] == "tanager" and s["rule"] == "tanager"
    assert s["level"] == "radiance"
    items = {r["item"]: r["source"] for r in s["checklist"]}
    assert items["sun"] == "metadata" and items["view"] == "metadata" and items["time"] == "metadata"
    r = np.asarray(_cube(tmp_path / "w", s["fid"], "rdn"))
    np.testing.assert_allclose(r[1, 1], radiance(rho)[1, :, 1], rtol=1e-4)   # W/m2/sr/um x0.1


def test_tanager_sr_is_passed_through(tmp_path, rho):
    p = write_tanager(tmp_path / "20250601_060000_00_4001_basic_sr_hdf5.h5", rho, kind="surface_reflectance")
    s = _run(p, tmp_path / "w")
    assert s["pass_through"]


def test_detect(tmp_path, rho):
    p = write_tif(tmp_path / "a.tif", rho)
    assert prep.detect(p) == ("geotiff", p)
    assert prep.detect(tmp_path) == ("geotiff", p)   # a folder with one cube
    stem = write_envi(tmp_path / "b", rho)
    assert prep.detect(stem)[0] == "envi"
    with pytest.raises(SystemExit):
        prep.detect(tmp_path)   # now two candidates
    with pytest.raises(SystemExit):
        prep.detect(tmp_path / "nothing.xyz")


def test_season_widen_flat_dem(tmp_path):
    from datetime import datetime
    assert prep.season(datetime(2025, 7, 1), 20.0) == "summer"
    assert prep.season(datetime(2025, 7, 1), -20.0) == "winter"
    assert prep.season(datetime(2025, 1, 1), None) == "winter"
    p = Path(prep.flat_dem(tmp_path))
    import rasterio
    with rasterio.open(p) as d:
        assert d.crs.to_epsg() == 4326 and d.read(1, window=((0, 2), (0, 2))).max() == 0
