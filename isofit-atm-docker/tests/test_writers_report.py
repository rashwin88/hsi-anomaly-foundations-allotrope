"""writers (GeoTIFF + BSQ submission product) and report (HTML quick-look), and the chain without ISOFIT:
readers.prep -> (stand-in for apply_oe) -> atm_post -> writers -> report."""
import json
import re
import shutil

import numpy as np
import pytest
import rasterio
from spectral.io import envi

from conftest import WHEN, WL, radiance, read_json, write_envi, write_tif
from readers import prep
from test_post import _fake_isofit


def _scene_json(w, inp, fmt="geotiff", lat=21.7):
    s = {"fid": "scn", "input": str(inp), "format": fmt, "rule": "generic", "isofit_sensor": "NA-20250601",
         "grade": "C", "centre_lat": lat, "centre_lon": 81.0, "date": "20250601", "level": "radiance",
         "pass_through": False, "pixel_m": 30.0, "lines": 60, "samples": 50, "checklist": [], "notes": []}
    (w / "scene.json").write_text(json.dumps(s))
    return w / "scene.json"


def test_write_map_georeferenced(tmp_path, rho):
    import writers
    src = write_tif(tmp_path / "in.tif", radiance(rho))
    w = tmp_path / "w"
    w.mkdir()
    rfl = str(write_envi(w / "scn_rfl", rho))
    prov = writers.write(rfl, str(_scene_json(w, src)), str(tmp_path / "out"), None)
    keep = writers.keep_bands(WL, np.ones(len(WL), bool))
    assert prov["bands_kept"] == keep.sum() and prov["georeference"] == "map"
    assert len(prov["bands_dropped_nm"]) == (~keep).sum() > 0
    with rasterio.open(tmp_path / "out" / "scn_L2_SR.tif") as d, rasterio.open(src) as s:
        assert d.count == keep.sum() and d.dtypes[0] == "int16" and d.nodata == -9999
        assert d.transform == s.transform and d.crs == s.crs
        np.testing.assert_array_equal(d.read(1)[3, 4], np.rint(rho[3, keep, 4][0] * 10000))
        assert float(d.tags(1)["wavelength"]) == pytest.approx(WL[keep][0])
    h = envi.read_envi_header(str(tmp_path / "out" / "scn_L2_SR.hdr"))
    assert len(h["wavelength"]) == keep.sum() and h["reflectance scale factor"] == "10000"
    with rasterio.open(tmp_path / "out" / "scn_L2_SR.bsq") as d:
        assert d.crs is not None and d.count == keep.sum()
    assert (tmp_path / "out" / "scn_L2_SR_provenance.json").exists()


def test_write_swath_gcps_and_no_location(tmp_path, rho):
    import writers
    src = write_tif(tmp_path / "in.tif", radiance(rho), crs=None)
    w = tmp_path / "w"
    w.mkdir()
    rfl = str(write_envi(w / "scn_rfl", rho))
    L, _, S = rho.shape
    lon, lat = np.meshgrid(81 + np.arange(S) * 3e-4, 21.7 - np.arange(L) * 3e-4)
    loc = np.stack([lon, lat, np.zeros_like(lon)], 1).astype(np.float32)   # (L, 3, S)
    write_envi(w / "scn_loc", loc, wl=np.arange(3.0), fwhm=np.ones(3), interleave="bip")
    prov = writers.write(rfl, str(_scene_json(w, src)), str(tmp_path / "o1"), None, formats=("tif",))
    assert prov["georeference"].startswith("gcps")
    with rasterio.open(tmp_path / "o1" / "scn_L2_SR.tif") as d:
        assert len(d.gcps[0]) > 4
    with rasterio.open(tmp_path / "o1" / "scn_L2_SR_latlon.tif") as d:
        assert d.read(1)[0, 0] == pytest.approx(21.7)
    prov = writers.write(rfl, str(_scene_json(w, src, lat=None)), str(tmp_path / "o2"), None, compress=False)
    assert prov["georeference"].startswith("none")


def test_report_sections_and_reference(tmp_path, rho):
    import report
    import writers
    src = write_tif(tmp_path / "in.tif", radiance(rho))
    ref = write_tif(tmp_path / "ref.tif", rho)   # the same reflectance as a vendor reference: MAE ~0
    w = tmp_path / "w"
    w.mkdir()
    rfl = str(write_envi(w / "scn_rfl", rho))
    write_envi(w / "scn_rdn", radiance(rho))
    _fake_isofit(w, rho)
    (w / "stage_times.txt").write_text("read 0\nisofit 60\nwrite 1800\nend 1830\n")
    writers.write(rfl, str(_scene_json(w, src)), str(tmp_path / "out"), None)
    page = report.build(tmp_path / "out", w, str(ref)).read_text()
    assert page.count("<img") >= 6 and "Grade C" in page and "Not available" not in page
    mae = float(re.search(r"MAE \(all bands\)</td><td>([0-9.]+)", page).group(1))
    assert mae < 1e-4
    assert "by location" in re.search(r"Pairing</td><td>([^<]*)", page).group(1)
    assert "29.0 min" in page and "30.5 min" in page   # isofit stage, total


def test_report_without_extras(tmp_path, rho):
    import report
    import writers
    w = tmp_path / "w"
    w.mkdir()
    src = write_tif(tmp_path / "in.tif", radiance(rho), crs=None)
    rfl = str(write_envi(w / "scn_rfl", rho))
    writers.write(rfl, str(_scene_json(w, src, lat=None)), str(tmp_path / "out"), None)
    page = report.build(tmp_path / "out", w, str(tmp_path / "missing.tif")).read_text()
    assert "Not available" in page                       # no atmosphere, no reference: said, not crashed
    assert page.count("<h2>") == 5


def test_report_helpers():
    import report
    assert report.band_at(WL, 860) == int(np.argmin(abs(WL - 860))) and report.band_at(WL, 5000) is None
    x = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    x[0, 0] = np.nan
    s = report.stretch(x, True)
    assert s.min() >= 0 and s.max() <= 1 and (s[0, 0] == 1).all()
    assert "<table>" in report.table([[1, 2]], ["a", "b"])


def test_chain_without_isofit(tmp_path, rho):
    """prep -> atm_post -> writers -> report on a radiance GeoTIFF; apply_oe replaced by the known surface."""
    import atm_post
    import report
    import writers
    p = write_tif(tmp_path / "scene_rad.tif", radiance(rho),
                  tags={"ACQUISITION_TIME": WHEN.isoformat(), "UNITS": "uW/(cm^2 sr nm)"})
    w = tmp_path / "work"
    assert prep.main(["--input", str(p), "--workdir", str(w)]) == 0
    s = read_json(w / "scene.json")
    fid = s["fid"]
    out = w / "isofit" / "output"
    out.mkdir(parents=True)
    write_envi(out / f"{fid}_rfl", rho)
    with rasterio.open(out / f"{fid}_atm_interp", "w", driver="ENVI", width=rho.shape[2], height=rho.shape[0],
                       count=3, dtype="float32") as d:
        for i, n in enumerate(("AOT550", "H2OSTR", "SURFACE_ELEVATION_KM")):
            d.write(np.full(rho[:, 0].shape, 0.1 * (i + 1), np.float32), i + 1)
            d.set_band_description(i + 1, n)
    assert atm_post.main(["--sensor", s["rule"], "--isofit-dir", str(w / "isofit"), "--obs", str(w / f"{fid}_obs")]) == 0
    rep = w / "isofit" / "atm_post.json"
    prov = writers.write(read_json(rep)["reflectance"], str(w / "scene.json"), str(tmp_path / "prod"), str(rep))
    assert prov["georeference"] == "map" and prov["atm_post"]["scene_h2o"] == pytest.approx(0.2)
    assert {"outlier", "snow"} <= set(prov["product"])
    page = report.build(tmp_path / "prod", w).read_text()
    assert "Not available" not in page and page.count("<img") >= 6
