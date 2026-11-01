"""rfl_qc (window mask, outliers, snow), atm_post on a fake ISOFIT run, aot_prior, sixs_parts.scene_state."""
import json

import numpy as np
import pytest
from spectral.io import envi

from conftest import WL, write_envi


def _fake_isofit(tmp_path, rfl_lbs, fid="scn", aot=0.2, h2o=1.5, elev=0.3, slope=0.0):
    """<tmp>/isofit/output/<fid>_rfl + _atm_interp, and an obs cube: what atm_post reads after apply_oe."""
    import rasterio
    out = tmp_path / "isofit" / "output"
    out.mkdir(parents=True)
    write_envi(out / f"{fid}_rfl", rfl_lbs)
    L, _, S = rfl_lbs.shape
    with rasterio.open(out / f"{fid}_atm_interp", "w", driver="ENVI", width=S, height=L, count=3, dtype="float32") as d:
        for i, (n, v) in enumerate((("AOT550", aot), ("H2OSTR", h2o), ("SURFACE_ELEVATION_KM", elev))):
            d.write(np.full((L, S), v, np.float32), i + 1)
            d.set_band_description(i + 1, n)
    obs = np.zeros((L, 10, S), np.float32)
    obs[:, 4], obs[:, 6], obs[:, 8] = 30.0, slope, 1.0
    write_envi(tmp_path / f"{fid}_obs", obs, wl=np.arange(10.0), fwhm=np.ones(10), interleave="bip")
    return str(out / f"{fid}_rfl"), str(tmp_path / f"{fid}_obs")


def test_mask_windows(tmp_path, rho):
    import rfl_qc
    p = str(write_envi(tmp_path / "r", rho.copy()))
    n = rfl_qc.mask_windows(p)
    outside = ~np.any([(WL >= a) & (WL <= b) for a, b in rfl_qc.INVERSION_WINDOWS_NM], 0)
    assert n == outside.sum() > 0
    x = envi.open(p + ".hdr", p).open_memmap(interleave="bil")
    assert np.all(x[:, outside, :] == rfl_qc.NODATA) and np.all(x[:, ~outside, :] > rfl_qc.NODATA)


def test_clip_outliers(tmp_path, rho):
    import rfl_qc
    x = rho.copy()
    i = int(np.argmin(abs(WL - 1000)))
    x[2, i, 3] = -0.5     # failed inversion: nodata everywhere
    x[4, i, 5] = 3.0      # flare: kept, flagged
    p = str(write_envi(tmp_path / "r", x))
    low, high, mpath = rfl_qc.clip_outliers(p)
    assert (low, high) == (1, 1)
    m = envi.open(mpath + ".hdr", mpath).read_band(0)
    assert m[2, 3] == rfl_qc.OUT_LOW and m[4, 5] == rfl_qc.OUT_HIGH and m[0, 0] == rfl_qc.CLEAR
    y = envi.open(p + ".hdr", p).open_memmap(interleave="bil")
    assert np.all(y[2, :, 3] == rfl_qc.NODATA) and y[4, i, 5] == pytest.approx(3.0)


def test_snow_mask(tmp_path, rho):
    import rfl_qc
    x = rho.copy()
    snow = np.interp(WL, [400, 900, 1300, 1600, 2500], [0.95, 0.85, 0.5, 0.08, 0.05])
    x[:5, :, :5] = snow[None, :, None]
    p = str(write_envi(tmp_path / "r", x))
    path, frac = rfl_qc.snow_mask(p)
    m = envi.open(path + ".hdr", path).read_band(0)
    assert m[:5, :5].all() and m[20:, :].sum() == 0
    assert frac == pytest.approx(25 / m.size)


def test_scene_state(tmp_path, rho):
    import sixs_parts
    rfl, _ = _fake_isofit(tmp_path, rho)
    aot, h2o, elev = sixs_parts.scene_state(tmp_path / "isofit" / "output" / "scn_atm_interp")
    assert (aot, h2o, elev) == pytest.approx((0.2, 1.5, 0.3))


def test_atm_post_ghost_rule(tmp_path, rho):
    import atm_post
    x = rho.copy()
    x[1, int(np.argmin(abs(WL - 1000))), 1] = -1.0
    rfl, obs = _fake_isofit(tmp_path, x)
    rep_path = tmp_path / "rep.json"
    assert atm_post.main(["--sensor", "ghost", "--isofit-dir", str(tmp_path / "isofit"), "--obs", obs,
                          "--report", str(rep_path)]) == 0
    rep = json.loads(rep_path.read_text())
    assert rep["reflectance"] == rfl and rep["scene_aot550"] == pytest.approx(0.2)
    assert rep["outlier_pixels_nodata"] == 1 and not rep["adjacency_applied"] and not rep["terrain_applied"]
    assert rep["bands_outside_inversion_windows"] > 0 and rep["snow_fraction"] == 0


def _as_empirical_line(tmp_path, fid="scn", n_sp=6):
    """Turn a _fake_isofit run into what --empirical_line leaves: no *_atm_interp, a *_subs_state with one row per
    superpixel (named AOT550 / H2OSTR / surface_elevation_km bands last) and the *_lbl segmentation."""
    import rasterio
    out = tmp_path / "isofit" / "output"
    (out / f"{fid}_atm_interp").unlink()
    for f in out.glob(f"{fid}_atm_interp*"):
        f.unlink()
    with rasterio.open(out / f"{fid}_subs_state", "w", driver="ENVI", width=1, height=n_sp, count=4, dtype="float32") as d:
        for i, (n, v) in enumerate((("RFL_0500_0.000", 0.1), ("AOT550", 0.3), ("H2OSTR", 1.8), ("surface_elevation_km", 0.5))):
            d.write(np.full((n_sp, 1), v, np.float32) + np.arange(n_sp, dtype=np.float32)[:, None] * 0.01, i + 1)
            d.set_band_description(i + 1, n)
    rfl = rasterio.open(out / f"{fid}_rfl")
    L, S = rfl.height, rfl.width
    rfl.close()
    lbl = (np.arange(L)[:, None] * n_sp // L).repeat(S, 1).astype(np.float32)
    write_envi(out / f"{fid}_lbl", lbl[:, None, :], wl=np.arange(1.0), fwhm=np.ones(1))
    return out


def test_atm_post_on_empirical_line_output(tmp_path, rho):
    import atm_post
    _, obs = _fake_isofit(tmp_path, rho)
    _as_empirical_line(tmp_path)
    rfl, atm, _ = atm_post.find_outputs(tmp_path / "isofit")
    assert atm.name.endswith("_subs_state")
    rep = atm_post.run("avirisng", tmp_path / "isofit", obs)
    assert rep["scene_aot550"] == pytest.approx(0.325) and rep["scene_h2o"] == pytest.approx(1.825)
    assert rep["scene_elev_km"] == pytest.approx(0.525)


def test_report_atmosphere_from_superpixels(tmp_path, rho):
    import report
    _fake_isofit(tmp_path, rho)
    _as_empirical_line(tmp_path)
    m = report.read_atm(tmp_path, 2)
    assert set(m) == {"AOT550", "H2OSTR", "SURFACE_ELEVATION_KM"}
    assert m["AOT550"].shape == (30, 25)
    assert m["AOT550"][0, 0] == pytest.approx(0.3) and m["AOT550"][-1, 0] == pytest.approx(0.35)


def test_rules_table():
    import atm_post
    assert atm_post.rule_name("aviris_ng") == "avirisng" and atm_post.rule_name("unknown") == "generic"
    assert atm_post.SENSOR_RULES["prisma"].aot_prior and not atm_post.SENSOR_RULES["enmap"].aot_prior
    for r in atm_post.SENSOR_RULES.values():
        assert r.harmonise_model is None or r.harmonise_model.endswith(".npz")
    # superpixel size (AC26-SEG / FAST): AVIRIS-NG 400 for the 2 h budget, every other sensor at its validated 50
    seg = {k: r.segmentation_size for k, r in atm_post.SENSOR_RULES.items()}
    assert seg.pop("avirisng") == 400 and set(seg.values()) == {50}
    assert atm_post.asdict(atm_post.SENSOR_RULES["avirisng"])["segmentation_size"] == 400   # what scene.json carries
    # per-pixel stage (AC26-EL): AVIRIS-NG empirical line, every other sensor the analytical line
    line = {k: r.line_mode for k, r in atm_post.SENSOR_RULES.items()}
    assert line.pop("avirisng") == "empirical" and set(line.values()) == {"analytical"}


def test_steep_fraction(tmp_path, rho):
    import atm_post
    _, obs = _fake_isofit(tmp_path, rho, slope=10.0)
    assert atm_post.steep_fraction(obs, 6.0) == 1.0
    with pytest.raises(FileNotFoundError):
        atm_post.find_outputs(tmp_path / "nowhere")


def test_aot_prior(tmp_path):
    import aot_prior
    aod, lo, hi = aot_prior.month_window(21.7, 81.0, "20250601")
    assert 0 < aod < 3 and lo == round(max(aot_prior.FLOOR, aod - aot_prior.WINDOW), 2) and hi > lo
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"aerosol_2_range": [0.0, 1.0], "aerosol_2_spacing": 0.1}))
    out, rec = tmp_path / "lut.json", tmp_path / "rec.json"
    assert aot_prior.main(["--sensor", "prisma", "--lat", "21.7", "--lon", "81", "--date", "20250601",
                           "--base-lut", str(base), "--out", str(out), "--record", str(rec)]) == 0
    c, r = json.loads(out.read_text()), json.loads(rec.read_text())
    assert r["applied"] and c["aerosol_2_range"] == [lo, hi]
    assert aot_prior.main(["--sensor", "enmap", "--lat", "21.7", "--lon", "81", "--date", "20250601",
                           "--base-lut", str(base), "--out", str(out), "--record", str(rec)]) == 0
    assert json.loads(out.read_text()) == json.loads(base.read_text())
    assert not json.loads(rec.read_text())["applied"]
