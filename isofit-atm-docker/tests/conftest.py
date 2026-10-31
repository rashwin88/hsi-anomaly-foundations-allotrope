"""Shared fixtures: synthetic hyperspectral scenes with the physics the module relies on.

Surfaces are smooth mixtures of soil, vegetation and water endmembers, so the spectra look real to the interleave
check (coherence and smoothness along bands, samples and lines). The atmosphere is a transmittance with the gas
bands level.py measures (O2 760, H2O 940 / 1135 / 1380 / 1870, CO2 2010) plus a blue path term, and radiance is
that TOA reflectance times a 5778 K solar spectrum. No ISOFIT, no real data, no network.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

WL = np.arange(400.0, 2501.0, 10.0)        # 211 bands, 10 nm, like a generic imaging spectrometer
FWHM = np.full(WL.shape, 10.0)
WHEN = datetime(2025, 6, 1, 6, 0, tzinfo=timezone.utc)
UTM_ORIGIN = (500000.0, 2400000.0)         # EPSG:32644, ~21.7 N 81 E


def endmembers(wl=WL) -> dict[str, np.ndarray]:
    soil = np.interp(wl, [400, 600, 1000, 1600, 2200, 2500], [0.08, 0.16, 0.26, 0.33, 0.30, 0.26])
    veg = np.interp(wl, [400, 500, 550, 600, 680, 720, 760, 1300, 1450, 1650, 1900, 2200, 2500],
                    [0.03, 0.04, 0.10, 0.06, 0.03, 0.20, 0.45, 0.42, 0.25, 0.30, 0.12, 0.17, 0.08])
    water = np.interp(wl, [400, 500, 600, 700, 800, 2500], [0.05, 0.05, 0.03, 0.01, 0.003, 0.0])
    return {"soil": soil, "veg": veg, "water": water}


def surface(lines=60, samples=50, wl=WL) -> np.ndarray:
    """(lines, bands, samples) surface reflectance with spatially smooth abundances."""
    e = endmembers(wl)
    r, c = np.meshgrid(np.arange(lines), np.arange(samples), indexing="ij")
    a_veg = 0.5 + 0.45 * np.sin(c / 9.0) * np.cos(r / 11.0)
    a_wat = np.clip((c - samples * 0.75) / (samples * 0.25), 0, 1) * 0.9
    a_soil = np.clip(1 - a_veg - a_wat, 0, 1)
    s = a_veg + a_wat + a_soil
    cube = (a_soil[..., None] * e["soil"] + a_veg[..., None] * e["veg"] + a_wat[..., None] * e["water"]) / s[..., None]
    return cube.transpose(0, 2, 1).astype(np.float32)


def transmittance(wl=WL) -> np.ndarray:
    t = np.ones_like(wl)
    for c, depth, w in ((761, 0.5, 6), (940, 0.75, 25), (1135, 0.7, 25), (1380, 0.995, 45), (1870, 0.995, 70),
                        (2010, 0.8, 18)):
        t *= 1 - depth * np.exp(-0.5 * ((wl - c) / w) ** 2)
    return t


def toa_reflectance(rho: np.ndarray, wl=WL) -> np.ndarray:
    path = 0.09 * (450.0 / wl) ** 4
    return (rho * transmittance(wl)[None, :, None] + path[None, :, None]).astype(np.float32)


def solar(wl=WL) -> np.ndarray:
    """TOA solar irradiance, uW/cm^2/nm (~190 at 500 nm)."""
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    lam = wl * 1e-9
    b = 2 * h * c ** 2 / lam ** 5 / (np.exp(h * c / (lam * k * 5778.0)) - 1)
    return 190.0 * b / np.interp(500e-9, lam, b)


def radiance(rho: np.ndarray, wl=WL, sza=30.0) -> np.ndarray:
    """uW/cm^2/sr/nm."""
    return (toa_reflectance(rho, wl) * solar(wl)[None, :, None] * np.cos(np.radians(sza)) / np.pi).astype(np.float32)


def write_tif(path: Path, cube_lbs: np.ndarray, wl=WL, fwhm=FWHM, crs="EPSG:32644", tags=None, dtype=None,
              nodata=None) -> Path:
    import rasterio
    from rasterio.transform import from_origin
    L, B, S = cube_lbs.shape
    kw = dict(driver="GTiff", width=S, height=L, count=B, dtype=dtype or cube_lbs.dtype, nodata=nodata)
    if crs:
        kw.update(crs=crs, transform=from_origin(*UTM_ORIGIN, 30.0, 30.0))
    with rasterio.open(path, "w", **kw) as d:
        d.write(cube_lbs.transpose(1, 0, 2).astype(kw["dtype"]))
        for i in range(B):
            d.update_tags(i + 1, wavelength=f"{wl[i]:.2f}", fwhm=f"{fwhm[i]:.2f}")
        d.update_tags(wavelength_units="nm", **(tags or {}))
    return path


def write_envi(stem: Path, cube_lbs: np.ndarray, wl=WL, fwhm=FWHM, interleave="bil", extra=None, dtype="<f4") -> Path:
    """ENVI cube at <stem> (no extension) + <stem>.hdr, data in the given interleave."""
    L, B, S = cube_lbs.shape
    arr = {"bil": cube_lbs, "bsq": cube_lbs.transpose(1, 0, 2), "bip": cube_lbs.transpose(0, 2, 1)}[interleave]
    np.ascontiguousarray(arr, dtype).tofile(stem)
    code = {"<f4": 4, "<i2": 2, "<u2": 12}[dtype]
    h = ["ENVI", f"samples = {S}", f"lines = {L}", f"bands = {B}", "header offset = 0", "file type = ENVI Standard",
         f"data type = {code}", f"interleave = {interleave}", "byte order = 0", "data ignore value = -9999",
         "wavelength units = Nanometers", "wavelength = {" + ", ".join(f"{v:.2f}" for v in wl) + "}",
         "fwhm = {" + ", ".join(f"{v:.2f}" for v in fwhm) + "}"] + list(extra or [])
    Path(str(stem) + ".hdr").write_text("\n".join(h) + "\n")
    return stem


def write_irradiance(path: Path) -> Path:
    """ISOFIT's prism_optimized_irr.dat layout: wavelength nm, irradiance mW/m^2/nm (geometry divides by 10)."""
    x = np.arange(350.0, 2600.0, 1.0)
    np.savetxt(path, np.c_[x, solar(x) * 10.0])
    return path


def write_tanager(path: Path, rho: np.ndarray, kind="toa_radiance") -> Path:
    """A Planet Tanager basic (swath) HDF-EOS5 file: radiance in W/m^2/sr/um or SR 0-1, per-pixel geometry."""
    import h5py
    L, B, S = rho.shape
    with h5py.File(path, "w") as f:
        g = f.create_group("HDFEOS/SWATHS/HYP")
        df, geo = g.create_group("Data Fields"), g.create_group("Geolocation Fields")
        x = radiance(rho) * 10.0 if kind == "toa_radiance" else rho
        d = df.create_dataset(kind, data=x.transpose(1, 0, 2))
        d.attrs["wavelengths"], d.attrs["fwhm"], d.attrs["_FillValue"] = WL, FWHM, np.array([-9999.0])
        d.attrs["Unit"] = b"W/(m^2 sr um)" if kind == "toa_radiance" else b"1"
        r, c = np.meshgrid(np.arange(L), np.arange(S), indexing="ij")
        geo["Latitude"], geo["Longitude"] = 40.6 - r * 0.00027, 22.9 + c * 0.00035
        geo["Time"] = np.full(L, WHEN.timestamp())
        for n, v in (("sun_zenith", 30.0), ("sun_azimuth", 140.0), ("sensor_zenith", 5.0), ("sensor_azimuth", 100.0),
                     ("sensor_to_ground_path_length", 522000.0)):
            df[n] = np.full((L, S), v)
    return path


@pytest.fixture
def irradiance(tmp_path, monkeypatch):
    """Point readers.geometry at a synthetic ISOFIT irradiance file."""
    from readers import geometry
    p = write_irradiance(tmp_path / "prism_optimized_irr.dat")
    monkeypatch.setattr(geometry, "ISOFIT_IRR", str(p))
    return p


@pytest.fixture
def rho():
    return surface()


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())
