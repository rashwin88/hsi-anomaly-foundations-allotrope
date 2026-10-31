"""Planet Tanager-1 HDF-EOS5 (basic = swath, ortho = UTM grid): radiance or surface reflectance, with per-pixel
sun / view angles, path length, time and location, so every geometry fact is metadata (grade A).

Radiance is W/(m^2 sr um) (x0.1 -> uW/cm^2/sr/nm); SR products are 0-1. Fill -9999 and nodata_pixels -> NaN.
Ortho grids carry no Lat/Lon datasets: their georeference is the StructMetadata UL / LR corners in the group's
EPSG (UTM), pixel-corner registered.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np

from .cube import CubeInfo, lonlat_reader
from .scene import Fact

VARS = (("toa_radiance", "radiance"), ("surface_reflectance", "surface_reflectance"))


def _arr(v):
    if isinstance(v, (bytes, str)):
        v = v.decode() if isinstance(v, bytes) else v
        return np.asarray([float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", v)])
    return np.asarray(v, float).ravel()


def open_tanager(path: str):
    """Returns (CubeInfo, dict of Facts for the Scene: level, location, sun, view, path_length, time)."""
    import h5py
    h = h5py.File(path, "r")
    kind = "SWATHS" if "HDFEOS/SWATHS/HYP" in h else "GRIDS"
    g = h[f"HDFEOS/{kind}/HYP"]
    df = g["Data Fields"]
    var, level = next(((v, lv) for v, lv in VARS if v in df), (None, None))
    if var is None:
        raise ValueError(f"{path}: no toa_radiance or surface_reflectance in {g.name}/Data Fields")
    d = df[var]
    B, L, S = d.shape
    wl, fwhm = _arr(d.attrs["wavelengths"]), _arr(d.attrs["fwhm"])
    nod = df["nodata_pixels"] if "nodata_pixels" in df else None
    fill = float(_arr(d.attrs.get("_FillValue", -9999.0))[0])

    def read_rows(r0, r1):
        x = np.array(d[:, r0:r1, :], np.float32).transpose(1, 0, 2)
        bad = ~np.isfinite(x) | (x == np.float32(fill))
        if nod is not None:
            bad |= (np.asarray(nod[r0:r1]) == 1)[:, None, :]
        x[bad] = np.nan
        return np.ascontiguousarray(x)

    def field(name):
        a = df[name]
        def f(r0, r1):
            v = np.array(a[r0:r1], np.float64)
            v[v == -9999.0] = np.nan
            return v
        return f

    facts = {}
    if kind == "SWATHS":
        geo = g["Geolocation Fields"]
        lat, lon = geo["Latitude"], geo["Longitude"]
        facts["location"] = Fact(lambda r0, r1, ns=None: (np.array(lon[r0:r1], float), np.array(lat[r0:r1], float)),
                                 "metadata", "HDF5 Latitude / Longitude per pixel")
        tt = np.asarray(geo["Time"][:], float)
        transform = crs = None
    else:
        sm = h["HDFEOS INFORMATION/StructMetadata.0"][()].decode()
        ul = [float(x) for x in re.search(r"UpperLeftPointMtrs=\(([^)]*)\)", sm).group(1).split(",")]
        lr = [float(x) for x in re.search(r"LowerRightMtrs=\(([^)]*)\)", sm).group(1).split(",")]
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
        transform = from_bounds(ul[0], lr[1], lr[0], ul[1], S, L)
        crs = CRS.from_epsg(int(_arr(g.attrs["epsg_code"])[0]))
        ll = lonlat_reader(transform, crs)
        facts["location"] = Fact(lambda r0, r1, ns=S: ll(r0, r1, ns), "metadata", f"ortho grid, EPSG {crs.to_epsg()}")
        tt = np.asarray(df["time"][L // 2, :], float)
    tt = tt[np.isfinite(tt) & (tt > 0)]
    t = datetime.fromtimestamp(float(np.median(tt)), tz=timezone.utc) if tt.size else None
    facts["time"] = Fact(t, "metadata" if t else "missing", "HDF5 per-line / per-pixel UTC time")
    sz, sa, vz, va = (field(n) for n in ("sun_zenith", "sun_azimuth", "sensor_zenith", "sensor_azimuth"))
    facts["sun"] = Fact(lambda r0, r1: (sz(r0, r1), sa(r0, r1)), "metadata", "HDF5 sun_zenith / sun_azimuth per pixel")
    facts["view"] = Fact(lambda r0, r1: (vz(r0, r1), va(r0, r1)), "metadata", "HDF5 sensor_zenith / sensor_azimuth")
    facts["path_length"] = Fact(field("sensor_to_ground_path_length"), "metadata", "HDF5 sensor_to_ground_path_length")
    facts["level"] = level
    u = d.attrs.get("Unit", "")
    facts["units"] = u.decode() if isinstance(u, bytes) else str(u)
    good = d.attrs.get("good_wavelengths")
    facts["good_bands"] = _arr(good).astype(bool) if good is not None and len(_arr(good)) == B else None
    info = CubeInfo(path, "tanager_h5", (L, B, S), "bsq", wl, fwhm, read_rows, "float32",
                    {"unit": facts["units"]}, [], transform, crs, [])
    return info, facts
