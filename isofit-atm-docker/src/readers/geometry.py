"""Geometry for the generic path: sun position, solar irradiance, DEM, slope, and ISOFIT's loc / obs cubes.

Sun position: NOAA's general solar position formulas (the ones enmap_l1c_preprocess uses), vectorised, about
0.1 deg. That is a standard formula, so sun angles from a known time and location are "derived", not assumed.
View geometry has no formula: without delivered view angles the view is assumed nadir.
"""
from __future__ import annotations

import glob
import math
from datetime import datetime

import numpy as np

ASSUMED_SZA, ASSUMED_SAA = 35.0, 180.0   # used only when the acquisition time or the location is unknown
ISOFIT_IRR = "/root/.isofit/examples/20151026_SantaMonica/data/prism_optimized_irr.dat"


def sun_position(t: datetime, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(zenith, azimuth clockwise from north), degrees, per pixel."""
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = np.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    cz = np.clip(np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha), -1, 1)
    z = np.degrees(np.arccos(cz))
    ca = np.clip((math.sin(decl) - np.sin(la) * cz) / (np.cos(la) * np.sin(np.radians(z)) + 1e-12), -1, 1)
    az = np.degrees(np.arccos(ca))
    return z, np.where(ha > 0, 360 - az, az)


def earth_sun_au(t: datetime | None) -> float:
    if t is None:
        return 1.0
    doy = t.timetuple().tm_yday
    return 1 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4)))


def solar_irradiance(wl: np.ndarray, fwhm: np.ndarray) -> np.ndarray:
    """Band-convolved top-of-atmosphere solar irradiance at 1 AU, uW/cm^2/nm: ISOFIT's own sRTMnet irradiance
    file, so a TOA-reflectance input converted with it returns to the same reflectance inside ISOFIT."""
    f = ISOFIT_IRR if glob.glob(ISOFIT_IRR) else (glob.glob("/root/.isofit/examples/**/prism_optimized_irr.dat",
                                                            recursive=True) or [None])[0]
    if f is None:
        raise FileNotFoundError("ISOFIT's prism_optimized_irr.dat not found (isofit download examples)")
    q = np.loadtxt(f)
    q = q[np.argsort(q[:, 0])]
    x, e = q[:, 0], q[:, 1] / 10.0   # file is mW/m^2/nm ... x0.1 -> uW/cm^2/nm (checked: ~95 at 860 nm)
    out = np.empty(len(wl))
    for i, (c, w) in enumerate(zip(wl, fwhm)):
        s = max(w, 0.5) / 2.3548
        m = (x > c - 4 * s) & (x < c + 4 * s)
        k = np.exp(-0.5 * ((x[m] - c) / s) ** 2)
        out[i] = (k * e[m]).sum() / k.sum() if m.any() else np.interp(c, x, e)
    return out


def sample_dem(dem: str | None, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, str]:
    """Elevation in metres per pixel, and a note. Falls back to 0 m (noted) without a DEM or outside it; ISOFIT's
    pressure-elevation retrieval then carries the surface pressure."""
    if not dem or not glob.glob(dem):
        return np.zeros(lon.shape, np.float32), "no DEM given or found: elevation 0 m assumed"
    import rasterio
    from pyproj import Transformer
    with rasterio.open(dem) as d:
        tr = Transformer.from_crs(4326, d.crs, always_xy=True) if d.crs and d.crs.to_epsg() != 4326 else None
        xs, ys = (tr.transform(lon.ravel(), lat.ravel()) if tr else (lon.ravel(), lat.ravel()))
        nod = d.nodata
        z = np.fromiter((v[0] for v in d.sample(zip(xs, ys))), np.float32, count=xs.size).reshape(lon.shape)
    bad = ~np.isfinite(z) | (z < -1000) | ((z == nod) if nod is not None else False)
    if bad.all():
        return np.zeros(lon.shape, np.float32), "scene outside the DEM: elevation 0 m assumed"
    z[bad] = 0.0
    return z, (f"DEM {dem}" + (f"; {bad.mean():.1%} of pixels outside it set to 0 m" if bad.any() else ""))


def slope_aspect(lat, lon, z, sza, saa):
    """Slope, aspect (degrees, aspect = downslope direction clockwise from north) and cos(i), on any geolocated
    grid (swath or map): gradients in local east / north metres from neighbour offsets (prisma_l1_preprocess)."""
    R = 6371000.0
    g = lambda a, ax: np.gradient(a, axis=ax)
    E = lambda dlon, la: np.radians(dlon) * R * np.cos(np.radians(la))
    N = lambda dlat: np.radians(dlat) * R
    e_c, n_c, z_c = E(g(lon, 1), lat), N(g(lat, 1)), g(z, 1)
    e_r, n_r, z_r = E(g(lon, 0), lat), N(g(lat, 0)), g(z, 0)
    det = e_c * n_r - e_r * n_c
    det = np.where(np.abs(det) < 1e-6, 1e-6, det)
    gx, gy = (z_c * n_r - z_r * n_c) / det, (e_c * z_r - e_r * z_c) / det
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy) % (2 * np.pi)
    s, a = np.radians(sza), np.radians(saa)
    cos_i = np.cos(s) * np.cos(slope) + np.sin(s) * np.sin(slope) * np.cos(a - aspect)
    return np.degrees(slope), np.degrees(aspect), np.clip(cos_i, 0.05, 1.0)


def obs_bands(path_m, vaa, vza, saa, sza, slope, aspect, cos_i, utc_hour) -> list[np.ndarray]:
    """ISOFIT's 10-band observation cube, in its order (as enmap_l1c_preprocess writes it)."""
    r = np.radians
    cphase = np.cos(r(sza)) * np.cos(r(vza)) + np.sin(r(sza)) * np.sin(r(vza)) * np.cos(r(saa - vaa))
    full = lambda v: np.broadcast_to(np.asarray(v, np.float32), path_m.shape)
    return [path_m, full(vaa), full(vza), full(saa), full(sza), np.degrees(np.arccos(np.clip(cphase, -1, 1))),
            slope, aspect, cos_i, full(utc_hour)]


OBS_NAMES = ["path_length_m", "to_sensor_azimuth", "to_sensor_zenith", "to_sun_azimuth", "to_sun_zenith", "phase",
             "slope", "aspect", "cosine_i", "utc_hour"]
