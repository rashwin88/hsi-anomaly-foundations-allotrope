"""
PRISMA L1 (PRS_L1_STD_OFFL_*.he5) -> ISOFIT input cubes, in L1 swath geometry.

  * <fid>_rdn : TOA radiance, uW/cm^2/sr/nm, float32 BIL. L1 radiance is
                DN / ScaleFactor - Offset in W/m^2/sr/um; x0.1 converts.
                VNIR + SWIR merged ascending; VNIR kept below 950 nm, SWIR
                from 950 nm, so the 920-980 nm overlap is not doubled.
  * <fid>_loc : lon, lat, elevation (m, Cop-DEM), float32 BIP
  * <fid>_obs : ISOFIT's 10-band observation cube, float32 BIP. View angles
                per pixel from the satellite track (prisma_l1_geometry), sun
                angles from the scene attributes, slope/aspect/cos_i from the
                DEM in swath geometry.

fid is PRS_<start yyyymmddHHMMSS>, which apply_oe's prisma branch parses.
Wavelengths are the scene List_Cw_* values; per-column smile is ignored.

  python prisma_l1_preprocess.py --l1 <he5> --dem <vrt> --workdir <dir>
"""
import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio
from pyproj import Transformer

from prisma_l1_geometry import satellite_track, view_angles

log = logging.getLogger("prisma_l1_preprocess")
HCO = "HDFEOS/SWATHS/PRS_L1_HCO"
ORBIT_KM = 615.0
SPLIT_NM = 950.0


def hdr(path, cols, rows, nb, inter, wl=None, fwhm=None, names=None, desc=""):
    L = ["ENVI", f"description = {{{desc}}}", f"samples = {cols}", f"lines = {rows}",
         f"bands = {nb}", "header offset = 0", "file type = ENVI Standard", "data type = 4",
         f"interleave = {inter}", "byte order = 0"]
    if wl is not None:
        L += ["wavelength units = Nanometers",
              "wavelength = {" + ", ".join(f"{w:.4f}" for w in wl) + "}",
              "fwhm = {" + ", ".join(f"{f:.4f}" for f in fwhm) + "}"]
    if names:
        L.append("band names = {" + ", ".join(names) + "}")
    Path(str(path) + ".hdr").write_text("\n".join(L) + "\n")


def radiance(f):
    a = f.attrs
    parts = []
    for fam, lo, hi in (("Vnir", 0, SPLIT_NM), ("Swir", SPLIT_NM, 3000)):
        wl = np.asarray(a[f"List_Cw_{fam}"], float)
        fw = np.asarray(a[f"List_Fwhm_{fam}"], float)
        keep = np.asarray(a[f"List_Cw_{fam}_Flags"]).astype(bool) & (wl > lo) & (wl < hi)
        sf, off = float(a[f"ScaleFactor_{fam}"]), float(a[f"Offset_{fam}"])
        dn = f[f"{HCO}/Data Fields/{fam.upper()}_Cube"][:, keep, :].astype(np.float32)
        rad = (dn / sf - off) * 0.1
        rad[dn == 0] = np.nan
        parts.append((wl[keep], fw[keep], rad))
    wl = np.concatenate([p[0] for p in parts])
    fw = np.concatenate([p[1] for p in parts])
    cube = np.concatenate([p[2] for p in parts], axis=1)  # (rows, bands, cols)
    # Bands that are DN 0 on >5% of pixels are dead/edge bands, not nodata.
    alive = np.isfinite(cube).mean(axis=(0, 2)) > 0.95
    log.info("dropping %d mostly-zero bands: %s", (~alive).sum(), np.round(wl[~alive], 1))
    wl, fw, cube = wl[alive], fw[alive], cube[:, alive, :]
    o = np.argsort(wl)
    return wl[o], fw[o], cube[:, o, :]


def slope_aspect(lat, lon, z, sza, saa):
    """Gradient of z in local east/north metres from swath-neighbour offsets."""
    R = 6371000.0
    def d(a, axis):
        return np.gradient(a, axis=axis)
    E = lambda dlon, la: np.radians(dlon) * R * np.cos(np.radians(la))
    N = lambda dlat: np.radians(dlat) * R
    e_c, n_c, z_c = E(d(lon, 1), lat), N(d(lat, 1)), d(z, 1)
    e_r, n_r, z_r = E(d(lon, 0), lat), N(d(lat, 0)), d(z, 0)
    det = e_c * n_r - e_r * n_c
    det = np.where(np.abs(det) < 1e-6, 1e-6, det)
    gx = (z_c * n_r - z_r * n_c) / det   # dz/dEast
    gy = (e_c * z_r - e_r * z_c) / det   # dz/dNorth
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy) % (2 * np.pi)  # downslope direction, clockwise from N
    s, a = math.radians(sza), math.radians(saa)
    cos_i = np.cos(s) * np.cos(slope) + np.sin(s) * np.sin(slope) * np.cos(a - aspect)
    return np.degrees(slope), np.degrees(aspect), np.clip(cos_i, 0.05, 1.0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--l1", required=True, type=Path)
    p.add_argument("--dem", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)
    a = p.parse_args(argv)
    w = a.workdir
    w.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.l1, "r") as f:
        start = f.attrs["Product_StartTime"].decode()
        t = datetime.fromisoformat(start)
        fid = "PRS_" + t.strftime("%Y%m%d%H%M%S")
        sza, saa = float(f.attrs["Sun_zenith_angle"]), float(f.attrs["Sun_azimuth_angle"])
        lat = f[f"{HCO}/Geolocation Fields/Latitude_SWIR"][:].astype(float)
        lon = f[f"{HCO}/Geolocation Fields/Longitude_SWIR"][:].astype(float)
        sat = satellite_track(f)
        wl, fw, cube = radiance(f)
    rows, nb, cols = cube.shape
    bad = ~np.isfinite(cube).all(axis=1) | (lat >= 65535)
    cube = np.where(np.isfinite(cube), np.clip(cube, 1e-4, None), -9999.0).astype(np.float32)
    cube[:, :, :][np.broadcast_to(bad[:, None, :], cube.shape)] = -9999.0
    hdr(w / f"{fid}_rdn", cols, rows, nb, "bil", wl, fw, desc=f"PRISMA L1 radiance {a.l1.name}")
    cube.tofile(w / f"{fid}_rdn")
    log.info("radiance %s: %d bands %.1f-%.1f nm, %d bad pixels", cube.shape, nb, wl[0], wl[-1], bad.sum())

    lat[bad], lon[bad] = np.nanmedian(lat), np.nanmedian(lon)
    with rasterio.open(a.dem) as dem:
        tr = Transformer.from_crs(4326, dem.crs, always_xy=True)
        xs, ys = tr.transform(lon.ravel(), lat.ravel())
        z = np.fromiter((v[0] for v in dem.sample(zip(xs, ys))), np.float32, count=xs.size).reshape(lat.shape)
    z = np.where(np.isfinite(z) & (z > -1000), z, 0.0)
    hdr(w / f"{fid}_loc", cols, rows, 3, "bip", names=["longitude", "latitude", "elevation"])
    np.stack([lon, lat, z], -1).astype(np.float32).tofile(w / f"{fid}_loc")

    vza, vaa = view_angles(lat, lon, sat)
    cphase = (np.cos(np.radians(sza)) * np.cos(np.radians(vza))
              + np.sin(np.radians(sza)) * np.sin(np.radians(vza)) * np.cos(np.radians(saa - vaa)))
    path_m = (ORBIT_KM * 1000.0 - z) / np.maximum(np.cos(np.radians(vza)), 0.1)
    slope, aspect, cos_i = slope_aspect(lat, lon, z, sza, saa)
    hour = t.hour + t.minute / 60 + t.second / 3600
    full = lambda v: np.full(lat.shape, v, np.float32)
    obs = [path_m, vaa, vza, full(saa), full(sza), np.degrees(np.arccos(np.clip(cphase, -1, 1))),
           slope, aspect, cos_i, full(hour)]
    hdr(w / f"{fid}_obs", cols, rows, 10, "bip",
        names=["path_length_m", "to_sensor_azimuth", "to_sensor_zenith", "to_sun_azimuth",
               "to_sun_zenith", "phase", "slope", "aspect", "cosine_i", "utc_hour"])
    np.stack(obs, -1).astype(np.float32).tofile(w / f"{fid}_obs")
    np.savetxt(w / "wavelengths.txt", np.c_[np.arange(1, nb + 1), wl, fw], fmt="%d\t%.4f\t%.4f")
    (w / "season.txt").write_text("summer" if 4 <= t.month <= 9 else "winter")
    (w / "fid.txt").write_text(fid)
    log.info("fid %s SZA %.2f SAA %.2f VZA %.2f..%.2f elev %.0f..%.0f m", fid, sza, saa,
             np.nanmin(vza), np.nanmax(vza), z.min(), z.max())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
