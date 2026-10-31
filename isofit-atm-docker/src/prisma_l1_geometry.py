"""
Per-pixel view/sun geometry for a PRISMA L1 product, written as a stand-in L2C file.

ACOLITE refuses a PRS_L1_STD_OFFL product unless ASI's matching L2C sits next
to it, and reads only three arrays from that L2C:
HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/{Observing_Angle,
Rel_Azimuth_Angle, Solar_Zenith_Angle}. At deployment only L1 arrives, so this
script derives those arrays from the L1 alone and ACOLITE is pointed at the
result with `l2cfile=`.

  * View angles: satellite ECEF position (Info/Ancillary/PVSdata/Wgs84_pos_*)
    against each pixel's Latitude_SWIR/Longitude_SWIR on the ellipsoid.
    The GPS fix refreshes about once a second, so the position is a step
    function of frame number; each fix is placed at the frame where it first
    appears and a quadratic in frame number is fitted through those points.
    The first fix is dropped - its first appearance predates the product.
  * Sun angles: the scene values in the L1 root attributes
    (Sun_zenith_angle, Sun_azimuth_angle). Across a 30 km scene the zenith
    moves ~0.3 deg, a 0.1% change in cos(sza).
  * Relative azimuth: |sun azimuth - view azimuth| folded into 0..180, the
    same formula ACOLITE applies to L1G products (view azimuth = direction
    from the pixel to the satellite).

Ground height is taken as 0: 1 km of terrain moves the view zenith by ~0.1 deg.

  python prisma_l1_geometry.py --l1 <PRS_L1_STD_OFFL_*.he5> --out <dir>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

log = logging.getLogger("prisma_l1_geometry")

HCO = "HDFEOS/SWATHS/PRS_L1_HCO"
OUT_GROUP = "HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields"


def satellite_track(f: h5py.File) -> np.ndarray:
    """ECEF satellite position (lines, 3) in metres for each HCO image line."""
    pvs = f["Info/Ancillary/PVSdata"]
    t = pvs["GPS_Time_of_Last_Position"][:]
    pos = np.c_[pvs["Wgs84_pos_x"][:], pvs["Wgs84_pos_y"][:], pvs["Wgs84_pos_z"][:]].astype(float)
    frames = f["Info/Header/FrameNumber"][:].astype(float)
    _, first = np.unique(t, return_index=True)
    first = np.sort(first)[1:]
    if len(first) < 2:
        raise ValueError(f"only {len(first) + 1} GPS fixes in the product, need 3")
    deg = 2 if len(first) >= 3 else 1
    line_frames = f[f"{HCO}/Data Fields/FrameNumber"][:].astype(float)
    return np.stack([np.polyval(np.polyfit(frames[first], pos[first, k], deg), line_frames)
                     for k in range(3)], axis=1)


def view_angles(lat: np.ndarray, lon: np.ndarray, sat: np.ndarray):
    """View zenith and azimuth (deg) from each pixel to the satellite of its line."""
    to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)
    gx, gy, gz = to_ecef.transform(lon, lat, np.zeros_like(lat))
    d = np.stack([sat[:, None, 0] - gx, sat[:, None, 1] - gy, sat[:, None, 2] - gz])
    d /= np.linalg.norm(d, axis=0)
    la, lo = np.radians(lat), np.radians(lon)
    east = -np.sin(lo) * d[0] + np.cos(lo) * d[1]
    north = -np.sin(la) * np.cos(lo) * d[0] - np.sin(la) * np.sin(lo) * d[1] + np.cos(la) * d[2]
    up = np.cos(la) * np.cos(lo) * d[0] + np.cos(la) * np.sin(lo) * d[1] + np.sin(la) * d[2]
    vza = np.degrees(np.arccos(np.clip(up, -1, 1)))
    vaa = np.degrees(np.arctan2(east, north)) % 360
    return vza, vaa


def build(l1: Path, out_dir: Path) -> Path:
    with h5py.File(l1, "r") as f:
        lat = f[f"{HCO}/Geolocation Fields/Latitude_SWIR"][:].astype(float)
        lon = f[f"{HCO}/Geolocation Fields/Longitude_SWIR"][:].astype(float)
        sza0 = float(f.attrs["Sun_zenith_angle"])
        saa0 = float(f.attrs["Sun_azimuth_angle"])
        sat = satellite_track(f)
    # 65535 marks missing geolocation, as ACOLITE's reader assumes.
    lat[lat >= 65535] = np.nan
    lon[lon >= 65535] = np.nan
    vza, vaa = view_angles(lat, lon, sat)
    raa = np.abs(saa0 - vaa)
    raa = np.where(raa > 180, 360 - raa, raa)
    sza = np.full(lat.shape, sza0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / l1.name.replace("PRS_L1_STD_OFFL_", "PRS_L2C_STD_")
    with h5py.File(out, "w") as g:
        grp = g.create_group(OUT_GROUP)
        grp["Observing_Angle"] = vza.astype("float32")
        grp["Rel_Azimuth_Angle"] = raa.astype("float32")
        grp["Solar_Zenith_Angle"] = sza.astype("float32")
        g.attrs["Source"] = f"geometry derived from {l1.name} by prisma_l1_geometry.py"
    log.info("%s: VZA %.2f..%.2f (mean %.2f), VAA mean %.1f, SZA %.2f, SAA %.2f, RAA mean %.1f",
             l1.name, np.nanmin(vza), np.nanmax(vza), np.nanmean(vza), np.nanmean(vaa),
             sza0, saa0, np.nanmean(raa))
    log.info("Wrote %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l1", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    build(args.l1, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
