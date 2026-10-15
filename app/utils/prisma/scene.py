"""
One PRISMA L2D scene read whole, at its native bands, for the training shards.

Used by app/utils/prisma/tiles.py and scripts/shard_prisma.py. The product reads PRISMA through
PrismaDatasetBuilder, which runs the band-filter pipeline (exclusions, gap interpolation, the
165-band grid); shards must hold the sensor's own bands untouched, so this reads the HE5 directly
and applies only what that builder applies before its pipeline - the same DN-to-reflectance scale
and the same validity rules.

    reflectance = DN / 65535 x (L2Scale<Det>Max - L2Scale<Det>Min) + L2Scale<Det>Min
    nodata      = DN == 0, or the detector's PIXEL_L2_ERR_MATRIX != 0 (values 1, 2 seen)
    dropped     = bands whose List_Cw_<Det>_Flags is 0 (5 of 239: fwhm 0, no real band)

Gotchas:
  - The cubes are BIL, (rows, bands, cols); everything returned here is (bands, rows, cols).
  - Bands keep the file's order: VNIR (listed from ~980 nm down) then SWIR. VNIR and SWIR overlap
    around 920-980 nm; nothing is sorted or de-duplicated - wavelengths travel with the pixels.
  - Band centres are fixed: 234 identical centres in every scene checked, 2020-2026 (2026-10-15).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NODATA = -32768
REFLECTANCE_SCALE = 0.0001
SWATH = "HDFEOS/SWATHS/PRS_L2D_HCO"
DETECTORS = ("Vnir", "Swir")


@dataclass
class PrismaScene:
    scene_id: str
    pixels: np.ndarray          # (C, H, W) int16, reflectance x 10,000, NODATA where invalid
    wavelengths_nm: np.ndarray  # (C,) float32, file order
    lat: np.ndarray             # (H, W) float32
    lon: np.ndarray             # (H, W) float32
    facts: dict                 # scene_id, date, sun angles, epsg, scale factors

    @property
    def centre(self) -> tuple[float, float]:
        """(lat, lon) mean over valid pixels."""
        v = (self.pixels != NODATA).any(axis=0)
        return float(self.lat[v].mean()), float(self.lon[v].mean())


def _to_int16(dn: np.ndarray, lo: float, hi: float) -> np.ndarray:
    refl = dn.astype(np.float32) / 65535.0 * (hi - lo) + lo
    return np.clip(np.rint(refl / REFLECTANCE_SCALE), -32767, 32767).astype(np.int16)


def read_scene(path: str | Path) -> PrismaScene:
    """Read a PRISMA L2D .he5 into native-band int16. Raises ValueError if a cube's band count
    differs from its wavelength list."""
    import h5py
    path = Path(path)
    cubes, wls = [], []
    with h5py.File(path, "r") as f:
        a, g = f.attrs, f[SWATH]
        for det in DETECTORS:
            cw, flags = np.asarray(a[f"List_Cw_{det}"], np.float64), np.asarray(a[f"List_Cw_{det}_Flags"])
            dn = g[f"Data Fields/{det.upper()}_Cube"][()]                  # (H, C, W) uint16
            if dn.shape[1] != len(cw):
                raise ValueError(f"{path.name}: {det} cube has {dn.shape[1]} bands, {len(cw)} wavelengths")
            keep = flags > 0
            err = g[f"Data Fields/{det.upper()}_PIXEL_L2_ERR_MATRIX"][:, keep, :]
            dn = dn[:, keep, :]
            px = _to_int16(dn, float(a[f"L2Scale{det}Min"]), float(a[f"L2Scale{det}Max"]))
            px[(dn == 0) | (err != 0)] = NODATA
            cubes.append(px.transpose(1, 0, 2))                           # (C, H, W)
            wls.append(cw[keep])
        lat = g["Geolocation Fields/Latitude"][()].astype(np.float32)
        lon = g["Geolocation Fields/Longitude"][()].astype(np.float32)
        facts = {"scene_id": path.stem, "date": path.stem.split("_")[3][:8],
                 "sun_zenith": float(a["Sun_zenith_angle"]), "sun_azimuth": float(a["Sun_azimuth_angle"]),
                 "epsg": int(a["Epsg_Code"]),
                 "l2_scale": {det: [float(a[f"L2Scale{det}Min"]), float(a[f"L2Scale{det}Max"])] for det in DETECTORS}}
    return PrismaScene(path.stem, np.ascontiguousarray(np.concatenate(cubes)),
                       np.concatenate(wls).astype(np.float32), lat, lon, facts)
