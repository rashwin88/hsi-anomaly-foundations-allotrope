"""
PACO-style post-hoc terrain correction of flat-surface ISOFIT reflectance (in place).

ISOFIT with the sRTMnet .h5 emulator ignores cos_i and sky view entirely: those
enter only through multipart transmittance, which needs a .6c or MODTRAN engine.
So the flat-terrain retrieval assumed irradiance Tdir + Tdif everywhere. This
module redistributes it as the EnMAP L2A ATBD does (EN-PCV-TN-6007 Iss. 2.5, 5.8.2):

1. Lambertian. A slope receives Tdir*cos(beta)/cos(sza) + Tdif*V plus light
   reflected by the surrounding terrain, rho_t*(1 - V)*(Tdir + Tdif), with sky view
   V = cos^2(slope/2) and rho_t a 0.5 km box mean. Then
   rho_L = rho_flat * (Tdir + Tdif) / that.
2. Modified Minnaert (Richter et al. 2009) where the local illumination angle beta
   exceeds beta_T = sza + 20 / 15 / 10 deg (sza < 45 / 45-55 / > 55):
   rho = rho_L * max((cos(beta)/cos(beta_T))^b, 0.25), with b = 1/2 off vegetation
   and 3/4 (< 720 nm) or 1/3 (>= 720 nm) on vegetation (NDVI > 0.3).

The Minnaert step only damps over-corrected values; it never lifts negatives.
A ray-traced (horizon) sky view instead of cos^2(slope/2) made agreement with PACO
worse on the Himalaya scene (AC26-M2), so it is not used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from spectral import envi

import sixs_parts
from rfl_qc import NODATA, _open

# ISOFIT obs cube bands (0-based).
OBS_SZA, OBS_SLOPE, OBS_COSI = 4, 6, 8


def minnaert_threshold_deg(sza_deg: float) -> float:
    """beta_T: the illumination angle beyond which the Minnaert damping applies."""
    return sza_deg + (20.0 if sza_deg < 45 else 15.0 if sza_deg < 55 else 10.0)


def minnaert_exponent(wl_nm: float, veg: np.ndarray) -> np.ndarray:
    return np.where(veg, 0.75 if wl_nm < 720 else 1.0 / 3.0, 0.5)


def correct(rfl: str, obs: str, lut_dir: Path, atm_interp: Path, px_m: float = 30.0) -> dict:
    """Apply the terrain correction to ENVI cube `rfl` in place, using slope and cos_i from `obs`."""
    m, wl = _open(rfl)
    o = envi.open(obs + ".hdr", obs).open_memmap(interleave="bip")
    sza, slope = np.asarray(o[..., OBS_SZA], float), np.asarray(o[..., OBS_SLOPE], float)
    cb = np.asarray(o[..., OBS_COSI], float)
    cs = np.cos(np.radians(sza))
    V = np.cos(np.radians(slope) / 2) ** 2
    aot, h2o, elev = sixs_parts.scene_state(atm_interp)
    t = sixs_parts.components(lut_dir, aot, h2o, elev, wl)
    tdir, tdif = t["down_dir"], t["down_dif"]
    sz = float(np.nanmedian(sza))
    bt = np.radians(minnaert_threshold_deg(sz))
    poor = np.arccos(np.clip(cb, -1, 1)) > bt
    r660, r860 = (np.asarray(m[:, int(np.argmin(np.abs(wl - nm))), :], float) for nm in (660, 860))
    veg = (r860 - r660) / (r860 + r660 + 1e-6) > 0.3
    box = int(round(500 / px_m))
    for i in range(m.shape[1]):
        a = np.asarray(m[:, i, :], float)
        good = a > NODATA + 9
        if not good.any():
            continue
        rt = uniform_filter(np.where(good, np.clip(a, 0, 1), 0), box) / \
            np.maximum(uniform_filter(good.astype(float), box), 1e-3)
        e_flat = tdir[i] + tdif[i]
        lam = e_flat / (tdir[i] * cb / cs + tdif[i] * V + e_flat * rt * (1 - V) + 1e-6)
        r = a * lam
        g = np.maximum((np.clip(cb, 1e-3, 1) / np.cos(bt)) ** minnaert_exponent(wl[i], veg), 0.25)
        r = np.where(poor, r * g, r)
        m[:, i, :] = np.where(good, r, NODATA)
    m.flush()
    return {"sza_deg": sz, "beta_t_deg": float(np.degrees(bt)), "poorly_lit_frac": float(poor.mean())}
