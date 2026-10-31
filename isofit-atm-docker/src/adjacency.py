"""
Post-hoc adjacency correction of ISOFIT reflectance (in place).

ISOFIT with the sRTMnet .h5 emulator assumes a uniform surface, so the light
scattered into the view path from the neighbourhood is credited to the pixel:
    rho_ret * T_up  ~  rho * t_up_dir + rho_bg * t_up_dif
Solving for rho gives the first-order form used by Tanre et al. 1981, ATCOR and PACO:
    rho = rho_ret + q * (rho_ret - rho_bg),    q = t_up_dif / t_up_dir
q comes from the 6S runs at the scene-median atmosphere (sixs_parts). rho_bg is
rho_ret averaged with an exp(-r/R) kernel over valid pixels. R = 1 km is the
satellite-altitude range (Richter 2011; PACO's EnMAP setting); airborne data uses
about 0.5 km.

This is what makes PACO's water dark in the NIR. Without it NDWI > 0.4 caught
0.003% of PACO's water pixels on DT0000122280. It over-corrects against vendors
that do no adjacency correction (ASI L2D, JPL AVIRIS-NG), so atm_post enables it
per sensor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

import sixs_parts
from rfl_qc import NODATA, _open


def kernel(range_km: float, px_m: float) -> np.ndarray:
    """Normalised exp(-r/R) kernel, truncated at 3R."""
    n = int(np.ceil(3 * range_km * 1000 / px_m))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    k = np.exp(-np.hypot(x, y) * px_m / (range_km * 1000))
    return (k / k.sum()).astype(np.float32)


def correct(rfl: str, lut_dir: Path, atm_interp: Path,
            range_km: float = 1.0, px_m: float = 30.0) -> dict:
    """Apply the adjacency correction to ENVI cube `rfl` in place. Returns q at a few wavelengths."""
    m, wl = _open(rfl)
    aot, h2o, elev = sixs_parts.scene_state(atm_interp)
    t = sixs_parts.components(lut_dir, aot, h2o, elev, wl)
    q = t["up_dif"] / np.maximum(t["up_dir"], 1e-6)
    # A pixel takes part only if valid in every band that is valid anywhere: the
    # inversion-window bands are nodata everywhere and must not void the image.
    live = [b for b in range(m.shape[1]) if (m[:, b, :] > NODATA + 9).mean() > 0.5]
    valid = np.all([m[:, b, :] > NODATA + 9 for b in live], axis=0)
    k = kernel(range_km, px_m)
    weight = fftconvolve(valid.astype(np.float32), k, mode="same")
    for b in live:
        r = np.where(valid, m[:, b, :], 0.0)
        # Clip before blurring so a few unstable snow/shadow pixels don't spread over a kilometre.
        bg = fftconvolve(np.clip(r, 0.0, 1.0), k, mode="same") / np.maximum(weight, 1e-6)
        m[:, b, :] = np.where(valid, r + q[b] * (r - bg), NODATA)
    m.flush()
    return {"aot550": aot, "h2o": h2o, "elev_km": elev, "range_km": range_km,
            **{f"q_{nm}": float(q[np.argmin(np.abs(wl - nm))]) for nm in (460, 860, 1600)}}
