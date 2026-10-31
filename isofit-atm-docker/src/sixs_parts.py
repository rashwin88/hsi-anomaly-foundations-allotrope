"""
6S transmittance components at a scene's retrieved atmosphere, for the post-hoc steps.

ISOFIT with the sRTMnet .h5 emulator runs single-part transmittance: its LUT
(`lut_full/6S.lut.zarr`) keeps total transmittances only. The adjacency and
terrain corrections need the split into direct and diffuse, up and down. The 6S
runs ISOFIT made to build that LUT sit next to it, one text output per grid point
(`AOT550-<a>_H2OSTR-<w>[_observer_zenith-<z>]_surface_elevation_km-<e>`), and each
prints a "Multipart transmittances" table. This module reads those.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio

POINT = re.compile(r"AOT550-([\d.]+)_H2OSTR-([\d.]+)_(?:.*_)?surface_elevation_km-([\d.]+)$")


def multipart(path: Path) -> np.ndarray:
    """(5, n) rows: wavelength nm, t_down_dir, t_down_dif, t_up_dir, t_up_dif."""
    rows, on = [], False
    for line in open(path):
        if "Multipart transmittances" in line:
            on = True
            continue
        if on:
            p = line.strip("* \n").split()
            if len(p) == 5:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
            elif rows:
                break
    if not rows:
        raise ValueError(f"no multipart transmittance table in {path}")
    a = np.array(rows).T
    a[0] *= 1000.0  # 6S prints micrometres
    return a


def scene_state(atm_interp: Path) -> tuple[float, float, float]:
    """Scene-median (AOT550, H2O g/cm2, surface elevation km) from ISOFIT's *_atm_interp.

    Bands are found by name. The elevation band exists only with --pressure_elevation;
    without it the elevation is reported as 0 km.
    """
    with rasterio.open(atm_interp) as src:
        names = [(d or "").upper() for d in src.descriptions]
        med = {}
        for key in ("AOT550", "H2OSTR", "SURFACE_ELEVATION_KM"):
            if key in names:
                b = src.read(names.index(key) + 1)
                med[key] = float(np.nanmedian(np.where(b > -9990, b, np.nan)))
    return med["AOT550"], med["H2OSTR"], med.get("SURFACE_ELEVATION_KM", 0.0)


def components(lut_dir: Path, aot: float, h2o: float, elev_km: float,
               wl_nm: np.ndarray) -> dict[str, np.ndarray]:
    """Down/up direct/diffuse transmittances on `wl_nm` at the scene state.

    Nearest grid point in H2O and elevation, linear in AOT between the bracketing
    points. Where the LUT has extra geometry dimensions (PRISMA's observer zenith),
    one geometry per AOT is used: the split varies far less with view angle than
    with aerosol.
    """
    pts = [(*map(float, m.groups()), f) for f in Path(lut_dir).glob("AOT550-*")
           if (m := POINT.search(f.name))]
    if not pts:
        raise FileNotFoundError(f"no 6S point outputs (AOT550-*) in {lut_dir}")
    h = min({p[1] for p in pts}, key=lambda v: abs(v - h2o))
    e = min({p[2] for p in pts}, key=lambda v: abs(v - elev_km))
    sel = sorted({p[0]: p for p in sorted(pts) if p[1] == h and p[2] == e}.values())
    aots = np.array([p[0] for p in sel])
    i = int(np.clip(np.searchsorted(aots, aot), 1, max(len(aots) - 1, 1)))
    lo, hi = sel[i - 1], sel[min(i, len(sel) - 1)]
    w = 0.0 if hi is lo else float(np.clip((aot - lo[0]) / (hi[0] - lo[0]), 0, 1))
    a, b = multipart(lo[3]), multipart(hi[3])
    t = (1 - w) * a + w * b
    keys = ("down_dir", "down_dif", "up_dir", "up_dif")
    return {k: np.interp(wl_nm, a[0], t[j + 1]) for j, k in enumerate(keys)}
