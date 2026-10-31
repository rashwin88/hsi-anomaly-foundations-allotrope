"""
Climatological AOT prior: pin ISOFIT's aerosol LUT axis to a window around the
NASA POWER monthly AOD at 550 nm for the scene's 1 deg cell.

Why: on PRISMA, a free AOT retrieval collapses to 0.05-0.15 where the climatology
says 0.40-0.75 (AC26-A2), leaving a +0.037 visible bias against ASI L2C. Pinning
to climatology +/- 0.15 removes about a third of it (land MAE 0.0244 -> 0.0233),
and the PRISMA harmonisation was trained on runs pinned this way (AC26-A2b).
Experiment IDs refer to research/experiments/atm-correction/EXPERIMENT_LOG.md.

The rule must stay exactly as trained: the plain monthly cell value, not a
spatially or temporally interpolated one (interpolation moves the window by up
to 0.13 in fast-changing months).

Grid: configs/aod_clim_1deg.npz, from scripts/aod_grid.py in the research folder:
POWER regional climatology (CERES SYN1deg, 2001-2025), arrays aod55/aod84
(12, 180, 360) on 1 deg cell centres. It holds no secrets and needs no network,
so it ships in the air-gapped image.

Whether a sensor gets the prior is SENSOR_RULES[sensor].aot_prior (atm_post.py):
PRISMA only. For other sensors the base LUT is copied unchanged, so a runner can
always call this step. The --record JSON goes to atm_post (--aot-prior), whose
report carries it to harmonisation (the PRISMA model's climatology covariate).

usage: python -m aot_prior --sensor prisma --lat 22.6 --lon 82.1 --date 20260604
                           --base-lut configs/enmap_summer.json --out <work>/lut.json
                           [--record <work>/aot_prior.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("aot_prior")

WINDOW = 0.15     # half width of the pinned AOT range (AC26-A2)
FLOOR = 0.05      # lowest AOT the LUT may start at
DEFAULT_GRID = Path(os.environ.get("CONFIG_ROOT", Path(__file__).resolve().parents[1] / "configs")) / "aod_clim_1deg.npz"


def month_window(lat: float, lon: float, date: str, grid: Path = DEFAULT_GRID) -> tuple[float, float, float]:
    """(climatological AOD550, window low, window high) for a scene centre and yyyymmdd date."""
    g = np.load(grid)
    i = int(np.clip(np.floor(lat + 90), 0, 179))
    j = int(np.floor(lon + 180)) % 360
    aod = float(g["aod55"][int(date[4:6]) - 1, i, j])
    if not np.isfinite(aod):
        raise ValueError(f"no climatology for cell ({lat:.2f}, {lon:.2f})")
    return aod, round(max(FLOOR, aod - WINDOW), 2), round(aod + WINDOW, 2)


def pinned_lut(base: Path, lo: float, hi: float, out: Path) -> None:
    """Copy a flat LUTConfig override with its aerosol axis pinned to [lo, hi] (3 grid points)."""
    c = json.loads(Path(base).read_text())
    c["_comment"] = f"pinned from {Path(base).name}: aerosol_2_range [{lo}, {hi}] (aot_prior)"
    c["aerosol_2_range"] = [lo, hi]
    c["aerosol_2_spacing"] = max((hi - lo) / 2, 0.01)
    Path(out).write_text(json.dumps(c, indent=1))


def main(argv: list[str] | None = None) -> int:
    from atm_post import SENSOR_RULES   # lazy: atm_post pulls in the post-processing stack
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", required=True, choices=sorted(SENSOR_RULES))
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--date", required=True, help="acquisition date, yyyymmdd")
    p.add_argument("--base-lut", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    p.add_argument("--record", type=Path, help="write the prior used (JSON) for atm_post --aot-prior")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    if SENSOR_RULES[a.sensor].aot_prior:
        aod, lo, hi = month_window(a.lat, a.lon, a.date, a.grid)
        pinned_lut(a.base_lut, lo, hi, a.out)
        rec = {"applied": True, "aod550_clim": aod, "aot_window": [lo, hi], "grid": a.grid.name,
               "lat": a.lat, "lon": a.lon, "date": a.date}
        log.info("climatology AOD550 %.3f -> aerosol window [%.2f, %.2f], %s", aod, lo, hi, a.out)
    else:
        a.out.write_text(Path(a.base_lut).read_text())
        rec = {"applied": False, "base_lut": Path(a.base_lut).name}
        log.info("%s: no AOT prior, base LUT copied to %s", a.sensor, a.out)
    if a.record:
        a.record.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
