"""
Native-band groups for harmonisation. Every band belongs to exactly one group,
and each group is fitted and scored on its own. That keeps the known noise of
sensor edges and absorption windows out of the core signal, both in the models
(core outputs use core inputs only) and in the reported performance.

  core     460-2450 nm outside the four windows. This is the spectral domain of the
           Allotrope 165-band grid (BandFilterConfig exclusions), where every
           harmonisation result so far was measured.
  edge_lo  below 460 nm      (EnMAP from ~420, PRISMA from ~400)
  edge_hi  above 2450 nm     (PRISMA up to ~2500; EnMAP stops at ~2445)
  weak     912-978, 1131-1152 nm: water-vapour bands where ISOFIT still returns values
  strong   1350-1450, 1800-1950 nm: saturated water vapour. ISOFIT returns nodata
           over most of them (outside its 350-1360 / 1410-1800 / 1970-2500 nm
           inversion windows, rfl_qc.INVERSION_WINDOWS_NM).

Boundaries are inclusive. The windows lie inside 460-2450 nm and do not overlap,
so only the edges could compete with them, and they cannot.
"""
from __future__ import annotations

import numpy as np

CORE_RANGE_NM = (460.0, 2450.0)
WEAK_NM = [(912.0, 978.0), (1131.0, 1152.0)]
STRONG_NM = [(1350.0, 1450.0), (1800.0, 1950.0)]
GROUPS = ("core", "edge_lo", "edge_hi", "weak", "strong")


def _in(w: np.ndarray, spans) -> np.ndarray:
    return np.any([(w >= lo) & (w <= hi) for lo, hi in spans], axis=0)


def band_groups(wl_nm) -> np.ndarray:
    """Group name per band, for wavelengths in nm (any order)."""
    w = np.asarray(wl_nm, float)
    g = np.full(w.shape, "core", dtype=object)
    g[_in(w, WEAK_NM)] = "weak"
    g[_in(w, STRONG_NM)] = "strong"
    g[w < CORE_RANGE_NM[0]] = "edge_lo"
    g[w > CORE_RANGE_NM[1]] = "edge_hi"
    return g


def group_masks(wl_nm) -> dict[str, np.ndarray]:
    """{group: boolean band mask}; the masks partition the bands."""
    g = band_groups(wl_nm)
    return {name: g == name for name in GROUPS}
