"""Numbers for chapter 03-sensors (the 165-band grid), checked against the source.
Run from the repo root:  python research/textbook/tools/compute_sensors.py
"""
import math, sys
import numpy as np
sys.path.insert(0, ".")
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as G, BandFilterConfig
from app.utils.data_transformations.spectral_band_filter import DEFAULT_EXCLUSION_RANGES

full = np.arange(460.0, 2450.0 + 5.0, 10.0)
print("full ladder:", len(full), full[0], full[-1])
print("arange without half step:", len(np.arange(460.0, 2450.0, 10.0)))
tot = 0
for lo, hi in BandFilterConfig().exclusion_ranges:
    n_hand = max(0, math.floor(hi / 10) - max(math.ceil(lo / 10), 46) + 1)
    n_src = int(((full >= lo) & (full <= hi)).sum())
    tot += n_src
    print("window (%5g,%5g): by formula %2d, by source mask %2d" % (lo, hi, n_hand, n_src))
print("removed", tot, " remaining", len(full) - tot, " len(DEFAULT_COMMON_WAVELENGTH_GRID) =", len(G))
br = list(np.nonzero(np.diff(G) > 10)[0])
starts, ends = [0] + [b + 1 for b in br], br + [len(G) - 1]
for a, b in zip(starts, ends):
    print("segment idx %3d..%3d  %4.0f..%4.0f nm  %2d bands" % (a, b, G[a], G[b], b - a + 1))
for i in (0, 43, 45, 46, 48, 61, 62, 80, 81, 114, 115, 164):
    print("index", i, "->", G[i], "nm")
print("SpectralBandFilter DEFAULT_EXCLUSION_RANGES:", DEFAULT_EXCLUSION_RANGES)
# Layout offsets on a C=3, H=2, W=2 cube
C, H, W = 3, 2, 2
c, h, w = 2, 1, 0
print("BSQ offset", c * H * W + h * W + w, " BIL offset", h * C * W + c * W + w, " BIP offset", h * W * C + w * C + c)
print("PRISMA bytes", 1210 * 1219 * 239 * 4, " values", 1210 * 1219 * 239)
