"""Numbers for chapter 30 (threshold, export, Indradhanu V2 in a chain). Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_material.py > research/textbook/tools/compute_material_output.txt"""
import math

import numpy as np
from rasterio.transform import from_bounds

print("== percentile threshold as the api computes it: thr = np.percentile(finite, p); mask = composite >= thr ==")
rng = np.random.default_rng(0)
for N, p in ((956_679, 99.5), (1_474_990, 99.5), (1_000_000, 99.0), (2_400_000, 99.75), (10, 70.0)):
    v = rng.permutation(N).astype(np.float64)          # distinct values
    thr = np.percentile(v, p)
    k = int((v >= thr).sum())
    pos = p / 100 * (N - 1)
    print(f"N={N:,} p={p}: (1-p/100)*N = {(1 - p / 100) * N:,.3f}; position p/100*(N-1) = {pos:,.3f}; "
          f"flagged = N - ceil(pos) = {N - math.ceil(pos):,}; numpy flagged = {k:,}")

print("\n== Indradhanu-only composite: zeros outside the kept area are finite and join the population ==")
H, W = 1210, 1219
total = H * W
kept = 956_679
print("total", total, "kept", kept, "zeros", total - kept)
print("top 0.5% of all finite pixels:", total - math.ceil(0.995 * (total - 1)), "=", round(100 * (total - math.ceil(0.995 * (total - 1))) / kept, 3), "% of kept")
p_needed = 100 * (1 - 4784 / total)
print("percentile that flags ~4,784 pixels when all 1,474,990 are in the population:", round(p_needed, 4))

print("\n== min-max rescale and weighted composite ==")
print("Indradhanu score 7.2 in [0, 36]:", 7.2 / 36, "; MNF-RX 30 in [2, 82]:", (30 - 2) / 80)
print("weights 2 and 1 -> normalised", 2 / 3, 1 / 3, "; composite", 2 / 3 * 0.2 + 1 / 3 * 0.35)

print("\n== excluded_percent (AD23) ==")
for (h, w, inv, cloud) in ((1000, 1200, 72_000, 90_240), (800, 900, 20_000, 35_800)):
    valid = h * w - inv
    print(f"{h}x{w}: valid {valid:,}; excluded_percent = {round(100 * cloud / valid, 3)}")

print("\n== PRISMA georeferencing: rasterio from_bounds on an assumed bounding box ==")
west, south, east, north = 77.00, 12.80, 77.36, 13.13      # assumed, for illustration
tr = from_bounds(west, south, east, north, W, H)
print(tr)
print("a = (east-west)/W =", (east - west) / W, "; e = -(north-south)/H =", -(north - south) / H)
for r, c in ((0, 0), (605, 610), (1209, 1218)):
    x, y = tr * (c + 0.5, r + 0.5)
    print(f"pixel row {r} col {c}: centre lon {x:.6f} lat {y:.6f}")
print("degree of longitude at 13 deg N ~ 111.32 km * cos(13 deg) =", 111.32 * math.cos(math.radians(13)), "km")
print("pixel width in m ~", (east - west) / W * 111_320 * math.cos(math.radians(13)), "; height", (north - south) / H * 110_574)

print("\n== the funnel (AD13) ==")
print(1210 * 1219, round(1210 * 1219 * 0.94), round(1386491 * 0.69), 956679 * 0.005, round(1474990 / 4783, 1))
