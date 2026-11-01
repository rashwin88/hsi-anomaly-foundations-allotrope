"""Numbers computed for chapter 35 (the sorting bug). Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_sortbug.py > research/textbook/tools/compute_sortbug_output.txt
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")

print("numpy", np.__version__)

# 1. np.interp on ascending and descending sample lists (the repo's test numbers)
q = np.array([450, 500, 550, 600, 650, 700, 750], float)
up, vals = np.array([500.0, 600.0, 700.0]), np.array([1.0, 2.0, 4.0])
print("ascending :", np.interp(q, up, vals))
print("descending:", np.interp(q, up[::-1], vals[::-1]))

# 2. the old (unsorted) and new (sorted) resample matrices, bands listed 700, 600, 500
wl = np.array([700.0, 600.0, 500.0])
grid = np.array([550.0, 650.0])
M_old = np.stack([np.interp(grid, wl, np.eye(3)[i]) for i in range(3)], 1)
order = np.argsort(wl)
M_new = np.zeros((2, 3))
M_new[:, order] = np.stack([np.interp(grid, wl[order], np.eye(3)[i]) for i in range(3)], 1)
pix = np.array([4.0, 2.0, 1.0])
print("order", order)
print("M_old\n", M_old, "\n -> pixel", M_old @ pix)
print("M_new\n", M_new, "\n -> pixel", M_new @ pix)

# 3. a signature difference spectrum listed descending, carried to host bands
src = np.array([700.0, 600.0, 500.0]); d = np.array([0.06, 0.03, -0.02])
host = np.array([520.0, 580.0, 650.0])
print("signature correct:", np.interp(host, src[::-1], d[::-1]))
print("signature garbled:", np.interp(host, src, d))

# 4. the common grid against the real host and bench band lists
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as G
G = np.asarray(G, float)
host_wl = np.load("C:/es/implant/bench-v2.1/prisma/PRS_L2D_STD_20230311055340/wavelengths.npy")
print("host bands", len(host_wl), "first", host_wl[0], "last", host_wl[-1])
M = np.stack([np.interp(G, host_wl, np.eye(len(host_wl))[i]) for i in range(len(host_wl))], 1)
cols = sorted(set(np.flatnonzero(M.sum(0))))
print("host: columns used by old rows", cols, [round(float(host_wl[c]), 2) for c in cols])
print("host: grid bands <= last listed", int((G <= host_wl[-1]).sum()), "of", len(G))
bench = np.array(json.load(open("C:/es/implant/real_v2.json"))["wavelengths_nm"]["set1"])
Mb = np.stack([np.interp(G, bench, np.eye(len(bench))[i]) for i in range(len(bench))], 1)
print("bench set1 bands", len(bench), "first", bench[0], "last", bench[-1],
      "columns used", sorted(set(np.flatnonzero(Mb.sum(0)))))

# 5. objects lost, standard errors
print("top1% lost", (0.442 - 0.158) * 909, "top0.1% lost", (0.141 - 0.056) * 909)
p, n = 0.35, 2817
se = np.sqrt(p * (1 - p) / n)
print("se", p * (1 - p), p * (1 - p) / n, se, "gap/se", 0.006 / se)
p = 0.34
se2 = np.sqrt(p * (1 - p) / n)
print("se p=0.34", se2, "2.0/se", 0.020 / se2, "2.4/se", 0.024 / se2)
