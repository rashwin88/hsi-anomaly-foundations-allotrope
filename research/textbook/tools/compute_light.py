"""Numbers for chapter 02-light, computed from repo files.
Run from the repo root:  python research/textbook/tools/compute_light.py
Output saved to research/textbook/tools/compute_light_output.txt
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as G

s = json.load(open("app/constants/hyperspectral.json"))
m, sd = np.array(s["mean"]), np.array(s["std"])
print("len mean/std:", len(m), len(sd), " num_patches:", s.get("num_patches"))
print("band0 mean/std:", m[0], sd[0])
print("max mean %.4f at %.0f nm; min mean %.4f at %.0f nm" % (m.max(), G[m.argmax()], m.min(), G[m.argmin()]))
for wl in (660.0, 760.0, 860.0, 1650.0, 2210.0):
    i = int(np.where(G == wl)[0][0]); print("idx %3d  %6.0f nm  mean %.4f std %.4f" % (i, wl, m[i], sd[i]))
# pgfplots coordinates, one segment per spectral family (gaps between)
br = list(np.nonzero(np.diff(G) > 10)[0])
starts, ends = [0] + [b + 1 for b in br], br + [len(G) - 1]
for k, (a, b) in enumerate(zip(starts, ends)):
    print("SEG%d mean:" % k, " ".join("(%.0f,%.4f)" % (G[i], m[i]) for i in range(a, b + 1)))
    print("SEG%d lo  :" % k, " ".join("(%.0f,%.4f)" % (G[i], m[i] - sd[i]) for i in range(a, b + 1)))
    print("SEG%d hi  :" % k, " ".join("(%.0f,%.4f)" % (G[i], m[i] + sd[i]) for i in range(a, b + 1)))
# Gaussian SRF: sigma from FWHM
for f in (10.0, 12.0):
    print("FWHM %.0f -> sigma %.4f" % (f, f / (2 * np.sqrt(2 * np.log(2)))))
print("2*sqrt(2 ln2) =", 2 * np.sqrt(2 * np.log(2)))
