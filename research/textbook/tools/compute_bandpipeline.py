"""Numbers for chapter 05-bandpipeline, computed from repo code and files.
Run from the repo root:  python research/textbook/tools/compute_bandpipeline.py
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from scipy.interpolate import CubicSpline, PchipInterpolator
from app.utils.data_transformations.spectral_resampler import resample_to_common_grid, _assign_families
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily as F
from app.utils.pixel_fill.nearest_valid_fill import nearest_valid_fill

# 1. np.interp ascending vs descending (the repo test numbers 500/600/700 -> 1/2/4)
print("interp ascending  [550,650]:", np.interp([550, 650], [500, 600, 700], [1, 2, 4]))
print("interp descending [520,550,650,690]:", np.interp([520, 550, 650, 690], [700, 600, 500], [4, 2, 1]))

# 2. spline overshoot vs PCHIP on a step 0,0,1,1
x, y = [0, 1, 2, 3], [0, 0, 1, 1]
cs, pc = CubicSpline(x, y), PchipInterpolator(x, y)
for q in (0.5, 1.5, 2.5):
    print("q=%.1f  natural-not-a-knot spline %.4f   pchip %.4f" % (q, cs(q), pc(q)))

# 3. the product resampler on a 3-pixel toy cube with DESCENDING source wavelengths
src = np.array([1030.0, 1015.0, 1000.0])              # descending, as PRISMA lists them
cube = np.array([0.30, 0.25, 0.20], dtype=np.float32)[:, None, None]   # (3,1,1)
val = np.ones((3, 1, 1), dtype=np.int8)
out, v, fam = resample_to_common_grid(cube, val, src, np.array([1000.0, 1010.0, 1020.0, 1030.0]), [F.SWIR] * 3)
print("resampler (sorts internally):", out[:, 0, 0])

# 4. family-assignment check with descending source order
print("_assign_families([500,2000]) for src [2000,1500,800,500] fam [SWIR,SWIR,VNIR,VNIR]:",
      [f.value for f in _assign_families(np.array([500.0, 2000.0]), [F.SWIR, F.SWIR, F.VNIR, F.VNIR],
                                         np.sort(np.array([2000.0, 1500.0, 800.0, 500.0])))])

# 5. invalid pixel at 0 after z-score, across the 165 bands
s = json.load(open("app/constants/hyperspectral.json"))
m, sd = np.array(s["mean"]), np.array(s["std"])
z0 = -m / sd
print("z of a zeroed pixel: band0 %.4f, min %.4f (idx %d), max %.4f (idx %d)" % (z0[0], z0.min(), z0.argmin(), z0.max(), z0.argmax()))

# 6. nearest_valid_fill on the docstring 1-D example and a 3x3 example
# (a one-row image cannot be passed: mask.squeeze() would drop the row axis, so use two equal rows)
p = np.array([[[30, 32, 0, 0, 0, 25, 28]] * 2], dtype=float)
mk = np.array([[[1, 1, 0, 0, 0, 1, 1]] * 2])
print("fill 1-D (row 0 of a 2-row image):", nearest_valid_fill(p, mk)[0, 0])
p2 = np.array([[[0.10, 0.12, 0.00], [0.11, 0.00, 0.00], [0.00, 0.00, 0.30]]])
m2 = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 1]]])
print("fill 3x3:\n", nearest_valid_fill(p2, m2)[0])
