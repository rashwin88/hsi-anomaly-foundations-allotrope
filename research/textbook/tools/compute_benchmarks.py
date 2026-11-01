"""Book chapter ch:benchmarks - the worked examples, checked with the repo's own functions (app/evaluation/implant.py).
Run from the repo root:
  C:/av/Scripts/python.exe research/textbook/tools/compute_benchmarks.py > research/textbook/tools/compute_benchmarks_output.txt"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.evaluation.implant import (contrast_z, foreign_fraction, match_foreignness, psf_weights,  # noqa: E402
                                    ring_mask, solve_strength, tier_of)

# top-of-atmosphere reflectance, course numbers
print("TOA", math.pi * 10 / (186 * math.cos(math.radians(30))))

# ring of a one-pixel object
obj = np.zeros((15, 15), bool); obj[7, 7] = True
ring = ring_mask(obj, np.ones_like(obj))
print("ring size 1px", int(ring.sum()))
obj2 = np.zeros((15, 15), bool); obj2[7, 7] = obj2[7, 8] = True
print("ring size domino", int(ring_mask(obj2, np.ones_like(obj2)).sum()))

# contrast z, the test's example
cube = np.zeros((2, 15, 15)); cube[1] = 0.5
yy, xx = np.mgrid[:15, :15]; dist = np.abs(yy - 7) + np.abs(xx - 7)
cube[0][dist == 2] = 0.1; cube[0][dist == 3] = 0.3; cube[0, 7, 7] = 0.42
print("z test", contrast_z(cube, np.ones(cube.shape, bool), obj, ring), "tier", tier_of(2.04))

# PSF weights, sigma 1.4, one-pixel object: the 3x3 window and the ring mean
w = psf_weights(obj, 1.4)
print("psf 3x3 window\n", w[6:9, 6:9].round(4))
print("ring mean weight", w[ring].mean().round(4), "halo weight", w[7, 8].round(4))
print("analytic exp(-1/3.92)", math.exp(-1 / 3.92), "exp(-2/3.92)", math.exp(-2 / 3.92))
shr = 1 - w[ring].mean()
print("shrink", shr.round(4), "flat halo ratio", ((w[7, 8] - w[ring].mean()) / shr).round(4))
# two-pixel object
w2 = psf_weights(obj2, 1.4)
print("psf domino window\n", w2[6:9, 6:10].round(4))

# abundance on 3x3 at strength s = 0.1
print("abundance s=0.1\n", (0.1 * w[6:9, 6:9]).round(4))

# linear mixing, two bands
b, t = np.array([0.30, 0.20]), np.array([0.10, 0.60])
for a in (0.0, 0.25, 0.5, 1.0):
    print("mix a", a, ((1 - a) * b + a * t).round(4))

# foreignness and matching
P = np.array([[1.0, 0.0, 0.0]])
print("foreign (3,4,0)", foreign_fraction(np.array([3.0, 4.0, 0.0]), P))
print("match (4,0,2) t=0.2", match_foreignness(np.array([4.0, 0.0, 2.0]), P, 0.2))
print("match (4,0,2) t=0.1", match_foreignness(np.array([4.0, 0.0, 2.0]), P, 0.1).round(4))

# solve_strength, the course example and the reverse step
for step in (-0.10, 0.10):
    s, z = solve_strength(np.array([0.22]), np.array([0.20]), np.array([0.05]), np.array([step]), 1.0)
    print("solve step", step, "s", round(s, 4), "z", round(z, 4))
print("already above", solve_strength(np.array([0.25]), np.array([0.20]), np.array([0.05]), np.array([0.1]), 0.5))

# objects-in-thousands
for name, rate, n in (("FT03 v2", 0.323, 2913), ("PCA-RX v2", 0.302, 2913)):
    print(name, rate * n)

# exercise: sigma 1.0
w1 = psf_weights(obj, 1.0)
print("sigma1 ring mean", w1[ring].mean().round(4), "halo", w1[7, 8].round(4),
      "flat halo ratio", ((w1[7, 8] - w1[ring].mean()) / (1 - w1[ring].mean())).round(4))
# exercise: z of (0.36, 0.27) vs ring 0.30, sigma 0.06
print("ex z", (abs(0.36 - 0.30) + abs(0.27 - 0.30)) / 2 / 0.06)
s, z = solve_strength(np.array([0.43]), np.array([0.40]), np.array([0.10]), np.array([-0.20]), 1.5)
print("ex solve", round(s, 4), round(z, 4))
print("ex match", match_foreignness(np.array([3.0, 0.0, 3.0]), P, 0.1).round(4))
