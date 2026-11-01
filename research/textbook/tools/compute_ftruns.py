"""Numbers computed for chapter 36 (FT01 to FT05). Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_ftruns.py > research/textbook/tools/compute_ftruns_output.txt
"""
import math

import numpy as np
from scipy import ndimage

# IoU toy
print("IoU toy", 4 / (6 + 5 - 4))

# segmentation head parameter count (shapes from the shipped FT04 checkpoint)
proj = 128 * (32 + 64 + 160 + 256) + 4 * 128
fuse = 128 * 512 + 128
ref0 = 128 * 128 * 9 + 128
ref2 = 80 * 128 * 9 + 80
print("seg head", proj, fuse, ref0, ref2, proj + fuse + ref0 + ref2)
try:
    import torch
    sd = torch.load("C:/es/ship/spectral_stem_finetune/spectral_stem_finetune_v2.0.0_epoch10.pt",
                    map_location="cpu", weights_only=False)["model_state_dict"]
    print("seg head from checkpoint", sum(v.numel() for k, v in sd.items() if k.startswith("seg_head.")))
except Exception as e:  # torch or the checkpoint missing
    print("checkpoint not read:", e)

# FT02 gate
s = [0.931, 0.852, 0.757, 0.879]
print("FT02 best.pt mean", sum(s) / 4, "gate miss", 0.861 - round(sum(s) / 4, 3))
f60 = [0.930, 0.860, 0.775, 0.881]
print("criterion 2 diffs", [round(a - b, 3) for a, b in zip(s, f60)])
print("record", 202 + 337 + 177 + 17)

# edge measures on the 3x3 patch
P = np.array([[0.10, 0.10, 0.30], [0.10, 0.12, 0.30], [0.10, 0.10, 0.31]])
gx = (P[1, 2] - P[1, 0]) / 2
gy = (P[2, 1] - P[0, 1]) / 2
print("centre central gx gy mag", gx, gy, math.hypot(gx, gy))
print("centre max-abs-neighbour", max(abs(P[1, 1] - P[0, 1]), abs(P[1, 1] - P[2, 1]), abs(P[1, 1] - P[1, 0]),
                                      abs(P[1, 1] - P[1, 2])))
T = np.array([[0.10, 0.10, 0.10], [0.10, 0.30, 0.10], [0.10, 0.10, 0.10]])
print("spike central", (T[1, 2] - T[1, 0]) / 2, (T[2, 1] - T[0, 1]) / 2, "spike max-abs", 0.20)

# rank minus median rank of its edge bin
names = list("ABCDEFGH")
score = dict(A=9, B=10, C=8, D=7, E=3, F=2, G=1, H=4)
strong = set("ABCD")
order = sorted(names, key=lambda n: score[n])
rank = {n: i + 1 for i, n in enumerate(order)}
med_s = np.median([rank[n] for n in names if n in strong])
med_w = np.median([rank[n] for n in names if n not in strong])
corr = {n: rank[n] - (med_s if n in strong else med_w) for n in names}
print("ranks", rank, "medians", med_s, med_w, "corrected", corr)

# gradient-L1 row
x = np.array([0.10, 0.10, 0.50, 0.50]); xh = np.array([0.10, 0.20, 0.40, 0.50])
print("value L1", np.abs(xh - x).mean(), "grad L1", np.abs(np.diff(xh) - np.diff(x)).mean())
print("edge term change", 0.01163 - 0.01153, (0.01163 - 0.01153) / 0.01163)

# EMA teacher
t = 0.50
for i in range(2):
    t = 0.999 * t + 0.001 * 0.80
    print("EMA step", i + 1, t)
print("EMA half-life steps", math.log(2) / -math.log(0.999))

# L2-SP
print("L2-SP", 0.01 / 2 * ((0.6 - 0.5) ** 2 + (-0.1 + 0.2) ** 2))

# match_foreignness
inside, outside, f = 9.0, 16.0, 0.1
b2 = f / (1 - f) * inside / outside
print("beta^2", b2, "beta", math.sqrt(b2))

# FT05 near weight
print("diamond r=3", ndimage.binary_dilation(np.pad([[1]], 5), iterations=3).sum(),
      "two-pixel", ndimage.binary_dilation(np.pad([[1, 1]], 5), iterations=3).sum())
N, n = 0.65 * 16384, 0.65 * 25
print("hidden", N, "near", n, "share w1", n / N, "share w5", 5 * n / (N - n + 5 * n))
print("pace", 160 / 18, 18 / 160, 10 * 160 / 60, "fix", 161 / 43)

# SVD = eigen of X^T X
X = np.array([[1, 1], [-1, -1], [2, -2], [-2, 2]], float)
print("XtX", X.T @ X, "eig", np.linalg.eigh(X.T @ X), "sv", np.linalg.svd(X)[1])

# FT04 pseudo labels
print("bytes", 0.9 * 255, 0.1 * 255, 237 / 255, 240 / 255, 128 / 255, 13 / 255, 235 / 255)
print("flatness", 0.18 / 0.30, (0.30 + 0.27 + 0.24) / 3)

# weighted BCE
def bce(y, p):
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
per = {
    "water": (bce(1, 0.9) + bce(0, 0.2)) / 2,
    "cloud": (bce(1, 0.7) + bce(0, 0.1)) / 2,
    "thin": (bce(0, 0.4) + bce(1, 0.6)) / 2,
    "shadow": (bce(1, 0.3) + bce(0, 0.2)) / 2,
}
for k, v in per.items():
    print("bce", k, round(v, 4))
vals = list(per.values()); w = [1, 1, 1, 3]
print("unweighted", sum(vals) / 4, "weighted", sum(a * b for a, b in zip(w, vals)) / sum(w))
print("course example", (0.2 + 0.3 + 0.5 + 0.8) / 4, (0.2 + 0.3 + 0.5 + 3 * 0.8) / 6, 0.8 / 1.8, 2.4 / 3.4)

# decomposition and factorial
print("model", 0.162 - 0.159, "excl", 0.170 - 0.162, "share", 0.008 / 0.011)
cells = {("FT03", "FT03"): 0.166, ("FT03", "FT04"): 0.165, ("FT04", "FT03"): 0.170, ("FT04", "FT04"): 0.170}
print("model effect", (0.170 + 0.170) / 2 - (0.166 + 0.165) / 2, "mask effect", (0.165 + 0.170) / 2 - (0.166 + 0.170) / 2)

# standard error on bench-v2.1
print("se", math.sqrt(0.35 * 0.65 / 2817))
