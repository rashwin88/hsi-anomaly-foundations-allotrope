"""Numbers computed for chapter 38 (scoring gains and gated scene adaptation). Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_scoringgains.py > research/textbook/tools/compute_scoringgains_output.txt
"""
import csv
import math

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

# matched filter, 2 bands
S = np.array([[2.0, 1.0], [1.0, 2.0]]); t = np.array([1.0, 0.0])
Si = np.linalg.inv(S)
w = Si @ t / (t @ Si @ t)
print("Sigma^-1", Si, "w", w, "t'Si t", t @ Si @ t)
snr = lambda v: (v @ t) / math.sqrt(v @ S @ v)
print("SNR matched", snr(w), "SNR naive w=t", snr(t), "sqrt(t'Si t)", math.sqrt(t @ Si @ t))

# spatial 3-tap
k = np.array([0.25, 0.5, 0.25])
sd = math.sqrt((k ** 2).sum())
print("noise var", (k ** 2).sum(), "sd", sd, "footprint", k @ [1, 2, 1] / sd, "single", k @ [0, 2, 0] / sd,
      "two-pixel", (0.5 * 3 + 0.25 * 3) / sd, "sqrt6", math.sqrt(6))

# the 12-pixel strip
r = np.array([0, 4.5, -4.2, 0, 3, 4, 5, 3, 0, 0, 4.8, 0])
b = np.convolve(np.pad(r, 1), k, mode="valid")
print("plain scores", r ** 2)
print("blurred", np.round(b, 3), "scores", np.round(b ** 2, 2))
print("plain top4 (1-based)", np.argsort(-r ** 2)[:4] + 1, "blurred top4", np.argsort(-b ** 2)[:4] + 1)
r2 = r.copy(); r2[10] = 5.2
b2 = np.convolve(np.pad(r2, 1), k, mode="valid")
print("exercise: b11", b2[10], b2[10] ** 2, "top5", np.argsort(-b2 ** 2)[:5] + 1)

# object counts by size
for p, n in ((0.365, 561), (0.333, 561), (0.524, 170), (0.453, 170), (0.80, 20), (0.75, 20)):
    print("count", p, n, p * n)

# rank fusion, 5 pixels
rm = np.array([9, 12, 3, 2, 1.0]); pca = np.array([8, 1, 2, 10, 3.0])
a, c = rankdata(rm) / 5, rankdata(pca) / 5
print("ranks", a, c, "mean", (a + c) / 2, "max", np.maximum(a, c))
q = 0.01
print("P(max>1-q)", 1 - (1 - q) ** 2, "P(mean>1-q)", 2 * q * q)

# Gaussian mixture, 1-D
def npdf(x, m, v):
    return math.exp(-(x - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
x = 3.0
p1, p2 = 0.8 * npdf(x, 0, 1), 0.2 * npdf(x, 5, 4)
print("densities", npdf(x, 0, 1), npdf(x, 5, 4), "weighted", p1, p2, "resp", p1 / (p1 + p2), p2 / (p1 + p2))
d1, d2 = (x - 0) ** 2 / 1, (x - 5) ** 2 / 4
print("maha", d1, d2, "resp-weighted", p1 / (p1 + p2) * d1 + p2 / (p1 + p2) * d2)
mu = 0.8 * 0 + 0.2 * 5; ex2 = 0.8 * (1 + 0) + 0.2 * (4 + 25)
print("global mean", mu, "var", ex2 - mu * mu, "global score", (x - mu) ** 2 / (ex2 - mu * mu))

# k-means in 1-D
pts = np.array([0, 1, 2, 8, 9, 10.0]); C = np.array([0, 2.0])
for it in range(3):
    lab = np.abs(pts[:, None] - C[None]).argmin(1)
    C = np.array([pts[lab == j].mean() for j in range(2)])
    print("kmeans iter", it + 1, lab, C)

# the course's type toy
field = [1] * 7 + [-1] * 7 + [2.5, -2.5]; built = [-2, -2, 2, 2, -6, 6]
allv = sum(v * v for v in field + built) / 22
fv = sum(v * v for v in field) / 16; bv = sum(v * v for v in built) / 6
print("global var", allv, "target", 36 / allv, "edge", 6.25 / allv, "field var", fv, "edge", 6.25 / fv,
      "built var", bv, "target", 36 / bv, "implant 4: global", 16 / allv, "per type", 16 / fv)

# float16 max
print("float16 max", (2 - 2 ** -10) * 2 ** 15, np.finfo(np.float16).max)

# gate dilation
for a_, b_ in ((1, 1), (2, 2), (3, 3)):
    m = np.zeros((15, 15), bool); m[5:5 + a_, 5:5 + b_] = True
    n = ndimage.binary_dilation(m, iterations=2).sum()
    print("dilate", a_, b_, n, "formula", a_ * b_ + 4 * (a_ + b_) + 4, "growth", n / (a_ * b_))
m = np.zeros((10, 10), bool); m[0, 0] = True
print("corner", ndimage.binary_dilation(m, iterations=2).sum())

# learning-rate probe
for lr, f, l in (("1e-5", 0.0495, 0.0389), ("3e-5", 0.0459, 0.0368), ("1e-4", 0.0461, 0.0354)):
    print("lr", lr, f - l, (f - l) / f)

# leave-one-scene-out from the CSV
rows = list(csv.DictReader(open("research/experiments/scene-adaptation/eval_per_scene.csv")))
P = {}
for row in rows:
    if row["dataset"] == "prisma_bench" and row["scorer"] == "RM, cloud excluded":
        P[(row["model"], row["scene"])] = float(row["pauc_0.01"])
scenes = ["set1", "set2", "set3", "set4"]
models = ["adapt50", "adapt200", "adapt1000"]
for mdl in ["FT04_ep10"] + models + ["adapt200_ungated"]:
    print(mdl, [round(P[(mdl, s)], 4) for s in scenes], "mean", round(sum(P[(mdl, s)] for s in scenes) / 4, 4))
for rounding in (False, True):
    g = (lambda v: round(v, 3)) if rounding else (lambda v: v)
    out = []
    for s in scenes:
        others = [o for o in scenes if o != s]
        best = max(models, key=lambda mdl: sum(g(P[(mdl, o)]) for o in others) / 3)
        out.append((s, best, g(P[(best, s)])))
    print("LOSO rounded" if rounding else "LOSO exact", out, sum(v for _, _, v in out) / 4)

# targets' share and pixel visits
print("share", 1000 / 1_440_000, "visits", 1000 * 8 * 128 * 128, 1000 * 8 * 128 * 128 / 1_440_000)
print("time", 1000 / 1.6, (50 + 200 + 1000 + 200) * 4, (50 + 200 + 1000 + 200) * 4 / 1.6)
