"""Book chapter ch:bootstrap - the worked examples, checked with the repo's own functions
(app/evaluation/significance.py, app/evaluation/metrics.py). Run from the repo root:
  C:/av/Scripts/python.exe research/textbook/tools/compute_bootstrap.py > research/textbook/tools/compute_bootstrap_output.txt"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.evaluation.metrics import exact_roc, pauc  # noqa: E402
from app.evaluation.significance import object_pauc_parts, paired_bootstrap  # noqa: E402

# 1. five numbers, six given draws (1-based indices)
x = np.array([2, 4, 5, 7, 12], float)
draws = [(4, 1, 5, 5, 2), (2, 2, 3, 1, 4), (3, 5, 1, 3, 2), (1, 1, 2, 4, 3), (5, 4, 3, 2, 5), (2, 3, 4, 4, 1)]
means = np.array([x[np.array(d) - 1].mean() for d in draws])
print("mean", x.mean(), "draw means", means.tolist())
print("mean of draw means", means.mean().round(4), "sd (pop) of draw means", means.std().round(4),
      "sd (n-1)", means.std(ddof=1).round(4))
print("pop sd of x", x.std().round(4), "SE = sd/sqrt(5)", (x.std() / math.sqrt(5)).round(4))
print("percentile 2.5, 97.5 of the six", np.percentile(means, [2.5, 97.5]).round(4).tolist())
rng = np.random.default_rng(0)
big = np.array([x[rng.integers(5, size=5)].mean() for _ in range(100000)])
print("100k draws: sd", big.std().round(4), "interval", np.percentile(big, [2.5, 97.5]).round(3).tolist())

# 2. paired variance
A = np.array([0.9, 0.2, 0.6, 0.1, 0.7]); B = np.array([0.95, 0.25, 0.6, 0.15, 0.75])
va, vb = A.var(), B.var(); cov = ((A - A.mean()) * (B - B.mean())).mean()
print("VarA", va.round(5), "VarB", vb.round(5), "Cov", cov.round(5), "Var(A-B) formula", (va + vb - 2 * cov).round(5),
      "direct", (A - B).var().round(5), "unpaired", (va + vb).round(5), "ratio", ((va + vb) / (A - B).var()).round(1))

# 3. c(p) on the tiny scene vs exact ROC pAUC
bg = np.arange(1000, dtype=float)                         # 1000 distinct background scores 0..999
tvals = {"a": 997.5, "b": 993.5, "c": 1000.0, "d": 969.5}   # 2, 6, 0, 30 background pixels at or above
score = np.r_[bg, [tvals[k] for k in "abcd"]][None, :]
objects = np.r_[np.zeros(1000, int), [1, 1, 2, 3]][None, :]
valid = np.ones_like(objects, bool)
parts = object_pauc_parts(score, valid, objects)
print("parts", {k: (round(v[0], 4), v[1]) for k, v in parts.items()})
print("pAUC from parts", sum(c for c, _ in parts.values()) / sum(n for _, n in parts.values()))
f, t = exact_roc(score[0], objects[0] > 0)
print("pAUC exact", pauc(f, t, 0.01))

# 4. the paired toy, and the two hand draws
pa = {"s": {1: (1.2, 2), 2: (1.0, 1), 3: (0.0, 1)}}
pb = {"s": {1: (1.0, 2), 2: (0.9, 1), 3: (0.2, 1)}}
for d in ([1, 1, 3], [2, 2, 2], [1, 2, 3]):
    sa = sum(pa["s"][i][0] for i in d) / sum(pa["s"][i][1] for i in d)
    sb = sum(pb["s"][i][0] for i in d) / sum(pb["s"][i][1] for i in d)
    print("draw", d, "A", round(sa, 4), "B", round(sb, 4), "B-A", round(sb - sa, 4))
r = paired_bootstrap(pa, pb, n=1000)
print("toy bootstrap", {k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(float(v), 4)) for k, v in r.items()})

# 5. percentile positions
for n in (1000, 2000):
    print(n, "draws: 0-based positions", (n - 1) * 0.025, (n - 1) * 0.975)

# 6. scenes: multisets of 4 from 4
print("scene draws C(7,4) =", math.comb(7, 4))

# 7. objects vs pixels: an illustrative simulation (assumed numbers)
rng = np.random.default_rng(1)
sizes = rng.choice([1, 1, 1, 2, 2, 3, 5, 10], size=300)
cobj = rng.random(300)                                   # each object's c, shared by its pixels
pix_c = np.concatenate([np.clip(c + rng.normal(0, 0.05, s), 0, 1) for c, s in zip(cobj, sizes)])
pix_obj = np.repeat(np.arange(300), sizes)
sum_c = np.bincount(pix_obj, weights=pix_c); cnt = np.bincount(pix_obj)
def obj_boot():
    k = rng.integers(300, size=300); return sum_c[k].sum() / cnt[k].sum()
def pix_boot():
    k = rng.integers(len(pix_c), size=len(pix_c)); return pix_c[k].mean()
so = np.std([obj_boot() for _ in range(2000)]); sp = np.std([pix_boot() for _ in range(2000)])
print("sim: objects", 300, "pixels", len(pix_c), "SE object-boot", round(so, 4), "SE pixel-boot", round(sp, 4),
      "ratio", round(so / sp, 2))

# 8. design effect example
m, rho = 4, 0.8
deff = 1 + (m - 1) * rho
print("deff", deff, "n_eff of 3636 pixels", 3636 / deff)

# 9. interval half-widths of the README table
for name, lo, hi in (("cloud", 0.0047, 0.0101), ("adapt", -0.0176, -0.0028), ("ft06", -0.0174, 0.0037),
                     ("ft09b", 0.0002, 0.0118), ("ft09c", -0.0026, 0.0064), ("ft09c-b", -0.0076, -0.0010)):
    print(name, "half-width", round((hi - lo) / 2, 4), "centre", round((hi + lo) / 2, 4))

# 10. the exact bootstrap distribution of the five-number mean: all 5^5 = 3125 equally likely draws
import itertools
allm = np.array([x[list(d)].mean() for d in itertools.product(range(5), repeat=5)])
vals, cnt = np.unique(np.round(allm, 1), return_counts=True)
print("exact: draws", len(allm), "sd", allm.std().round(4), "2.5/97.5 pct", np.percentile(allm, [2.5, 97.5]).round(3).tolist())
print("hist", " ".join(f"({v:g},{c})" for v, c in zip(vals, cnt)))
