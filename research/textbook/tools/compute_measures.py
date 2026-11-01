"""Book chapter ch:measures - the worked examples, checked with the repo's own functions (app/evaluation/metrics.py).
Run from the repo root:
  C:/av/Scripts/python.exe research/textbook/tools/compute_measures.py > research/textbook/tools/compute_measures_output.txt"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.evaluation.metrics import auc_exact, chance_hit, exact_roc, pauc, tpr_at  # noqa: E402

# six pixels
s = np.array([0.95, 0.85, 0.70, 0.60, 0.40, 0.20])
g = np.array([1, 0, 1, 1, 0, 0], bool)
fpr, tpr = exact_roc(s, g)
print("six pixels fpr", fpr.round(4).tolist())
print("six pixels tpr", tpr.round(4).tolist())
print("auc", auc_exact(fpr, tpr), "7/9 =", 7 / 9)
print("pauc 0.5", pauc(fpr, tpr, 0.5), "5/9 =", 5 / 9)
print("pauc 0.01", pauc(fpr, tpr, 0.01))
print("tpr_at 0.001", tpr_at(fpr, tpr, 0.001))
wins = sum((a > b) + 0.5 * (a == b) for a in s[g] for b in s[~g])
print("mann-whitney", wins, "/", g.sum() * (~g).sum())

# a tie: target and background both at 0.5
s2 = np.array([0.9, 0.5, 0.5, 0.1])
g2 = np.array([1, 1, 0, 0], bool)
f2, t2 = exact_roc(s2, g2)
print("tie fpr", f2.tolist(), "tpr", t2.tolist(), "auc", auc_exact(f2, t2), "pauc0.25", pauc(f2, t2, 0.25))

# detectors A, B (course AD16) as piecewise curves
A = (np.array([0, 0.01, 0.30, 1]), np.array([0, 0.40, 0.80, 1]))
B = (np.array([0, 0.01, 0.10, 1]), np.array([0, 0.02, 0.80, 1]))
for n, (f, t) in (("A", A), ("B", B)):
    print(n, "auc", auc_exact(f, t), "pauc0.01", pauc(f, t, 0.01))

# random scores: pauc near max_fpr/2
rng = np.random.default_rng(0)
gr = rng.random(2_000_000) < 0.01
pr = [pauc(*exact_roc(rng.random(gr.size), gr), 0.01) for _ in range(3)]
print("random pauc (3 runs)", np.round(pr, 5).tolist())

# McClish standardisation of a normalised pAUC of 0.170 at max_fpr 0.01
m = 0.01
A_raw = 0.170 * m
mn, mx = m * m / 2, m
print("McClish(0.170)", 0.5 * (1 + (A_raw - mn) / (mx - mn)))

# chance levels: dilated sizes
def dil(cells):
    a = np.zeros((9, 9), bool)
    for y, x in cells:
        a[y + 4, x + 4] = True
    return int(ndimage.binary_dilation(a).sum())

for name, cells in (("1 px", [(0, 0)]), ("domino", [(0, 0), (0, 1)]), ("3 in a row", [(0, 0), (0, 1), (0, 2)]),
                    ("L triomino", [(0, 0), (1, 0), (1, 1)]), ("2x2", [(0, 0), (0, 1), (1, 0), (1, 1)]),
                    ("3x3", [(y, x) for y in range(3) for x in range(3)])):
    m_ = dil(cells)
    print(name, "m =", m_, "chance top1%", round(chance_hit(m_, 1.0), 4), "top0.1%", round(chance_hit(m_, 0.1), 4))

# 4-connectivity: two diagonal pixels are two objects
a = np.zeros((4, 4), int); a[1, 1] = 1; a[2, 2] = 1
print("diagonal pair -> objects", ndimage.label(a)[1])

# compute_roc thresholds
for n in (600_000, 1_000_000, 2_000_000):
    print(n, "pixels -> thresholds", min(256, max(20, n // 10000 + 64)))
