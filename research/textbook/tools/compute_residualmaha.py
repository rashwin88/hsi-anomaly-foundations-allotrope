"""Numbers for chapter 29 (residual Mahalanobis, the ring, integral images). Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_residualmaha.py > research/textbook/tools/compute_residualmaha_output.txt
Checks the hand examples against the product's own functions in app/utils/anomaly_detection/residual_mahalanobis.py."""
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.utils.anomaly_detection.residual_mahalanobis import (_box, local_mean_score, residual_mahalanobis,
                                                             whitened_pca)

print("== 3-band worked example: four normal residuals ==")
R = np.array([[1, 2, -1], [1, -1, -1], [-2, -1, -1], [0, 0, 3]], float)
mu = R.mean(0)
print("mean", mu)
S = np.cov(R.T)          # divides by N - 1 = 3, as the code does
print("covariance (np.cov, /3)\n", S)
Si = np.linalg.inv(S)
print("inverse\n", Si, "\n x3:\n", Si * 3, "\n det", np.linalg.det(S))
ev, U = np.linalg.eigh(S)
print("eigenvalues", ev, "\neigenvectors (columns)\n", U)
for name, r in [("r1", R[0]), ("r2", R[1]), ("r3", R[2]), ("r4", R[3]),
                ("P", np.array([3, 3, 0.])), ("Q", np.array([2, -2, 0.])), ("Rr", np.array([0, 0, 4.]))]:
    d2 = r @ Si @ r
    print(f"{name} {r}: |r|={np.linalg.norm(r):.4f} L1={np.abs(r).mean():.4f} D2={d2:.4f} ({Fr(d2).limit_denominator(100)})")
print("in-sample D2 for N = d + 1 points: (N-1)^2/N =", (4 - 1) ** 2 / 4)

print("\n== same, through the product's whitened_pca (d=3, global mean) ==")
img = R.T.reshape(3, 1, 4)
keep = np.ones((1, 4), bool)
X = whitened_pca(img, keep, d=3)
print("whitened components\n", X[:, 0, :])
print("residual_mahalanobis(local_window=None):", residual_mahalanobis(img, np.zeros_like(img), keep, d=3, local_window=None))

print("\n== truncation: keep only the top 2 directions ==")
order = np.argsort(ev)[::-1][:2]
for name, r in [("P", np.array([3, 3, 0.])), ("Q", np.array([2, -2, 0.])), ("Rr", np.array([0, 0, 4.]))]:
    z = (r @ U[:, order]) / np.sqrt(ev[order])
    print(name, "top-2 whitened", z, "score", (z ** 2).sum())

print("\n== near-singular 2x2 and the floor ==")
for c in (0.9, 0.99, 0.999):
    A = np.array([[1, c], [c, 1]])
    print(c, "eigs", np.linalg.eigvalsh(A), "inverse", np.linalg.inv(A).round(3).tolist(),
          "D2 of (0.01,-0.01)", np.array([0.01, -0.01]) @ np.linalg.inv(A) @ np.array([0.01, -0.01]))
print("contribution 0.001^2/1e-10 =", 0.001 ** 2 / 1e-10)

print("\n== integral image toy (IW46) ==")
V = np.array([[1, 2, 0, 1], [3, 1, 2, 0], [0, 2, 1, 3], [1, 0, 2, 1]], float)
I = np.pad(V.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
print(I.astype(int))
def box(r0, r1, c0, c1):
    return I[r1 + 1, c1 + 1] - I[r0, c1 + 1] - I[r1 + 1, c0] + I[r0, c0]
print("rows1-2 cols1-2:", box(1, 2, 1, 2), "check", V[1:3, 1:3].sum())
print("rows2-3 cols2-3:", box(2, 3, 2, 3), "check", V[2:4, 2:4].sum())
print("rows0-3 cols0-3:", box(0, 3, 0, 3))
for (pr, pc) in ((1, 1), (2, 2)):
    outer = box(pr - 1, pr + 1, pc - 1, pc + 1); guard = V[pr, pc]
    print(f"ring around ({pr},{pc}): outer {outer} guard {guard} ring {outer - guard} mean {(outer - guard) / 8}")
print("product local_mean_score with window (1,3), d=1:")
s = local_mean_score(V[None], np.ones((4, 4), bool), window=(1, 3))
print(s)
print("expected at (1,1): (1 - 1.375)^2 =", (1 - 1.375) ** 2, "; at (2,2): (1 - 1.375)^2 =", (1 - 1.375) ** 2)
print("note: d=1 so the ring needs >= 2 valid pixels; corner (0,0) ring count = 3")

print("\n== local covariance via integral images (sums of x x^T), toy 2 channels ==")
rng = np.random.default_rng(0)
Y = rng.integers(0, 4, size=(2, 5, 5)).astype(float)
print("channel a\n", Y[0], "\nchannel b\n", Y[1])
win = Y[:, 1:4, 1:4].reshape(2, -1)
print("3x3 box at centre (2,2): sums", win.sum(1), "sum of products aa ab bb",
      (win[0] * win[0]).sum(), (win[0] * win[1]).sum(), (win[1] * win[1]).sum())
n = 9
m = win.sum(1) / n
C = np.array([[(win[i] * win[j]).sum() / n - m[i] * m[j] for j in range(2)] for i in range(2)])
print("mean", m, "cov (divide by n)\n", C, "\nnp.cov bias=True\n", np.cov(win, bias=True))

print("\n== ring sizes and costs ==")
for inner, outer in ((21, 61), (11, 31), (5, 13), (7, 15), (1, 3)):
    print(f"({inner},{outer}): ring = {outer}^2 - {inner}^2 = {outer**2} - {inner**2} = {outer**2 - inner**2}")
for H, W in ((1000, 1000), (1210, 1219)):
    px = H * W
    print(f"{H}x{W} = {px} px; direct ring sums 3280 x 31 x px = {3280 * 31 * px:,}; "
          f"integral 2 boxes x 4 lookups x 31 x px = {8 * 31 * px:,}; ratio {3280 / 8}")
d = 30
print("local covariance channels d(d+1)/2 =", d * (d + 1) // 2, "; plus d sums + 1 count =", d * (d + 1) // 2 + d + 1)
for H, W in ((1210, 1219),):
    ch = d * (d + 1) // 2 + d + 1
    print(f"float64 integral images for {ch} channels padded by 30: {ch} x {H + 61} x {W + 61} x 8 bytes = "
          f"{ch * (H + 61) * (W + 61) * 8:,} bytes")
    print(f"float64 integral images for 31 channels: {31 * (H + 61) * (W + 61) * 8:,} bytes")

print("\n== eleven-way comparison (IW46, F001 epoch 52) ==")
A = dict(A=0.715, Afade=0.756, B=0.709, C=0.849, D=0.865, EC=0.790, FC=0.718, GC=0.845)
for k, v in A.items():
    print(k, v, "minus A", round(v - A["A"], 3), "minus D", round(v - A["D"], 3))
for s1, s2 in ((0.929, 0.921), (0.859, 0.848), (0.787, 0.751), (0.885, 0.876)):
    print("D - C per scene", round(s1 - s2, 3))
print("means of D per scene", np.mean([0.929, 0.859, 0.787, 0.885]), "C", np.mean([0.921, 0.848, 0.751, 0.876]),
      "A", np.mean([0.731, 0.755, 0.677, 0.698]))
print("sign test two-sided, n=4 all one way:", 2 * 0.5 ** 4)
