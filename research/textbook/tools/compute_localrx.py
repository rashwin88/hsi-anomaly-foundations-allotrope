"""Book chapter ch:localrx - worked examples, checked against the repo's own code where possible:
batch_mahalanobis (app/detectors/_local_background.py) for local RX, MNFCompressionDetector's private
_estimate_noise_covariance / _compute_mnf_transform for MNF, and local_mean_score / whitened_pca
(app/utils/anomaly_detection/residual_mahalanobis.py) for PCA-RX.
Run from the repo root:
  C:/av/Scripts/python.exe research/textbook/tools/compute_localrx.py > research/textbook/tools/compute_localrx_output.txt"""
import sys
from pathlib import Path

import numpy as np
import spectral
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.detectors._local_background import batch_mahalanobis  # noqa: E402
from app.detectors.mnf_compression_detector import MNFCompressionDetector  # noqa: E402
from app.utils.anomaly_detection.residual_mahalanobis import local_mean_score, whitened_pca  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
cpu = torch.device("cpu")

print("== ring counts")
ring = lambda o, i: (2 * o + 1) ** 2 - (2 * i + 1) ** 2
print("LocalRX defaults half-sizes 25/5:", ring(25, 5), "= 4(o-i)(o+i+1) =", 4 * 20 * 31)
print("course exercise 15/3:", ring(15, 3))
print("corner pixel 25/5:", 26 * 26 - 6 * 6, " edge pixel:", 26 * 51 - 6 * 11)
print("PCA-RX full widths 21/61:", 61 ** 2 - 21 ** 2, " fallback below", 2 * 30)
print("5x5 example outer 2 inner 1:", ring(2, 1), " inner 0:", ring(2, 0))

print("== local RX on a 5x5 single-band grid, via batch_mahalanobis")
g = np.array([[20, 22, 20, 22, 20],
              [22, 25, 25, 25, 22],
              [20, 25, 25, 25, 20],
              [22, 25, 25, 25, 22],
              [20, 22, 20, 22, 20]], float)
yy, xx = np.mgrid[0:5, 0:5]
for inner in (1, 0):
    bg = (np.maximum(abs(yy - 2), abs(xx - 2)) > inner)
    X = g[bg].reshape(1, -1, 1)
    n = np.array([bg.sum()])
    s = batch_mahalanobis(X, n, np.array([[g[2, 2]]]), 1, 1, 0.0, cpu)
    s_reg = batch_mahalanobis(X, n, np.array([[g[2, 2]]]), 1, 1, 1e-4, cpu)
    v = g[bg]
    print(f" inner={inner}: n={bg.sum()} mean={v.mean():.4f} var(N-1)={v.var(ddof=1):.4f} "
          f"score={s[0]:.4f} (reg 1e-4: {s_reg[0]:.4f})")
    print("   sum sq dev", ((v - v.mean()) ** 2).sum())

print("== padding: two pixels, max_bg 4")
Xp = np.zeros((2, 4, 1)); Xp[0, :, 0] = [0.30, 0.32, 0.34, 0.36]; Xp[1, :2, 0] = [0.30, 0.34]
nb = np.array([4, 2])
print(" correct mean pixel 1", (0.30 + 0.34) / 2, "wrong", (0.30 + 0.34) / 4)
print(" scores of x=0.40 (reg 1e-4)", batch_mahalanobis(Xp, nb, np.array([[0.40], [0.40]]), 2, 1, 1e-4, cpu))
print(" check pixel1 by hand: var", np.var([0.30, 0.34], ddof=1), "score", (0.40 - 0.32) ** 2 / (np.var([0.30, 0.34], ddof=1) + 1e-4))

print("== rank of a sample covariance from n points in B dims")
rng = np.random.default_rng(0)
for n in (2, 3, 5, 10, 11, 20):
    P = rng.normal(size=(n, 10))
    print(f" n={n:2d} B=10 rank={np.linalg.matrix_rank(np.cov(P.T))}")

print("== cost arithmetic (PRISMA 1210 x 1219 from docs/lld/local-background.md)")
N = 1210 * 1219
print(" pixels", N)
for B in (10, 177):
    per = 2480 * B * B + B ** 3 / 3
    print(f" B={B}: per-pixel cov {2480 * B * B:,} + solve {B**3/3:,.0f} -> total {N * per:.3e} multiply-adds;"
          f" batch bytes per elem (2*max_bg*B + B^2)*4 = {(2 * 2480 * B + B * B) * 4:,}")
print(" global: one cov from N pixels at B=10:", f"{N * 100:.3e}")

print("== MNF noise estimate from a 2-band strip of 5 pixels")
b1 = np.array([10.0, 10.2, 9.9, 10.1, 10.0]); b2 = np.array([20.0, 20.1, 20.1, 20.0, 20.0])
d1 = np.diff(b1); d2 = np.diff(b2)
print(" diffs", d1, d2)
Sd = np.cov(np.stack([d1, d2]))
print(" cov of diffs", Sd, "noise cov = /2", Sd / 2)
# the repo's own function on the same strip
det = MNFCompressionDetector.__new__(MNFCompressionDetector)
cube = np.stack([b1, b2])[:, None, :]          # (B=2, H=1, W=5)
print(" repo _estimate_noise_covariance", det._estimate_noise_covariance(cube, np.ones((1, 5), bool)))

print("== MNF by hand: Sigma = [[8,2],[2,1.25]], Sigma_n = diag(4, 0.25)")
S = np.array([[8, 2], [2, 1.25]]); Sn = np.diag([4, 0.25]); Ss = S - Sn
print(" signal cov", Ss, "rank", np.linalg.matrix_rank(Ss))
Wn = np.diag(1 / np.sqrt(np.diag(Sn)))
Sw = Wn @ S @ Wn
print(" whitened cov", Sw)
ev, V = np.linalg.eigh(Sw); print(" eig", ev, "V", V)
for k, (lam, v) in enumerate(zip(ev[::-1], V[:, ::-1].T)):
    a = v @ Wn
    print(f"  MNF{k+1}: lambda={lam:.4f} direction(original)={a} signal={a@Ss@a:.4f} noise={a@Sn@a:.4f} SNR={(a@Ss@a)/(a@Sn@a):.4f}")
evp, Up = np.linalg.eigh(S)
for lam, u in zip(evp[::-1], Up[:, ::-1].T):
    print(f"  PCA: lambda={lam:.4f} u={u} signal={u@Ss@u:.4f} noise={u@Sn@u:.4f} SNR={(u@Ss@u)/(u@Sn@u):.4f}")
lam_g, A = __import__("scipy.linalg", fromlist=["eigh"]).eigh(S, Sn)
print(" scipy generalized eigh(S, Sn):", lam_g, A)

print("== MNF in the repo's code on a synthetic cube, and MNF-RX identities")
rng = np.random.default_rng(1)
B, H, W = 12, 50, 50
yy, xx = np.mgrid[0:H, 0:W] / 50.0
abund = np.stack([np.sin(3 * xx) + yy, np.cos(2 * yy) * xx, (xx - 0.5) ** 2])          # smooth spatial fields
E = rng.normal(size=(3, B))                                                             # 3 spectral signatures
signal = np.einsum("khw,kb->bhw", abund, E)
noise_sd = np.linspace(0.05, 0.6, B)                                                   # noisier bands later
noise = rng.normal(size=(B, H, W)) * noise_sd[:, None, None]
cube = signal + noise
mask = np.ones((H, W), bool)
det = MNFCompressionDetector.__new__(MNFCompressionDetector); det._n_components = B
Sn_hat = det._estimate_noise_covariance(cube, mask)
print(" true noise var (first 4)", (noise_sd ** 2)[:4], " estimated diag", np.diag(Sn_hat)[:4])
print(" true noise var (last 2)", (noise_sd ** 2)[-2:], " estimated diag", np.diag(Sn_hat)[-2:])
pixels = cube[:, mask].T
det._compute_mnf_transform(pixels, Sn_hat)
print(" MNF eigenvalues", det._eigenvalues)
Y = (pixels - det._mnf_mean) @ det._mnf_components.T
rx_raw = spectral.rx(pixels[:, None, :]).ravel()
rx_mnf_all = spectral.rx(Y[:, None, :]).ravel()
print(" full-rank MNF-RX equals raw RX: max abs diff", np.abs(rx_raw - rx_mnf_all).max(), "max score", rx_raw.max())
k = 3
Yk = Y[:, :k]
rx_k = spectral.rx(Yk[:, None, :]).ravel()
manual = (Yk ** 2 / np.var(Yk, axis=0, ddof=1)).sum(1)
print(f" k={k}: MNF-RX vs sum y^2/var: max abs diff", np.abs(rx_k - manual).max())
print(" cov of MNF components (should be diag of eigenvalues)", np.round(np.cov(Yk.T), 4))
print(" mean in-sample score k=3:", rx_k.mean(), " expected 3(N-1)/N =", 3 * (len(Yk) - 1) / len(Yk))
evp = np.linalg.eigvalsh(np.cov(pixels.T))[::-1]
print(" PCA eigenvalues of the raw cube", evp)

print("== PCA-RX: local_mean_score on a tiny case (d = 1, window (1, 5) full widths)")
X = np.zeros((1, 7, 7), np.float32); X[0] = 1.0; X[0, 3, 3] = 4.0
keep = np.ones((7, 7), bool)
s = local_mean_score(X, keep, (1, 5))
print(" ring count at centre:", 5 * 5 - 1, " score at centre", s[3, 3], "= (4 - 1)^2 =", 9.0)
print(" whitened_pca output variance check:", np.var(whitened_pca(cube, mask, d=3).reshape(3, -1), axis=1, ddof=1))
