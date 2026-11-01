"""Book chapter ch:rx - every worked example, checked with numpy / scipy, and the RX score cross-checked
against spectral.rx (the library call GlobalRXDetector and MNFCompressionDetector make).
Run from the repo root:
  C:/av/Scripts/python.exe research/textbook/tools/compute_rx.py > research/textbook/tools/compute_rx_output.txt"""
import json

import numpy as np
import spectral
from scipy import stats

np.set_printoptions(precision=6, suppress=True)

print("== thermal z-score (app/constants/thermal_pixel_consts.json)")
c = json.load(open("app/constants/thermal_pixel_consts.json"))
m, s = c["mean"][0], c["std"][0]
print("mean", m, "std", s)
for T in (71, 61):
    z = (T - m) / s
    print(" T", T, "z", z, "z^2", z * z, "one-sided tail", stats.norm.sf(z))

print("== normal tails")
for t in (1, 2, 3, 3.42):
    print(" t", t, "P(|z|>t)", 2 * stats.norm.sf(t), "P(z>t)", stats.norm.sf(t))
for a in (0.01, 0.001):
    print(" two-sided threshold for", a, stats.norm.isf(a / 2), "squared", stats.norm.isf(a / 2) ** 2)

print("== chi-square quantiles  (rows C, columns false-alarm rate)")
for C in (1, 2, 3, 10, 30, 165):
    print(" C", C, "mean", C, "sd", np.sqrt(2 * C),
          "q0.99", stats.chi2.isf(0.01, C), "q0.999", stats.chi2.isf(0.001, C))
print(" C=2 closed form 2 ln 100", 2 * np.log(100), "2 ln 1000", 2 * np.log(1000))
print(" P(chi2_3 > 13)", stats.chi2.sf(13, 3), "P(chi2_3 > 5)", stats.chi2.sf(5, 3), "P(chi2_1 > 11.7)", stats.chi2.sf(11.7, 1))
print(" pdf C=10 at 8", stats.chi2.pdf(8, 10), "formula", 8**4 * np.exp(-4) / 768)

print("== 3-band worked example (percent reflectance)")
X = np.array([[8, 18, 30], [9, 21, 30], [11, 19, 28], [11, 21, 30], [11, 21, 32]], float)
mu = X.mean(0); D = X - mu
print("mean", mu); print("dev", D)
S = np.cov(X.T); print("cov", S)
print("det", np.linalg.det(S)); Si = np.linalg.inv(S); print("inv*4", Si * 4)
ev, U = np.linalg.eigh(S); print("eig", ev, "2-sqrt2, 2, 2+sqrt2", 2 - np.sqrt(2), 2, 2 + np.sqrt(2))
for x in ([12, 19, 32], [12, 23, 32]):
    d = np.array(x) - mu
    print(" x", x, "d", d, "S^-1 d", Si @ d, "D2", d @ Si @ d, "naive sum z^2", np.sum(d**2 / np.diag(S)),
          "tail chi2_3", stats.chi2.sf(d @ Si @ d, 3))
    W = np.diag(ev ** -0.5) @ U.T
    print("   whitened", W @ d, "|Wd|^2", np.sum((W @ d) ** 2))
# spectral.rx with the 5 background pixels plus the test pixel would change the stats; use background stats
bg = spectral.calc_stats(X[:, None, :])
print(" spectral.rx with the 5-pixel background stats:",
      spectral.rx(np.array([[[12, 19, 32], [12, 23, 32]]], float), background=bg).ravel())
print(" spectral.rx on the 5 pixels themselves:", spectral.rx(X[:, None, :]).ravel())
print(" in-sample D2 of the 5 pixels", [float(d @ Si @ d) for d in D], "sum", sum(float(d @ Si @ d) for d in D))

print("== AD08's 3-band table is rank 1")
A = np.array([[0.10, 0.20, 0.30], [0.12, 0.24, 0.28], [0.08, 0.16, 0.32], [0.10, 0.20, 0.30]])
SA = np.cov(A.T); print("cov", SA, "det", np.linalg.det(SA), "rank", np.linalg.matrix_rank(SA), "eig", np.linalg.eigvalsh(SA))

print("== global mixture example (thermal, assumed populations)")
# half forest 20 C sd 2, half bare ground 40 C sd 2
mix_mean = 0.5 * 20 + 0.5 * 40
mix_var = 0.5 * (4 + 20**2) + 0.5 * (4 + 40**2) - mix_mean**2
print("mixture mean", mix_mean, "var", mix_var, "sd", np.sqrt(mix_var))
print("machinery 46 C: global z", (46 - mix_mean) / np.sqrt(mix_var), "local z", (46 - 40) / 2)
print("forest pixel 26 C: global z", (26 - mix_mean) / np.sqrt(mix_var), "local z", (26 - 20) / 2)

print("== near-singular 2x2")
for rho in (0.999, 0.9995, 0.9999):
    Sg = np.array([[1, rho], [rho, 1]])
    ev = np.linalg.eigvalsh(Sg)
    print(" rho", rho, "eig", ev, "kappa", ev[1] / ev[0], "inv", np.linalg.inv(Sg),
          "D2(0.1,-0.1)", np.array([0.1, -0.1]) @ np.linalg.inv(Sg) @ np.array([0.1, -0.1]),
          "D2(1,1)", np.array([1, 1.]) @ np.linalg.inv(Sg) @ np.array([1, 1.]))
print(" 1-0.999^2", 1 - 0.999**2, "1/that", 1 / (1 - 0.999**2))
for eps in (1e-4, 1e-3, 1e-2):
    Sg = np.array([[1, 0.999], [0.999, 1]]) + eps * np.eye(2)
    ev = np.linalg.eigvalsh(Sg)
    print(" eps", eps, "eig", ev, "kappa", ev[1] / ev[0], "D2(0.1,-0.1)",
          np.array([0.1, -0.1]) @ np.linalg.inv(Sg) @ np.array([0.1, -0.1]))

print("== sampling noise in the small eigenvalue: 200 pixels, true rho 0.999, 5 seeds")
true = np.array([[1, 0.999], [0.999, 1]])
for seed in range(5):
    P = np.random.default_rng(seed).multivariate_normal([0, 0], true, size=200)
    ev = np.linalg.eigvalsh(np.cov(P.T))
    print(" seed", seed, "eig", ev)

print("== float32 rounding of a tiny eigenvalue")
print(" float32 eps", np.finfo(np.float32).eps, "float64 eps", np.finfo(np.float64).eps)
Sg = np.array([[1, 1 - 1e-8], [1 - 1e-8, 1]])
print(" true small eig 1e-8; float64", repr(np.linalg.eigvalsh(Sg).tolist()), "float32", repr(np.linalg.eigvalsh(Sg.astype(np.float32)).tolist()))
print(" float32(1 - 1e-8) == 1.0 ?", np.float32(1 - 1e-8) == np.float32(1.0))

print("== shrinkage (not used by the code): (1-a) S + a (tr S / C) I on the 3-band S, a = 0.1")
a = 0.1
Ssh = (1 - a) * S + a * np.trace(S) / 3 * np.eye(3)
print(Ssh, np.linalg.eigvalsh(Ssh))
