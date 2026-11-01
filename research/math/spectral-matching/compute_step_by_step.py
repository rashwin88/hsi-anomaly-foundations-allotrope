"""Checks for spectral_matching_step_by_step.tex (the slow, step-by-step textbook version of spectral_matching.tex).

Run from the repo root:  PYTHONPATH=. C:/av/Scripts/python.exe research/math/spectral-matching/compute_step_by_step.py
Output is kept in compute_step_by_step_output.txt next to this file.

Two jobs.
1. Re-check every toy worked example in the textbook (3-band spectra, 2x2 covariances), so the hand arithmetic
   printed there is right to the digits shown.
2. Compute the few project quantities that the hand re-derivations need and that compute_numbers.py does not
   print: the lengths of the galvanised-metal and grass/soil spectra (for the SAM-on-a-mixed-pixel formula), and a
   point-source version of the PSF halo split (the 52% / 40% figures), so the textbook can show how the hand version
   compares with the pixel-integrated number in compute_numbers_output.txt.
Every project number the textbook quotes otherwise comes from spectral_matching.tex or compute_numbers_output.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, chi2, ncx2, t as student
from scipy.optimize import brentq


def section(t):
    print(f"\n=== {t} ===")


# ---- toy spectra (chapters 1-3, 22) ----------------------------------------------------------------------------------
section("toy spectra")
g = np.array([0.1, 0.1, 0.5])      # grass
t = np.array([0.3, 0.3, 0.3])      # metal
s = np.array([0.1, 0.2, 0.3])      # soil
def ang(u, v):
    return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
print("t.g", t @ g, "|t|", np.linalg.norm(t), "|g|", np.linalg.norm(g), "cos", t @ g / (np.linalg.norm(t) * np.linalg.norm(g)),
      "angle", ang(t, g))
print("angle(2t, t)", ang(2 * t, t), "angle(s, g)", ang(s, g), "angle(s,t)", ang(s, t), "s.g", s @ g, "|s|", np.linalg.norm(s))
for a in (0.1, 0.2, 0.5, 0.8):
    x = a * t + (1 - a) * g
    th = ang(t, g)
    tan_formula = a * np.linalg.norm(t) * np.sin(np.radians(th)) / ((1 - a) * np.linalg.norm(g) + a * np.linalg.norm(t) * np.cos(np.radians(th)))
    print(f"a={a}: x={x}, angle(x,g)={ang(x, g):.3f}, angle(x,t)={ang(x, t):.3f}, formula angle(x,g)={np.degrees(np.arctan(tan_formula)):.3f}"
          f", x.g={x @ g:.4f} x.t={x @ t:.4f} |x|^2={x @ x:.4f} |x|={np.linalg.norm(x):.5f}")

# ---- 2-band covariance, eigen, whitening (chapters 5-7) --------------------------------------------------------------
section("2-band covariance")
X = np.array([[2, 1], [-2, -1], [1, 2], [-1, -2]], float)
S = X.T @ X / len(X)
print("S (divide by n)", S.tolist(), "S (n-1)", (X.T @ X / 3).tolist())
lam, U = np.linalg.eigh(S)
print("eig", lam, U)
Si = np.linalg.inv(S)
print("S^-1", Si, "x9", Si * 9, "det", np.linalg.det(S))
W = np.array([[1 / 3, 1 / 3], [1, -1]])
print("W S W^T", W @ S @ W.T, "W^T W", W.T @ W)
print("whitened data", (W @ X.T).T)
for r in ([1, -1], [1, 1], [2, 1], [3, 0]):
    r = np.array(r, float)
    print("r", r, "Mahalanobis^2", r @ Si @ r, "Euclid^2", r @ r, "Wr", W @ r)

# ---- projection examples (chapter 9) ---------------------------------------------------------------------------------
section("projections")
r = np.array([3.0, 1.0]); d = np.array([1.0, 1.0])
p = (d @ r) / (d @ d) * d
print("proj (3,1) on (1,1):", p, "rem", r - p)
H = np.array([[1, 0], [1, 1], [0, 1]], float); r3 = np.array([1.0, 2.0, 3.0])
th = np.linalg.solve(H.T @ H, H.T @ r3)
P = H @ np.linalg.inv(H.T @ H) @ H.T
print("H^T H", H.T @ H, "inv x3", np.linalg.inv(H.T @ H) * 3, "H^T r", H.T @ r3, "theta", th, "proj", H @ th,
      "rem", r3 - H @ th, "|proj|^2", (H @ th) @ (H @ th), "|rem|^2", (r3 - H @ th) @ (r3 - H @ th),
      "cos^2", (H @ th) @ (H @ th) / (r3 @ r3), "angle", np.degrees(np.arccos(np.sqrt((H @ th) @ (H @ th) / (r3 @ r3)))))
print("P x9", P * 9, "P^2-P max", np.abs(P @ P - P).max())

# ---- normal table values used (chapters 10-11) -----------------------------------------------------------------------
section("normal table")
for z in (0.0, 0.5, 1.0, 1.28, 1.645, 1.96, 2.0, 2.326, 2.5, 3.0, 3.09, 3.29):
    print(f"z={z}: Phi={norm.cdf(z):.5f}  Q={norm.sf(z):.6f}")
print("Q^-1(1e-3)", norm.isf(1e-3), "Q^-1(0.005)", norm.isf(0.005), "required SNR (Pd 0.5)", norm.isf(1e-3) ** 2,
      "Q^-1(0.9)", norm.isf(0.9), "SNR Pd 0.9", (norm.isf(1e-3) - norm.isf(0.9)) ** 2)
print("density at 0", norm.pdf(0), "at 1", norm.pdf(1), "at 2", norm.pdf(2))
print("sqrt(2/pi)", np.sqrt(2 / np.pi), "sqrt(pi/2)", np.sqrt(np.pi / 2))
for q in (1, 2, 3, 30):
    thr = chi2.isf(1e-3, q)
    lam_nc = brentq(lambda L: ncx2.sf(thr, q, L) - 0.5, 1e-6, 1e4)
    print(f"chi2 q={q}: threshold {thr:.3f}; noncentrality Pd 0.5 {lam_nc:.3f}; mean {q}")
print("chi2_2 threshold = -2 ln 1e-3 =", -2 * np.log(1e-3))
for nu in (3, 5):
    print(f"t_{nu} unit-var 1e-3 threshold {student.isf(1e-3, nu) * np.sqrt((nu - 2) / nu):.3f}")
# toy Gaussian likelihood
xs = np.array([0.9, 1.1, 1.3])
for m in (1.0, 1.1, 1.2):
    print(f"likelihood of {xs} at mean {m}, sd 0.2: {np.prod(norm.pdf(xs, m, 0.2)):.5f} log {np.sum(norm.logpdf(xs, m, 0.2)):.5f}")

# ---- fill estimate, MF, ACE on 3 bands (chapters 18-20) --------------------------------------------------------------
section("3-band fill / MF / ACE")
d = np.array([2.0, 2.0, -2.0]); r = np.array([0.5, 0.3, -0.6]); Sd = np.diag([0.01, 0.01, 0.04]); Sdi = np.linalg.inv(Sd)
num = d @ Sdi @ r; den = d @ Sdi @ d; rr = r @ Sdi @ r
print("d'S^-1 r", num, "d'S^-1 d", den, "a_hat", num / den, "sd", 1 / np.sqrt(den), "T_MF", num / np.sqrt(den),
      "r'S^-1 r", rr, "ACE", num ** 2 / (den * rr), "cos", num / np.sqrt(den * rr), "angle", np.degrees(np.arccos(num / np.sqrt(den * rr))),
      "plain LS", (d @ r) / (d @ d))
for a in (0.15, 0.2, 0.25, 0.2111):
    e = r - a * d
    print(f"  a={a}: weighted SSE {e @ Sdi @ e:.4f}")
print("SNR at a=0.2", 0.04 * den, "sqrt", np.sqrt(0.04 * den), "Pd", norm.sf(norm.isf(1e-3) - np.sqrt(0.04 * den)))
print("a_min", norm.isf(1e-3) / np.sqrt(den))
# whitened version
Wd = np.diag(1 / np.sqrt([0.01, 0.01, 0.04]))
print("whitened d", Wd @ d, "whitened r", Wd @ r)
# GLRT sigma^2 for ACE: N = 3
Nn = 3
s0 = rr / Nn; s1 = (rr - num ** 2 / den) / Nn
print("sigma0^2", s0, "sigma1^2", s1, "GLR", (s0 / s1) ** (Nn / 2), "1/(1-ACE)", 1 / (1 - num ** 2 / (den * rr)))
# scale invariance
print("ACE at 10x r", (d @ Sdi @ (10 * r)) ** 2 / (den * (10 * r) @ Sdi @ (10 * r)))
# clipping bias example
for kap in (0.0, 0.5, 1.0, 2.0):
    print(f"kappa={kap}: bias in sd units {norm.pdf(kap) - kap * norm.sf(kap):.4f}")
# mismatch bias
dt = np.array([-0.4, -0.4, 0.0])
print("mismatch: d'S^-1 dt", d @ Sdi @ dt, "ratio", d @ Sdi @ dt / den, "E a_hat at a=0.2", 0.2 * (1 + d @ Sdi @ dt / den))

# ---- matched subspace detector toy (chapter 21) ---------------------------------------------------------------------
section("MSD toy (white noise)")
r = np.array([1.0, 2.0, 3.0]); J = np.ones(3); h = np.array([1.0, 0.0, 0.0])
PJp = np.eye(3) - np.outer(J, J) / 3
rp = PJp @ r; hp = PJp @ h
num = (hp @ rp) ** 2 / (hp @ hp); den = rp @ rp - num
print("P_J^perp r", rp, "h_perp", hp, "h_perp.r", hp @ rp, "|h_perp|^2", hp @ hp, "num", num, "den", den, "T", num / den * (3 - 1 - 1) / 1)
# gain-offset check
tt = np.array([0.3, 0.3, 0.3]); bb = np.array([0.1, 0.1, 0.5]); bh = np.array([0.12, 0.09, 0.48]); a, gg, c = 0.2, 0.8, 0.02
x = a * (gg * tt + c) + (1 - a) * bb
lhs = x - bh
rhs = a * gg * (tt - bh) + a * (gg - 1) * bh + a * c + (1 - a) * (bb - bh)
print("gain/offset identity lhs", lhs, "rhs", rhs)

# ---- two-level covariance a_min, worked (chapter 19) ----------------------------------------------------------------
section("a_min hand check (mean|e|=0.10, phi=0.006)")
dn = 904.1; B, k = 165, 10
sig = 0.10 * np.sqrt(np.pi / 2)
lam_in = 0.9 * B * sig ** 2 / k; lam_out = 0.1 * B * sig ** 2 / (B - k)
q1 = (1 - 0.006) * dn / lam_in; q2 = 0.006 * dn / lam_out
print("sigma", sig, "sigma^2", sig ** 2, "lam_in", lam_in, "lam_out", lam_out, "in-part", q1, "out-part", q2, "sum", q1 + q2,
      "a_min", np.sqrt(9.5496 / (q1 + q2)), "sd(a_hat)", 1 / np.sqrt(q1 + q2))
print("reality check set1", np.sqrt(0.218 ** 2 - 0.057 ** 2), "/2.18", np.sqrt(0.218 ** 2 - 0.057 ** 2) / 2.18)
print("price of direction 30.59/14.24", 30.59 / 14.24, "sqrt", np.sqrt(30.59 / 14.24))
print("noise: 0.15/354", 0.15 / 354, "/0.077", 0.15 / 354 / 0.077, "0.15/260", 0.15 / 260, "/0.077", 0.15 / 260 / 0.077)

# ---- SAM on the real galvanised-metal mixture (chapter 22) -----------------------------------------------------------
section("SAM galvanised metal over grass/soil: lengths")
try:
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as GRID
    from app.spectral_match.resample import gaussian_resample_to_target
    LIB = Path("data/family_library")
    asd = np.loadtxt(LIB / "wavelengths_asd_nm.txt", comments="#")
    def load(name_part, fam_prefix=None):
        for dd in sorted((LIB / "spectra").iterdir()):
            for f in sorted(dd.iterdir()):
                if (name_part and name_part in f.name) or (fam_prefix and dd.name == fam_prefix):
                    v = np.loadtxt(f, skiprows=1); v[v < -1e30] = np.nan
                    return gaussian_resample_to_target(asd, v, GRID, np.full(len(GRID), 10.0)), f.name
    tm, n1 = load("GalvanizedSheetMetal")
    gr, n2 = load("Cheatgrass")
    so, n3 = load(None, "natural.soil")
    bg = 0.7 * gr + 0.3 * so
    nt, nb = np.linalg.norm(tm), np.linalg.norm(bg)
    cth = tm @ bg / (nt * nb); th = np.degrees(np.arccos(cth))
    print(n1, n2, n3)
    print(f"|t| {nt:.4f} |b| {nb:.4f} ratio {nt / nb:.4f} t.b {tm @ bg:.4f} cos {cth:.5f} angle {th:.3f} sin {np.sin(np.radians(th)):.5f}")
    print(f"mean t {tm.mean():.4f} mean b {bg.mean():.4f}")
    for a in (0.1, 0.2, 0.3, 0.5, 0.85):
        num = a * nt * np.sin(np.radians(th)); den = (1 - a) * nb + a * nt * np.cos(np.radians(th))
        axb = np.degrees(np.arctan(num / den))
        x = a * tm + (1 - a) * bg
        print(f"a={a}: num {num:.4f} den {den:.4f} tan {num / den:.5f} angle(x,b) {axb:.3f} angle(x,t) {th - axb:.3f} "
              f"direct angle(x,t) {ang(x, tm):.3f} small-a approx {np.degrees(a * nt / nb * np.sin(np.radians(th))):.3f}")
except Exception as exc:     # the textbook still builds; the numbers just cannot be refreshed
    print("could not load the library:", exc)

# ---- PSF: point-source hand version of the halo split (chapters 13, 23) ---------------------------------------------
section("PSF halo, point source at pixel centre (hand version)")
sig = 1.4
def frac(lo, hi, c):
    return norm.cdf((hi - c) / sig) - norm.cdf((lo - c) / sig)
print("1-px window centred: Phi(0.5/1.4)", norm.cdf(0.5 / sig), "fraction", frac(-0.5, 0.5, 0), "squared", frac(-0.5, 0.5, 0) ** 2)
rows = {}
for c in (0.5, 1.5, 2.5, 3.5):
    own = frac(0, 4, c); left = frac(-4, 0, c); right = frac(4, 8, c)
    rows[c] = (own, left, right)
    print(f"c={c}: z-values own ({(0 - c) / sig:.4f},{(4 - c) / sig:.4f}) own {own:.4f} left {left:.4f} right {right:.4f} "
          f"Phi((4-c)/s) {norm.cdf((4 - c) / sig):.4f} Phi((0-c)/s) {norm.cdf((0 - c) / sig):.4f} "
          f"Phi((-4-c)/s) {norm.cdf((-4 - c) / sig):.4f} Phi((8-c)/s) {norm.cdf((8 - c) / sig):.4f}")
own_mean = np.mean([rows[c][0] for c in rows]); side_mean = np.mean([rows[c][1] + rows[c][2] for c in rows])
print(f"mean own (1D) {own_mean:.4f}; mean side (1D) {side_mean:.4f}")
print(f"2D own token = {own_mean:.4f}^2 = {own_mean ** 2:.4f}; 2D edge tokens = 2 x {own_mean:.4f} x {side_mean:.4f} = {2 * own_mean * side_mean:.4f}")
# per-position extremes
vals = [rows[cx][0] * (rows[cy][1] + rows[cy][2]) + rows[cy][0] * (rows[cx][1] + rows[cx][2]) for cx in rows for cy in rows]
print("edge-token share per position: min %.3f max %.3f" % (min(vals), max(vals)))
# leakage of the 16-border-pixel predictor, point-source version, centre pixel weight
print("ring/halo: psf_weights numbers are quoted from compute_numbers_output.txt")

# ---- Kantorovich 2-band check (chapter 24) ---------------------------------------------------------------------------
section("gain rotation")
for kappa in (1.02, 1.05, 1.1, 1.2):
    c = 2 * np.sqrt(kappa) / (1 + kappa)
    best = min((ang(np.array([1, kappa]) * dd, dd) for dd in
                (np.array([np.cos(u), np.sin(u)]) for u in np.linspace(0.001, np.pi / 2 - 0.001, 20001))), key=lambda v: -v)
    print(f"kappa {kappa}: sqrt {np.sqrt(kappa):.5f} bound cos {c:.6f} angle {np.degrees(np.arccos(c)):.3f}; brute-force worst 2-band angle {best:.3f}")
k = 1.1
d = np.array([np.sqrt(k), 1.0]); Gd = np.array([1.0, k]) * d
print("worst d", d, "Gd", Gd, "d.Gd", d @ Gd, "|d|^2", d @ d, "|Gd|^2", Gd @ Gd, "cos", d @ Gd / np.sqrt((d @ d) * (Gd @ Gd)))
print("MAE 0.015 / 0.155", 0.015 / 0.155)

# ---- leakage (chapter 23) --------------------------------------------------------------------------------------------
section("leakage")
for l, a in ((0.3, 0.2), (0.1, 0.2), (0.22, 0.05), (0.5, 0.2)):
    print(f"l={l} a={a}: SNR x{(1 - l) ** 2:.4f}; a_hat -> {a * (1 - l) / (1 - l * a):.4f} (ratio {(1 - l) / (1 - l * a):.4f})")
print("l8 = 0.22*0.24/0.40", 0.22 * 0.24 / 0.40, "(0.87/0.78)^2", (0.87 / 0.78) ** 2)
# toy: r = a(1-l)u ; d = (1-la) u with u = (2,2,-2)
u = np.array([2.0, 2.0, -2.0]); a, l = 0.2, 0.3
print("toy leak r", a * (1 - l) * u, "d", (1 - l * a) * u, "a_hat", (u @ (a * (1 - l) * u)) / ((1 - l * a) * (u @ u)))

# ---- calibration, sample sizes (chapters 26-29) ----------------------------------------------------------------------
section("calibration and sample size")
print("p* =", 1 - 0.05 / 10, "exceedances", 0.005 * 10000, "sd", np.sqrt(0.005 * 0.995 / 10000), "rel", np.sqrt(0.005 * 0.995 / 10000) / 0.005)
print("bound alpha + F/N0", 0.05 + 10 / 10000)
z = norm.ppf(0.975) + norm.ppf(0.8)
print("z975", norm.ppf(0.975), "z80", norm.ppf(0.8), "sum", z, "sum^2", z ** 2)
for disc, delta in ((0.2, 0.1), (0.1, 0.1), (0.3, 0.1)):
    print(f"n for disc {disc}, delta {delta}: {z ** 2 * disc / delta ** 2:.1f}")
for n in (30, 100, 300):
    print(f"n={n}: MDE {z * np.sqrt(0.2 / n):.4f}, half-width p=0.5 {1.96 * np.sqrt(0.25 / n):.4f}")
print("implied discordance", (0.019 / 1.96) ** 2 * 909)
print("design effect m=20 rho=0.1", 1 + 19 * 0.1)
print("label noise: eps 0.1, K 10: Acc 0.6 ->", 0.9 * 0.6 + 0.1 / 10, "Acc 0.5 ->", 0.9 * 0.5 + 0.1 / 10)
# cross-entropy toy
for qv in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"q={qv}: expected CE {-0.7 * np.log(qv) - 0.3 * np.log(1 - qv):.5f}")
# label shift toy: generator prior none 0.5 metal 0.25 plastic 0.25 ; real prior none 0.98 metal 0.01 plastic 0.01
pg = np.array([0.5, 0.25, 0.25]); pp = np.array([0.98, 0.01, 0.01]); post_g = np.array([0.2, 0.6, 0.2])
w = post_g * pp / pg
print("label shift: unnormalised", w, "sum", w.sum(), "posterior", w / w.sum())
# contamination
print("contamination 1/(1-0.02)", 1 / (1 - 0.02))
# paired bootstrap toy: 6 objects
A = np.array([1, 1, 0, 1, 0, 1]); Bm = np.array([1, 0, 0, 1, 1, 0])
print("toy objects acc A", A.mean(), "acc B", Bm.mean(), "diff", A.mean() - Bm.mean(), "discordant", np.mean(A != Bm))
rng = np.random.default_rng(1)
diffs = []
for _ in range(20000):
    i = rng.integers(0, 6, 6)
    diffs.append(A[i].mean() - Bm[i].mean())
print("bootstrap 2.5/97.5 pct", np.percentile(diffs, [2.5, 97.5]))

# ---- ROC curve data for the hypothesis-test chapter (figdata/roc_toy.dat) --------------------------------------------
section("toy ROC (H0 N(0,1), H1 N(mu,1))")
from scipy.integrate import quad
OUT = Path(__file__).parent / "figdata"
fpr = np.concatenate([[0.0], np.logspace(-6, np.log10(0.02), 300)])
cols = [fpr]
for mu in (2.0, 3.09):
    tp = norm.sf(norm.isf(np.clip(fpr, 1e-300, 1)) - mu)
    tp[0] = 0.0
    cols.append(tp)
    pa = quad(lambda f: norm.sf(norm.isf(f) - mu), 1e-15, 0.01, limit=200)[0] / 0.01
    print(f"mu={mu}: TPR at FPR 0.001 {norm.sf(norm.isf(0.001) - mu):.4f}, at 0.01 {norm.sf(norm.isf(0.01) - mu):.4f}; pAUC(<=1%) {pa:.4f}")
np.savetxt(OUT / "roc_toy.dat", np.column_stack(cols), header="fpr tpr_mu2 tpr_mu309", comments="", fmt="%.6g")
print("random ROC pAUC(<=1%):", 0.01 / 2)
