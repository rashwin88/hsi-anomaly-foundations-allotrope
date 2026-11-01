"""Numbers and figure data for spectral_matching.tex (research/math/spectral-matching/).

Run from the repo root:  PYTHONPATH=. C:/av/Scripts/python.exe research/math/spectral-matching/compute_numbers.py
Writes figdata/*.dat (pgfplots tables) and prints every number the note quotes, so each one can be traced.
Everything here is computed from the shipped family library (data/family_library, 411 spectra), the 165-band
common grid, the 165-grid reflectance std curve (app/constants/hyperspectral.json) and stated assumptions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import norm

from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as GRID
from app.spectral_match.resample import gaussian_resample_to_target

HERE = Path(__file__).parent
OUT = HERE / "figdata"
OUT.mkdir(exist_ok=True)
LIB = Path("data/family_library")


def load_library():
    asd = np.loadtxt(LIB / "wavelengths_asd_nm.txt", comments="#")
    idx = json.loads((LIB / "index.json").read_text())
    ents = []
    for d in sorted((LIB / "spectra").iterdir()):
        for f in sorted(d.iterdir()):
            v = np.loadtxt(f, skiprows=1)
            v[v < -1e30] = np.nan
            ents.append((d.name, f.name, v))
    return asd, ents


def resample(asd, v, wl, fwhm):
    return gaussian_resample_to_target(asd, v, wl, np.full(len(wl), fwhm))


def sam_deg(x, Y):
    ok = np.isfinite(x) & np.all(np.isfinite(Y), 0)
    x, Y = x[ok], Y[:, ok]
    c = Y @ x / (np.linalg.norm(Y, axis=1) * np.linalg.norm(x))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def section(t):
    print(f"\n=== {t} ===")


asd, ents = load_library()
fam = np.array([e[0] for e in ents])
names = np.array([e[1] for e in ents])
R = np.stack([resample(asd, e[2], GRID, 10.0) for e in ents])          # (411, 165)
good = np.all(np.isfinite(R), 1)
section("library")
print("entries", len(ents), "on 165 grid fully finite", good.sum(), "grid bands", len(GRID))
R, fam, names = R[good], fam[good], names[good]
std = np.array(json.loads(Path("app/constants/hyperspectral.json").read_text())["std"])
print("165-grid std: min %.4f median %.4f max %.4f" % (std.min(), np.median(std), std.max()))

# ---- 1. SAM on a mixed pixel ---------------------------------------------------------------------------------------
section("SAM on mixed pixels")
pick_t = [i for i, n in enumerate(names) if "GalvanizedSheetMetal" in n][0]
pick_b = [i for i, n in enumerate(names) if "Cheatgrass" in n][0]
pick_s = [i for i, n in enumerate(names) if fam[i] == "natural.soil"][0]
t, g, s = R[pick_t], R[pick_b], R[pick_s]
bg = 0.7 * g + 0.3 * s                                          # a grass/soil background
print("target", names[pick_t], "| grass", names[pick_b], "| soil", names[pick_s])
rows = []
switch = None
for a in np.linspace(0, 1, 41):
    x = a * t + (1 - a) * bg
    ang = sam_deg(x, R)
    best = fam[np.argmin(ang)]
    th_t = sam_deg(x, t[None])[0]
    th_b = sam_deg(x, bg[None])[0]
    mm = [j for j in range(len(R)) if fam[j].startswith("manmade")]
    nat = [j for j in range(len(R)) if fam[j].startswith("natural")]
    rows.append((a, th_t, th_b, ang[mm].min(), ang[nat].min(), int(best.startswith("manmade"))))
    if switch is None and fam[np.argmin(ang)] == fam[pick_t]:
        switch = a
np.savetxt(OUT / "sam_mixing.dat", np.array(rows), header="a theta_t theta_b best_manmade best_natural is_mm",
           comments="", fmt="%.5f")
print("angle(t, bg) = %.2f deg" % sam_deg(t, bg[None])[0])
print("first fill where SAM's nearest entry is in the target's family:", switch)
for a in (0.1, 0.2, 0.3, 0.5):
    x = a * t + (1 - a) * bg
    ang = sam_deg(x, R)
    print(f"a={a}: nearest family {fam[np.argmin(ang)]} ({ang.min():.2f} deg); angle to t {sam_deg(x, t[None])[0]:.2f}")
d_refl = t - bg
print("|t - b| mean abs reflectance %.4f; normalised mean abs %.3f" % (np.abs(d_refl).mean(), np.abs(d_refl / std).mean()))

# ---- 2. Family subspaces and principal angles ----------------------------------------------------------------------
section("family subspaces")
def basis(X, k):
    U, S, _ = np.linalg.svd(X.T, full_matrices=False)
    return U[:, :k], S
def principal_angles(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))
Z = R / std                                                     # normalised units, like the model's
mm_fams = sorted({f for f in fam if f.startswith("manmade")})
nat_rows = np.flatnonzero(np.char.startswith(fam.astype(str), "natural"))
for f in mm_fams:
    X = Z[fam == f]
    _, S = basis(X, 1)
    e = S ** 2 / (S ** 2).sum()
    k90 = int(np.searchsorted(np.cumsum(e), 0.99) + 1)
    print(f"{f:22s} n={len(X):3d}  rank for 99% energy {k90:2d}  top-1 share {e[0]:.3f}")
# background subspace proxy: top-10 PCs of natural spectra (centred), as in implant.background_subspace (k = 10)
Nc = Z[nat_rows] - Z[nat_rows].mean(0)
Ub, _ = basis(Nc, 10)
Pb = Ub @ Ub.T
rows = []
pairs = [("manmade.metal", "manmade.roofing"), ("manmade.plastic", "manmade.textile"),
         ("manmade.metal", "manmade.plastic"), ("manmade.pavement", "manmade.building"),
         ("manmade.paint", "manmade.plastic")]
for fa, fb in pairs:
    A, _ = basis(Z[fam == fa], 3)
    B, _ = basis(Z[fam == fb], 3)
    a0 = principal_angles(A, B)[0]
    Ap, _ = basis(((np.eye(len(GRID)) - Pb) @ Z[fam == fa].T).T, 3)
    Bp, _ = basis(((np.eye(len(GRID)) - Pb) @ Z[fam == fb].T).T, 3)
    a1 = principal_angles(Ap, Bp)[0]
    print(f"smallest principal angle {fa} vs {fb} (rank 3): {a0:.2f} deg; after projecting out bg-10: {a1:.2f} deg")
# foreign fraction of family directions (t - mean natural) with respect to the natural top-10
mu_nat = Z[nat_rows].mean(0)
ff = {}
for f in mm_fams:
    D = Z[fam == f] - mu_nat
    ff[f] = np.median(1 - ((D @ Ub) ** 2).sum(1) / (D ** 2).sum(1))
    print(f"foreign fraction of ({f} - natural mean) vs natural top-10: median {ff[f]:.3f}")

# ---- 3. Detection SNR and minimum detectable fill ------------------------------------------------------------------
section("detection SNR")
B, k = len(GRID), 10
pfa, pd = 1e-3, 0.5
eta = norm.isf(pfa)
need = (eta + norm.ppf(pd)) ** 2
print(f"known-direction MF: Pfa {pfa}, Pd {pd}: required SNR = {need:.2f}")
dn = np.linalg.norm(d_refl / std) ** 2                       # ||d||^2 normalised
print(f"||d||^2 (normalised, galvanised metal vs grass/soil) = {dn:.1f}")
rows = []
for m_abs in (0.05, 0.10, 0.14):
    sig = m_abs * np.sqrt(np.pi / 2)                          # Gaussian: mean |e| = sigma sqrt(2/pi)
    for eps in (0.05, 0.1, 0.2):
        lam_in = (1 - eps) * B * sig ** 2 / k
        lam_out = eps * B * sig ** 2 / (B - k)
        for phi in (0.006, 0.05, 0.2, 1.0):
            q = (1 - phi) * dn / lam_in + phi * dn / lam_out
            amin = np.sqrt(need / q)
            if eps == 0.1:
                print(f"mean|e|={m_abs:.2f} sigma={sig:.3f} eps={eps} phi={phi:<5}: d'Sd={q:9.1f}  a_min={amin:.4f}")
phis = np.logspace(-3, 0, 61)
for m_abs in (0.05, 0.10, 0.14):
    sig = m_abs * np.sqrt(np.pi / 2)
    lam_in, lam_out = 0.9 * B * sig ** 2 / k, 0.1 * B * sig ** 2 / (B - k)
    q = (1 - phis) * dn / lam_in + phis * dn / lam_out
    rows.append(np.sqrt(need / q))
np.savetxt(OUT / "amin_phi.dat", np.column_stack([phis] + rows), header="phi a005 a010 a014", comments="", fmt="%.6g")
# heavy tail: Student-t null instead of Gaussian (nu = 3, scaled to unit variance)
from scipy.stats import t as student
for nu in (3, 5):
    eta_t = student.isf(pfa, nu) * np.sqrt((nu - 2) / nu)
    print(f"Pfa {pfa} threshold, unit-variance t_{nu}: {eta_t:.2f} (Gaussian {eta:.2f}); SNR need x{((eta_t)/eta)**2:.2f}")

# ---- 4. PSF halo through token masking -----------------------------------------------------------------------------
section("PSF halo")
def pixel_psf(sigma, cx, cy, n=40):
    """energy of a unit point at (cx, cy) (pixel units) landing in each integer pixel of an n x n grid"""
    e = np.arange(n + 1)
    fx = 0.5 * (erf((e[1:] - cx) / (sigma * np.sqrt(2))) - erf((e[:-1] - cx) / (sigma * np.sqrt(2))))
    fy = 0.5 * (erf((e[1:] - cy) / (sigma * np.sqrt(2))) - erf((e[:-1] - cy) / (sigma * np.sqrt(2))))
    return np.outer(fy, fx)
sig = 1.4
n = 48
for cell in (4, 8):
    own, edge_nb, central = [], [], []
    for py in range(cell):
        for px in range(cell):
            # target filling one pixel: integrate the point over the pixel (5 x 5 sub-samples)
            E = np.zeros((n, n))
            for sy in np.linspace(0.1, 0.9, 5):
                for sx in np.linspace(0.1, 0.9, 5):
                    E += pixel_psf(sig, 2 * cell + px + sx, 2 * cell + py + sy, n) / 25
            ty, tx = 2 * cell, 2 * cell
            o = E[ty:ty + cell, tx:tx + cell].sum()
            nb = sum(E[ty + dy * cell: ty + (dy + 1) * cell, tx + dx * cell: tx + (dx + 1) * cell].sum()
                     for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            own.append(o); edge_nb.append(nb); central.append(E[ty + py, tx + px])
    print(f"cell {cell} px: centre-pixel energy {np.mean(central):.3f}; own cell {np.mean(own):.3f}; "
          f"4 edge-neighbour cells (visible in a checkerboard) {np.mean(edge_nb):.3f} "
          f"(min {np.min(edge_nb):.3f}, max {np.max(edge_nb):.3f})")
# halo/object weight for a 3x3 object under implant.psf_weights normalisation
from scipy import ndimage
obj = np.zeros((31, 31), bool); obj[14:17, 14:17] = True
w = ndimage.gaussian_filter(obj.astype(float), sig, mode="constant"); w /= w[obj].mean()
h1 = ndimage.binary_dilation(obj, iterations=1) & ~obj
print("3x3 object, psf_weights: mean weight on 1-px halo %.3f (object 1.0)" % w[h1].mean())
for size in (1, 2, 4):
    obj = np.zeros((31, 31), bool); obj[14:14 + size, 14:14 + size] = True
    w = ndimage.gaussian_filter(obj.astype(float), sig, mode="constant"); w /= w[obj].mean()
    h1 = ndimage.binary_dilation(obj, iterations=1) & ~obj
    ring = ndimage.binary_dilation(obj, iterations=3) & ~ndimage.binary_dilation(obj, iterations=1)
    print(f"{size}x{size} object: halo weight {w[h1].mean():.3f}; ring (2-3 px) weight {w[ring].mean():.3f}")

# ---- 5. Leakage ----------------------------------------------------------------------------------------------------
section("leakage")
rows = []
for l in np.linspace(0, 0.95, 39):
    rows.append((l, (1 - l) ** 2, (1 - l) / (1 - 0.2 * l), (1 - l) / (1 - 0.05 * l)))
np.savetxt(OUT / "leakage.dat", np.array(rows), header="l snr_ratio bias_a02 bias_a005", comments="", fmt="%.5f")
for l in (0.1, 0.3, 0.5):
    print(f"l={l}: SNR x{(1-l)**2:.2f}; a_min x{1/(1-l):.2f}")

# ---- 6. Gain error (Kantorovich) and SRF errors --------------------------------------------------------------------
section("gain and SRF")
for kappa in (1.02, 1.05, 1.1, 1.2):
    c = 2 * np.sqrt(kappa) / (1 + kappa)
    print(f"kappa={kappa}: cos >= {c:.6f}, angle <= {np.degrees(np.arccos(c)):.2f} deg")
# empirical: smooth gain ripple +/-5% on the library differences
rng = np.random.default_rng(0)
x01 = (GRID - GRID.min()) / (GRID.max() - GRID.min())
angs = []
for _ in range(200):
    c = rng.normal(0, 1, 4)
    gcurve = 1 + 0.05 * np.tanh(np.polynomial.legendre.legval(2 * x01 - 1, c) / 2)
    j = rng.integers(len(R))
    dd = R[j] - bg
    angs.append(np.degrees(np.arccos(np.clip((gcurve * dd) @ dd / (np.linalg.norm(gcurve * dd) * np.linalg.norm(dd)), -1, 1))))
print("smooth +/-5%% gain on (t - b): angle median %.2f deg, 95th pct %.2f" % (np.median(angs), np.percentile(angs, 95)))
for shift in (1.0, 2.0, 5.0):
    Rs = np.stack([resample(asd, e[2], GRID + shift, 10.0) for e in ents])[good]
    ang = np.array([sam_deg(Rs[j], R[j][None])[0] for j in range(len(R))])
    rel = np.nanmax(np.abs(Rs - R) / np.maximum(np.abs(R), 1e-3), 1)
    print(f"SRF centre shift {shift} nm: SAM(shifted, nominal) median {np.median(ang):.3f} deg, max {ang.max():.3f}; "
          f"median per-entry max rel change {np.median(rel):.3f}")
for fw in (8.0, 12.0, 15.0):
    Rs = np.stack([resample(asd, e[2], GRID, fw) for e in ents])[good]
    ang = np.array([sam_deg(Rs[j], R[j][None])[0] for j in range(len(R))])
    print(f"FWHM {fw} nm instead of 10: SAM median {np.median(ang):.3f} deg, max {ang.max():.3f}")

# ---- 7. Benchmark sample sizes -------------------------------------------------------------------------------------
section("sample size")
z = norm.ppf(0.975) + norm.ppf(0.8)
for p in (0.3, 0.5):
    for disc in (0.1, 0.2, 0.3):
        for delta in (0.05, 0.10, 0.15):
            nreq = z ** 2 * disc / delta ** 2              # paired (McNemar) normal approximation
            if p == 0.3:
                print(f"discordance {disc}, delta {delta}: n objects = {nreq:.0f}")
for nobj in (30, 100, 300):
    for disc in (0.2,):
        mde = z * np.sqrt(disc / nobj)
        print(f"n={nobj}, discordance {disc}: MDE {mde:.3f}")
    print(f"n={nobj}: 95% CI half-width on one accuracy p=0.5: {1.96*np.sqrt(0.25/nobj):.3f}")

# ---- 8. Price of not knowing the direction: chi-square thresholds and noncentrality ---------------------------------
section("chi-square: any direction vs known subspace")
from scipy.stats import chi2, ncx2, beta as betad
from scipy.optimize import brentq
for q in (1, 3, 10, 30):
    thr = chi2.isf(pfa, q)
    lam = brentq(lambda L: ncx2.sf(thr, q, L) - pd, 1e-6, 1e4)
    print(f"q={q:2d}: chi2 threshold at Pfa {pfa} = {thr:6.2f}; noncentrality for Pd {pd}: {lam:6.2f}")
for N, q in ((165, 1), (165, 3), (234, 3)):
    print(f"Gaussian null of cos^2 to a fixed {q}-dim subspace in {N} whitened dims: mean {q/N:.4f}, "
          f"99.5th pct {betad.ppf(0.995, q/2, (N-q)/2):.4f}")

# ---- 9. Precision of the empirical null threshold ------------------------------------------------------------------
section("empirical null threshold")
for Nn in (10_000, 100_000):
    for q in (0.005, 0.001):
        sd = np.sqrt(q * (1 - q) / Nn)
        print(f"N_null={Nn}, tail level {q}: exceedances {q*Nn:.0f}, sd of realised FPR {sd:.5f} ({sd/q:.0%} relative)")

# ---- 10. Leakage of a linear interpolating predictor ---------------------------------------------------------------
section("leakage of a linear predictor")
cell = 4
res = {}
for name, pick in (("16 border pixels", "border"), ("all 64 px of the 4 edge-neighbour tokens", "tokens")):
    ls = []
    for py in range(cell):
        for px in range(cell):
            E = np.zeros((n, n))
            for sy in np.linspace(0.1, 0.9, 5):
                for sx in np.linspace(0.1, 0.9, 5):
                    E += pixel_psf(sig, 2 * cell + px + sx, 2 * cell + py + sy, n) / 25
            ty = tx = 2 * cell
            m = np.zeros((n, n), bool)
            if pick == "border":
                m[ty - 1, tx:tx + cell] = m[ty + cell, tx:tx + cell] = True
                m[ty:ty + cell, tx - 1] = m[ty:ty + cell, tx + cell] = True
            else:
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    m[ty + dy * cell: ty + (dy + 1) * cell, tx + dx * cell: tx + (dx + 1) * cell] = True
            ls.append(E[m].mean() / E[ty + py, tx + px])
    print(f"linear predictor = mean of {name}: leakage l = mean {np.mean(ls):.3f} (range {np.min(ls):.3f}-{np.max(ls):.3f})")
# paired-bootstrap consistency: FTH1 vs FTB1 caught top 1% interval half-width ~1.9 pts on 909 objects
hw = 0.019
print("implied discordance from a +-1.9 pt half-width on 909 objects: %.3f" % ((hw / 1.96) ** 2 * 909))
