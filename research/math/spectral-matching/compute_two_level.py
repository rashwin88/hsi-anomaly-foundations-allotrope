"""Checks for Part VII of spectral_matching_step_by_step.tex (the two-level namer and the matching model, added 2026-10-31).

Run from the repo root:  C:/av/Scripts/python.exe research/math/spectral-matching/compute_two_level.py
Output is kept in compute_two_level_output.txt next to this file.

Every toy worked example in Part VII is re-done here to the digits the book prints, and the separability numbers are
read (never written) from research/experiments/spectral-namer/separability_<sensor>.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SEP = REPO / "research" / "experiments" / "spectral-namer"


def section(t):
    print(f"\n=== {t} ===")


# ---- ch 31: three-band ACE with a diagonal whitening -----------------------------------------------------------------
section("ACE on three bands, ring = grass, whitening sd (0.02, 0.04, 0.01)")
g = np.array([0.05, 0.40, 0.25])        # grass (ring mean)
r = np.array([0.30, 0.32, 0.35])        # roof
s = np.array([0.20, 0.25, 0.30])        # soil
sd = np.array([0.02, 0.04, 0.01])


def ace(d, t):
    return float((d @ t) ** 2 / ((t @ t) * (d @ d)))


roof_t, soil_t = (r - g) / sd, (s - g) / sd
print("whitened roof direction", roof_t, " |.|^2", roof_t @ roof_t)
print("whitened soil direction", soil_t, " |.|^2", soil_t @ soil_t)
noise = np.array([1.0, 1.0, -1.0])     # one sd per band, already whitened
for a in (0.2, 0.5):
    d = a * (r - g) / sd
    print(f"fill {a}: d~ = {d}, |d~|^2 = {d @ d:.4f}")
    print(f"   no noise : ACE roof {ace(d, roof_t):.4f}, soil {ace(d, soil_t):.4f}, d.soil {d @ soil_t:.4f}")
    dn = d + noise
    print(f"   + noise  : d~ = {dn}, |d~|^2 = {dn @ dn:.4f}, d.roof {dn @ roof_t:.4f}, d.soil {dn @ soil_t:.4f}")
    print(f"              ACE roof {ace(dn, roof_t):.4f}, soil {ace(dn, soil_t):.4f}")

# ---- ch 31: the hierarchical probability ---------------------------------------------------------------------------
section("two-level probabilities")
p_mm = 0.9
cond = {"painted metal": 0.6, "bare metal": 0.25, "tarp": 0.15}
for k, v in cond.items():
    print(f"  P({k}) = {p_mm} x {v} = {p_mm * v:.4f}")
print("  P(natural) =", round(1 - p_mm, 4), " sum =", round(1 - p_mm + sum(p_mm * v for v in cond.values()), 4))
for p_mm2, top in ((0.3, 0.6), (0.9, 0.4), (0.9, 0.6)):
    print(f"  P(mm)={p_mm2}, top P(class|mm)={top}: product {p_mm2 * top:.3f}")

# ---- ch 31: the separability numbers ---------------------------------------------------------------------------------
section("separability (read-only)")
for sensor in ("prisma", "enmap", "aviris_ng"):
    f = SEP / f"separability_{sensor}.json"
    if not f.exists():
        print(f"  {sensor}: not present")
        continue
    d = json.loads(f.read_text())
    for fill, v in d.items():
        prod = v["level1_manmade_called_manmade"] * v["level2_correct_given_level1_right"]
        print(f"  {sensor} fill {fill:>4}: n {v['n']}, L1 mm->mm {v['level1_manmade_called_manmade']:.4f}, "
              f"nat->nat {v['level1_natural_called_natural']:.4f}, L2 {v['level2_correct_family']:.4f}, "
              f"L2|L1 {v['level2_correct_given_level1_right']:.4f}, product {prod:.4f}")
    rows = d["1.0"]["confusion_manmade_rows"]
    nat = {"mineral", "npv", "rock", "snow_ice", "soil", "vegetation", "water"}
    for fam in ("glass", "building", "pavement", "oil"):
        row = rows[fam]
        tot = sum(row.values())
        print(f"    fill 1.0 {fam}: {row[fam]} of {tot} right, {sum(row[n] for n in nat)} called natural, "
              f"top {sorted(row.items(), key=lambda x: -x[1])[:3]}")
    rows = d["0.25"]["confusion_manmade_rows"]
    for fam, row in rows.items():
        tot = sum(row.values())
        print(f"    fill 0.25 {fam}: {row[fam]}/{tot} = {row[fam] / tot:.3f} right; "
              f"top others {sorted(((k, c) for k, c in row.items() if k != fam), key=lambda x: -x[1])[:3]}")

# ---- ch 32: sizes ----------------------------------------------------------------------------------------------------
section("what fits in a pixel")
car = 4.5 * 1.8
for gsd in (30, 5):
    print(f"  car {car:.2f} m^2 in a {gsd} m pixel ({gsd * gsd} m^2): fill {car / gsd ** 2:.4f}")
for deg in (0.51, 5.92):
    print(f"  {deg} deg: cos {math.cos(math.radians(deg)):.6f}, 1-cos {1 - math.cos(math.radians(deg)):.6f}, "
          f"sin {math.sin(math.radians(deg)):.5f}")

# ---- ch 32: closed classifier vs matching ----------------------------------------------------------------------------
section("matching in a 2-D embedding")
f = np.array([0.8, 0.6])
lib = {"painted metal": np.array([1.0, 0.0]), "tarp": np.array([0.0, 1.0]), "glass (new)": np.array([0.6, 0.8])}
for k, v in lib.items():
    print(f"  cos(f, {k}) = {f @ v / (np.linalg.norm(f) * np.linalg.norm(v)):.4f}")
W = np.array([[1.0, 0.0], [0.0, 1.0]])     # a closed classifier's two rows
print("  classifier logits", W @ f)

# ---- ch 33: the contrastive (InfoNCE) loss ---------------------------------------------------------------------------
section("InfoNCE on three candidates")
cosines = np.array([0.8, 0.6, 0.96])       # painted metal, tarp, glass; the true one is glass (index 2)
for tau in (1.0, 0.5, 0.1):
    z = cosines / tau
    e = np.exp(z)
    p = e / e.sum()
    print(f"  tau {tau}: logits {np.round(z, 4)}, exp {np.round(e, 4)}, sum {e.sum():.4f}, p {np.round(p, 4)}, "
          f"loss {-math.log(p[2]):.4f}")
z = cosines / 0.5
p = np.exp(z) / np.exp(z).sum()
print("  gradient p - y at tau 0.5:", np.round(p - np.array([0, 0, 1]), 4))
# two pixels, two spectra: the batch version
C = np.array([[0.9, 0.2], [0.3, 0.7]])
for tau in (0.5,):
    L = []
    for i in range(2):
        e = np.exp(C[i] / tau)
        L.append(-math.log(e[i] / e.sum()))
        print(f"  batch row {i}: exp {np.round(e, 4)}, sum {e.sum():.4f}, p_true {e[i] / e.sum():.4f}, loss {L[-1]:.4f}")
    print(f"  batch mean loss {np.mean(L):.4f}")

# ---- ch 34: open-set level 1 -----------------------------------------------------------------------------------------
section("open-set level 1")
nat_scores = np.array([0.4, 1.1, 0.7, 2.3, 0.2, 0.9, 1.6, 0.5, 3.1, 1.3])
print("  sorted natural scores", np.sort(nat_scores))
q = np.percentile(nat_scores, 90)
print(f"  90th percentile (numpy linear) {q:.4f}; flagged natural {int((nat_scores > q).sum())} of 10")
pos = 0.9 * (10 - 1)
print(f"  pos {pos}, between {np.sort(nat_scores)[8]} and {np.sort(nat_scores)[9]}")
# nature on a line x + y = 1 (a stand-in for the natural manifold); one library man-made spectrum at (0.9, 0.9)
nat_pts = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
mm_lib = np.array([0.9, 0.9])
camo = np.array([0.4, 0.4])
nat_c = nat_pts.mean(0)
print("  natural centroid", nat_c)
print(f"  camouflage to natural centroid {np.linalg.norm(camo - nat_c):.4f}, to man-made library {np.linalg.norm(camo - mm_lib):.4f}")
print("  natural points to their own centroid", np.round(np.linalg.norm(nat_pts - nat_c, axis=1), 4))
print(f"  explanation residual |x + y - 1|: camouflage {abs(camo.sum() - 1):.4f}, natural points "
      f"{np.round(np.abs(nat_pts.sum(1) - 1), 4)}, library man-made {abs(mm_lib.sum() - 1):.4f}")
