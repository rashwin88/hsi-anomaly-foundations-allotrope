"""Checks for the chapter "The namer as built" of spectral_matching_step_by_step.tex (added 2026-10-31, evening).

Run from the repo root:  PYTHONPATH=. C:/av/Scripts/python.exe research/math/spectral-matching/compute_as_built.py
Output is kept in compute_as_built_output.txt next to this file. Uses the namer's own code where it can.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from app.spectral_match.namer.generator import ring_and_noise
from app.spectral_match.namer.model import SpectralNamer

ROOT = Path(__file__).resolve().parents[3]

print("== class score = logsumexp over the class's cosines / tau, planted entry left out ==")
tau = 0.5
cos = torch.tensor([[0.96, 0.90, 0.80, 0.60]])        # glass g1 (planted), glass g2, painted metal, tarpaulin
cls = torch.tensor([0, 0, 1, 2])
m = SpectralNamer(np.linspace(400, 2500, 20).tolist(), [0.2] * 20, [0.1] * 20, compressed_channels=8, token_dim=8,
                  attention_heads=2, hidden=8, embed=4)
with torch.no_grad():
    m.log_tau.fill_(math.log(tau))
    full = m.class_scores(cos, torch.eye(4), cls, 3)
    out = m.class_scores(cos, torch.eye(4), cls, 3, exclude=torch.tensor([0]))
z = (cos / tau)[0].tolist()
print("z = cos / tau:", [round(v, 4) for v in z])
for name, sc in (("g1 kept", full), ("g1 left out", out)):
    p = sc.softmax(1)[0]
    print(f"{name}: S = {[round(float(v), 4) for v in sc[0]]}, P = {[round(float(v), 4) for v in p]}, "
          f"L = {-math.log(float(p[0])):.4f}")
e = [math.exp(v) for v in z]
print("e^z:", [round(v, 4) for v in e])
print("by hand, kept: S_glass = ln(e^1.92 + e^1.80) =", round(math.log(e[0] + e[1]), 4))
q = [v / sum(e[1:]) for v in e[1:]]
print("left out: entry softmax over g2, metal, tarp:", [round(v, 4) for v in q])
w_in = [e[1] / e[1]]
print("gradient dL/dz_j = q_j - 1[j in c] w_j  (left out; glass has only g2, w = 1):",
      [round(q[0] - 1, 4), round(q[1], 4), round(q[2], 4)], "sum", round(q[0] - 1 + q[1] + q[2], 6))
qk = [v / sum(e) for v in e]
wk = [e[0] / (e[0] + e[1]), e[1] / (e[0] + e[1])]
print("kept: q =", [round(v, 4) for v in qk], "w(glass) =", [round(v, 4) for v in wk])
g = [qk[0] - wk[0], qk[1] - wk[1], qk[2], qk[3]]
print("kept gradient:", [round(v, 4) for v in g], "sum", round(sum(g), 6))
print("per-spectrum InfoNCE with g2 as a 'negative' (true = g1):", round(-math.log(qk[0]), 4),
      "; class-level:", round(-math.log(qk[0] + qk[1]), 4))

print("\n== log-uniform fill on [0.05, 1] ==")
for qq in (0.1, 0.25, 0.5):
    print(f"P(a <= {qq}) = ln({qq}/0.05)/ln 20 = {math.log(qq / 0.05):.4f}/{math.log(20):.4f} = {math.log(qq / 0.05) / math.log(20):.4f}")
print("mean of a: (1 - 0.05)/ln 20 =", round(0.95 / math.log(20), 4))
u = torch.rand(1_000_000, generator=torch.Generator().manual_seed(0))
a = torch.exp(torch.log(torch.tensor(0.05)) * (1 - u))
print("sampled mean", round(float(a.mean()), 4), "share <= 0.25", round(float((a <= 0.25).float().mean()), 4))

print("\n== noise sigma from neighbour differences ==")
row = torch.tensor([0.20, 0.23, 0.21, 0.22]).view(1, 1, 1, 4)
_, _, s = ring_and_noise(row, torch.ones_like(row, dtype=torch.bool))
print("row 0.20 0.23 0.21 0.22 -> sigma", round(float(s[0, 0]), 5))
grad = torch.linspace(0.20, 0.29, 10).view(1, 1, 1, 10)
_, _, s2 = ring_and_noise(grad, torch.ones_like(grad, dtype=torch.bool))
print("a noiseless slope 0.20..0.29 (step 0.01) -> sigma", round(float(s2[0, 0]), 5), "= 0.01/sqrt 2 =", round(0.01 / math.sqrt(2), 5))

print("\n== ring: 15x15 minus 5x5 ==")
print("pixels:", 15 * 15 - 5 * 5, "; Chebyshev distances 3..7: ", sum((2 * d + 1) ** 2 - (2 * d - 1) ** 2 for d in range(3, 8)))
for d in range(3, 8):
    print(f"  distance {d}: (2*{d}+1)^2 - (2*{d}-1)^2 = {(2 * d + 1) ** 2 - (2 * d - 1) ** 2} = 8*{d}")

print("\n== pass bars at fill 0.25 ==")
for s_ in ("prisma", "enmap", "aviris_ng"):
    j = json.loads((ROOT / f"research/experiments/spectral-namer/separability_{s_}.json").read_text())["0.25"]
    print(f"{s_}: f = {1 - j['level1_natural_called_natural']:.4f}, bar = {j['level1_manmade_called_manmade'] + 0.1:.4f}")


# ---------------------------------------------------------------------------------------------------------------------
# Added 2026-10-31 (late): the full model and level 1 by matching (chapter "The namer as built", expanded).
import torch.nn.functional as F
from app.foundation_models.components.spectral_reduced import band_widths

print("\n== parameter count as a formula of D (stem width), H (hidden), E (embed) ==")
def towers(D, H, E):
    return 4 * D * H + 3 * H * H + 7 * H + 2 * H * E + 2 * E + 2
def stem_count(D, d=32):
    return 4192 + (32 * 64 + 64) + (64 * D + D) + (d * d + d) + D * d + 4 * (d * d + d) + (d + 1)
for D, H, E in ((64, 256, 128), (64, 128, 64), (32, 256, 128)):
    m_ = SpectralNamer(np.linspace(400, 2500, 30).tolist(), [0.2] * 30, [0.1] * 30, compressed_channels=D, hidden=H, embed=E)
    real = sum(p.numel() for p in m_.parameters())
    tw = towers(D, H, E)
    print(f"D={D} H={H} E={E}: towers 4DH+3H^2+7H+2HE+2E+2 = {tw:,}; stem {stem_count(D):,}; BN {2 * D}; scalars 2;"
          f" formula total {tw + stem_count(D) + 2 * D + 2:,}; code {real:,}")
print("stem(D) = 11,617 + 65 D + 32 D: D=64 ->", 11617 + 97 * 64, "; D=32 ->", 11617 + 97 * 32)

print("\n== level 1 by matching: 3 entries plastic 6.0, metal 5.0 | soil 5.5; man-made truth ==")
mm = SpectralNamer(np.linspace(400, 2500, 20).tolist(), [0.2] * 20, [0.1] * 20, compressed_channels=8, token_dim=32,
                   attention_heads=2, hidden=8, embed=4)
s3 = torch.tensor([[6.0, 5.0, 5.5]], requires_grad=True)
ell = mm.level1_logit(s3); L1 = F.binary_cross_entropy_with_logits(ell, torch.ones(1)); L1.backward()
print("logit", round(float(ell), 4), "p", round(float(torch.sigmoid(ell)), 4), "BCE", round(float(L1), 4),
      "grads", [round(float(v), 4) for v in s3.grad[0]])
print("second natural entry at 5.5: S_nat =", round(5.5 + math.log(2), 4), "logit", round(math.log(math.exp(6) + math.exp(5)) - 5.5 - math.log(2), 4))

print("\n== d logit / d ln tau with cosines fixed: s_j = c_j / tau, ds_j/dln tau = -s_j ==")
c = torch.tensor([6.0, 5.0, 5.5]) * 0.1       # cosines at tau = 0.1 giving scores 6, 5, 5.5
lt = torch.tensor(math.log(0.1), requires_grad=True)
s = c / lt.exp()
l_ = torch.logsumexp(s[:2], 0) - s[2]
l_.backward()
u = torch.softmax(torch.tensor([6.0, 5.0]), 0)
print("autograd d logit/d ln tau =", round(float(lt.grad), 4), "; by hand -(u.s_M - s_N) = -(",
      round(float(u[0] * 6 + u[1] * 5), 4), "- 5.5) =", round(-(float(u[0] * 6 + u[1] * 5) - 5.5), 4))
print("dL1/dln tau = (p - y) * that =", round((float(torch.sigmoid(torch.tensor(0.8133))) - 1) * float(lt.grad), 4))

print("\n== the XOR square and nearest-shelf matching ==")
for deg in (80, 190, 45):
    x = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    cm = [float(x @ np.array([0, 1])), float(x @ np.array([0, -1]))]
    cn = [float(x @ np.array([1, 0])), float(x @ np.array([-1, 0]))]
    lg = math.log(sum(math.exp(v / 0.1) for v in cm)) - math.log(sum(math.exp(v / 0.1) for v in cn))
    print(f"pixel at {deg} deg: man-made cos {[round(v, 4) for v in cm]}, natural cos {[round(v, 4) for v in cn]}, logit {lg:.4f}")

print("\n== the hash split as a coin of 0.2 per spectrum: held-out count per class ~ Binomial(n, 0.2) ==")
for name, n_, held in (("oil", 40, 8), ("plastic", 46, 8), ("textile", 14, 3), ("wood/paper", 11, 2), ("paint", 4, 1),
                       ("metal", 10, 3), ("concrete", 34, 9)):
    mean, sd = 0.2 * n_, math.sqrt(n_ * 0.2 * 0.8)
    print(f"  {name:10s} n={n_:2d}: mean {mean:.1f}, sd {sd:.2f}, observed {held} ({(held - mean) / sd:+.2f} sd)")
print("  paint: P(no paint spectrum held out) = 0.8^4 =", round(0.8 ** 4, 4), "; P(all 4 held out) = 0.2^4 =", 0.2 ** 4)
print("  total: mean 0.2*159 =", 0.2 * 159, "sd", round(math.sqrt(159 * 0.16), 2), "observed 34")

print("\n== fill: variance ==")
Ea = 0.95 / math.log(20); Ea2 = (1 - 0.05 ** 2) / (2 * math.log(20))
print("E[a] =", round(Ea, 4), "E[a^2] = (1 - a_min^2)/(2 ln 20) =", round(Ea2, 4), "Var =", round(Ea2 - Ea ** 2, 4),
      "sd =", round(math.sqrt(Ea2 - Ea ** 2), 4))
u = torch.rand(1_000_000, generator=torch.Generator().manual_seed(0))
a = torch.exp(torch.log(torch.tensor(0.05)) * (1 - u))
print("sampled E[a^2]", round(float((a ** 2).mean()), 4), "sd", round(float(a.std()), 4))

print("\n== the contrast in x - r' ==")
b = np.array([0.10, 0.30, 0.25]); t = np.array([0.40, 0.35, 0.05]); r = np.array([0.11, 0.29, 0.25])
aa = 0.25; g = np.array([1.02, 1.00, 0.98])
x = g * (aa * t + (1 - aa) * b); rp = g * r
print("x - r' =", np.round(x - rp, 5), "; g*(a(t-b) + (b-r)) =", np.round(g * (aa * (t - b) + (b - r)), 5))

print("\n== band widths as quadrature: width-weighted mean of a line v = lambda/1000 over 410..470 ==")
wl4 = torch.tensor([410.0, 430.0, 445.0, 470.0]); w = band_widths(wl4)
v = wl4 / 1000
print("widths", w.tolist(), "; weighted mean", round(float((w * v).sum() / w.sum()), 5), "; plain mean", round(float(v.mean()), 5))

print("\n== learning rate schedule: lambda(s) ==")
for s_ in (0, 249, 499, 10000, 15000, 19999):
    lam = min(1.0, (s_ + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s_ / 20000)))
    print(f"  s={s_}: {lam:.5f}")
