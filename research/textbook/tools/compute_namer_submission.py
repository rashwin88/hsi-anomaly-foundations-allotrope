"""Numbers for chapters 42 (the spectral namer as built) and 43 (producing a submission), and for anomaly-detection
course parts 27-28 / Indradhanu part 71. Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_namer_submission.py > research/textbook/tools/compute_namer_submission_output.txt
Uses the real code wherever it can (ring_and_noise, smooth_gain, class_scores, prisma_utm_grid, SpectralNamer)."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from app.spectral_match.namer.generator import ring_and_noise
from app.spectral_match.namer.model import SpectralNamer

ROOT = Path(__file__).resolve().parents[3]

print("== ring mean: 15x15 box minus 5x5 box ==")
print("ring pixels in the interior: 15*15 - 5*5 =", 15 * 15 - 5 * 5)
tile = torch.full((1, 1, 21, 21), 0.20)
tile[0, 0, 9:12, 9:12] = 0.35                         # the target's 3x3 halo
tile[0, 0, 10, 10] = 0.50                             # the target pixel
ring, count, sigma = ring_and_noise(tile, torch.ones_like(tile, dtype=torch.bool))
big = 216 * 0.20 + 8 * 0.35 + 0.50
small = 16 * 0.20 + 8 * 0.35 + 0.50
print(f"by hand: 15x15 sum = 216*0.20 + 8*0.35 + 0.50 = {big:.4f}; 5x5 sum = 16*0.20 + 8*0.35 + 0.50 = {small:.4f}")
print(f"         ring = ({big:.4f} - {small:.4f}) / 200 = {(big - small) / 200:.4f}")
print(f"code:    ring = {float(ring[0, 0, 10, 10]):.4f}, count = {float(count[0, 0, 10, 10]):.0f}")
print("Chebyshev distances in the ring: 3..7 (the 5x5 box removes distances 0, 1 and 2)")

print("\n== noise sigma: RMS of horizontal neighbour differences / sqrt(2) ==")
row = torch.tensor([0.20, 0.23, 0.21, 0.22]).view(1, 1, 1, 4)
_, _, s = ring_and_noise(row, torch.ones_like(row, dtype=torch.bool))
d = np.diff([0.20, 0.23, 0.21, 0.22])
print("differences", np.round(d, 4), "squares", np.round(d ** 2, 6), "mean square", round(float((d ** 2).mean()), 6),
      "rms", round(float(np.sqrt((d ** 2).mean())), 5), "sigma", round(float(np.sqrt((d ** 2).mean()) / math.sqrt(2)), 5),
      "code", round(float(s[0, 0]), 5))
rng = np.random.default_rng(0)
x = 0.2 + rng.normal(0, 0.01, 200_000)
print("check on white noise sigma 0.01:", round(float(np.sqrt((np.diff(x) ** 2).mean()) / math.sqrt(2)), 5))

print("\n== log-uniform fill in [0.05, 1]: a = 0.05 ** (1 - u) ==")
for u in (0.0, 0.25, 0.5, 0.75, 1.0):
    print(f"u = {u}: a = {0.05 ** (1 - u):.4f}")
for q in (0.1, 0.25, 0.5):
    print(f"P(a <= {q}) = ln({q}/0.05)/ln(1/0.05) = {math.log(q / 0.05):.4f}/{math.log(20):.4f} = {math.log(q / 0.05) / math.log(20):.4f}")
u = torch.rand(1_000_000, generator=torch.Generator().manual_seed(0))
a = torch.exp(torch.log(torch.tensor(0.05)) * (1 - u))
print("sampled share <= 0.25:", round(float((a <= 0.25).float().mean()), 4), "median", round(float(a.median()), 4))

print("\n== smooth gain: 1 + 0.1 * clamp(c0 + c1 t + c2 t^2, -1, 1) ==")
c = (0.2, -0.4, 0.6)
for t in (-1.0, 0.0, 1.0):
    raw = c[0] + c[1] * t + c[2] * t * t
    print(f"t = {t:+.0f}: curve = {raw:.2f} -> clamp {max(-1, min(1, raw)):.2f} -> g = {1 + 0.1 * max(-1, min(1, raw)):.3f}")

print("\n== planting one pixel, three bands ==")
b = np.array([0.10, 0.30, 0.25]); t = np.array([0.40, 0.35, 0.05]); fill = 0.25
g = np.array([1.02, 1.00, 0.98]); n = np.array([0.003, -0.002, 0.001])
mix = fill * t + (1 - fill) * b
print("a t =", fill * t, "(1-a) b =", (1 - fill) * b, "sum =", mix, "g*sum =", g * mix, "x =", np.round(g * mix + n, 4))

print("\n== class scores: logsumexp over a class's cosines / tau, planted entry left out ==")
tau = 0.1
cos = torch.tensor([[0.95, 0.70, 0.60, 0.75, 0.20]])   # metal e1 (planted), e2, e3; plastic e4; oil e5
cls = torch.tensor([0, 0, 0, 1, 2])
z = cos / tau
m = SpectralNamer(np.linspace(400, 2500, 20).tolist(), [0.2] * 20, [0.1] * 20, compressed_channels=8, token_dim=8,
                  attention_heads=2, hidden=8, embed=4)
with torch.no_grad():
    m.log_tau.fill_(math.log(tau))
    # class_scores computes z @ e.T; feed unit "embeddings" whose dot products are the cosines above
    e = torch.eye(5); zz = cos
    full = m.class_scores(zz, e, cls, 3)
    out = m.class_scores(zz, e, cls, 3, exclude=torch.tensor([0]))
print("scaled scores z = cos/tau:", z.tolist())
print("with e1:   metal =", round(float(full[0, 0]), 4), "plastic =", round(float(full[0, 1]), 4), "oil =", round(float(full[0, 2]), 4))
print("e1 left out: metal = ln(e^7 + e^6) = 7 + ln(1 + e^-1) =", round(7 + math.log(1 + math.exp(-1)), 4), "; code", round(float(out[0, 0]), 4))
for name, sc in (("with e1", full), ("left out", out)):
    p = sc.softmax(1)[0]
    print(f"{name}: P(metal, plastic, oil) = {[round(float(v), 4) for v in p]}, CE(metal) = {-math.log(float(p[0])):.4f}")
p_entries = torch.softmax(torch.tensor([7.0, 6.0, 7.5, 2.0]), 0)
print("per-entry softmax over e2, e3, e4, e5:", [round(float(v), 4) for v in p_entries],
      "; metal = e2 + e3 =", round(float(p_entries[0] + p_entries[1]), 4))
w = torch.softmax(torch.tensor([7.0, 6.0]), 0)
print("gradient dL/dz_j = q_j - y_k * w_j: within-metal weights w =", [round(float(v), 4) for v in w])
q = p_entries
print("  e2:", round(float(q[0] - w[0]), 4), " e3:", round(float(q[1] - w[1]), 4), " e4:", round(float(q[2]), 4),
      " e5:", round(float(q[3]), 4), " sum:", round(float(q.sum() - 1), 6))
print("tau at start: exp(ln 0.07) = 0.07; 1/0.07 =", round(1 / 0.07, 3), "; ln 0.07 =", round(math.log(0.07), 4))
print("chance CE over 7 classes: ln 7 =", round(math.log(7), 4), "; chance BCE: ln 2 =", round(math.log(2), 4))

print("\n== the 20% hash split ==")
for name in ("Copper Metal", "Sheet_Metal GDS352 crg Galvn ASDFRa AREF", "Aluminum Metal"):
    h = hashlib.sha1(name.encode()).hexdigest()[:8]
    print(f"{name!r}: sha1[:8] = {h} = {int(h, 16):,}; / 0xFFFFFFFF ({0xFFFFFFFF:,}) = {int(h, 16) / 0xFFFFFFFF:.4f}"
          f" -> {'test' if int(h, 16) / 0xFFFFFFFF < 0.2 else 'train'}")

print("\n== the pass metric: man-made recall at ACE's natural false-alarm rate ==")
for s_ in ("prisma", "enmap", "aviris_ng"):
    j = json.loads((ROOT / f"research/experiments/spectral-namer/separability_{s_}.json").read_text())["0.25"]
    fa = 1 - j["level1_natural_called_natural"]
    print(f"{s_}: fill 0.25 ACE natural right {j['level1_natural_called_natural']:.4f} -> FA {fa:.4f}; ACE man-made "
          f"recall {j['level1_manmade_called_manmade']:.4f} -> namer needs >= {j['level1_manmade_called_manmade'] + 0.10:.4f}; "
          f"ACE level 2 {j['level2_correct_family']:.4f} -> needs >= {j['level2_correct_family'] + 0.10:.4f}")
nat = torch.tensor([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
mm = torch.tensor([-1.0, 0.4, 0.7, 1.5, 3.0])
cut = torch.quantile(nat, 1 - 0.2)
print("toy: cut = quantile(natural, 0.8) =", round(float(cut), 4), "(position 0.8*9 = 7.2: 0.5 + 0.2*(1.0-0.5))",
      "; recall =", float((mm > cut).float().mean()))

print("\n== parameter count, default namer (D = 64, hidden 256, embed 128) ==")
m = SpectralNamer(np.linspace(400, 2500, 50).tolist(), [0.2] * 50, [0.1] * 50)
for name, mod in m.named_children():
    print(f"  {name:10s} {sum(p.numel() for p in mod.parameters()):>8,}")
print(f"  log_tau    {m.log_tau.numel():>8,}")
print(f"  total      {sum(p.numel() for p in m.parameters()):>8,}")
for i, h, o in ((192, 256, 256), (64, 256, 128)):
    print(f"  MLP {i}->{h}->{h}->{o}: {i}*{h}+{h} + {h}*{h}+{h} + {h}*{o}+{o} = {i * h + h + h * h + h + h * o + o:,}")
print("  heads: to_embed 256*128+128 =", 256 * 128 + 128, "; to_manmade / to_fill 256+1 =", 257)
print("samples per step: 3 sensors x 16 tiles x 512 pixels =", 3 * 16 * 512, "; steps 40 x 500 =", 40 * 500,
      "; total =", f"{3 * 16 * 512 * 40 * 500:,}")

print("\n== PRISMA product grid (Set-1) ==")
from app.georef.from_scene_file import prisma_utm_grid
attrs = {"Epsg_Code": np.uint32(32642),
         "Product_ULcorner_easting": np.float32(545022.75), "Product_ULcorner_northing": np.float32(2501422.5),
         "Product_URcorner_easting": np.float32(580872.5), "Product_LLcorner_northing": np.float32(2465872.5)}
H, W = 1186, 1196
tr, epsg = prisma_utm_grid(attrs, (H, W))
print("rows H =", H, "cols W =", W)
print("east span  =", 580872.5 - 545022.75, "/ (W-1) =", (580872.5 - 545022.75) / (W - 1))
print("north span =", 2501422.5 - 2465872.5, "/ (H-1) =", (2501422.5 - 2465872.5) / (H - 1))
print("origin = (545022.75 - 15, 2501422.5 + 15) =", (545022.75 - 15, 2501422.5 + 15))
print("code:", tuple(tr)[:6], "EPSG", epsg)
print("centre of row 0 col 0:", tr * (0.5, 0.5), "; centre of row 1185 col 1195:", tr * (W - 0.5, H - 0.5))
print("float32 storage: 545022.75 ->", float(np.float32(545022.75)), "; 2501422.5 ->", float(np.float32(2501422.5)))
print("why pitch/(n-1): if the attrs were edges, pitch = span/n =", round((580872.5 - 545022.75) / W, 4), "m")
print("2.4 px at 30 m =", 2.4 * 30, "m; 5.6 px =", 5.6 * 30, "m")

print("\n== reflectance writer: x 10000, int16, nodata -9999, np.round is half-to-even ==")
vals = np.array([0.2, 0.12345, 0.12355, 1.5, np.nan, -0.5, 4.0])
out = np.where(np.isfinite(vals), np.clip(np.round(vals * 10000), -9998, 32767), -9999).astype(np.int16)
print("in ", vals.tolist()); print("x1e4", (vals * 10000).tolist()); print("out", out.tolist())
wl = np.array([1330.0, 1345.0, 1460.0, 1470.0, 1785.0, 1790.0, 1960.0, 1965.0])
keep = np.ones(len(wl), bool)
for lo, hi in ((1340, 1460), (1790, 1960)):
    keep &= ~((wl >= lo) & (wl <= hi))
print("drop 1340-1460,1790-1960 on", wl.tolist(), "-> keep", wl[keep].tolist())

print("\n== 8-connected objects ==")
from scipy.ndimage import label
mask = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], ])
mask = np.array([[1, 1, 0, 0, 0],
                 [0, 1, 0, 0, 1],
                 [0, 0, 1, 0, 1],
                 [0, 0, 0, 0, 0]])
l8, n8 = label(mask, structure=np.ones((3, 3), int))
l4, n4 = label(mask)
print(mask); print("8-connected:", n8, "objects\n", l8); print("4-connected:", n4, "objects\n", l4)

print("\n== post-stem BatchNorm running mean: 9 calls per step (3 sensors x pixel, ring, library), momentum 0.1 ==")
bn = torch.nn.BatchNorm1d(1)                      # PyTorch default momentum 0.1
print("momentum", bn.momentum)
calls = [f"{s}-{k}" for s in ("enmap", "prisma", "aviris_ng") for k in ("x", "r", "library")]
weights = {c: 0.0 for c in calls}
for step in range(2000):                          # many steps: steady state
    for c in calls:
        for k in weights:
            weights[k] *= 0.9
        weights[c] += 0.1
tot = sum(weights.values())
for c in calls:
    print(f"  {c:18s} {weights[c] / tot:.4f}")
print("  by hand: last call 0.1 / (1 - 0.9^9) =", round(0.1 / (1 - 0.9 ** 9), 4), "; 0.9^9 =", round(0.9 ** 9, 4))
for k in ("x", "r", "library"):
    print(f"  all {k:8s}", round(sum(w for c, w in weights.items() if c.endswith('-' + k)) / tot, 4))
# the same with a real BatchNorm1d fed constant batches whose mean marks the call
bn = torch.nn.BatchNorm1d(1); bn.train()
for step in range(300):
    for i, c in enumerate(calls):
        bn(torch.full((4, 1), float(i)) + torch.tensor([[-1.0], [1.0], [-1.0], [1.0]]))
print("  real BatchNorm1d running mean with call i feeding mean i:", round(float(bn.running_mean), 4),
      "; predicted", round(sum(i * weights[c] / tot for i, c in enumerate(calls)), 4))
print("  (the blend above is the unfixed code; since the namer2 bundle SpectralNamer.train() keeps norm in eval mode)")
m = SpectralNamer(np.linspace(400, 2500, 50).tolist(), [0.2] * 50, [0.1] * 50)
m.train()
before = m.norm.running_mean.clone()
m.encode(torch.rand(32, 50), torch.linspace(400, 2500, 50))
print("  fixed code: model.training =", m.training, "; norm.training =", m.norm.training,
      "; running mean unchanged after a training-mode call:", bool(torch.equal(before, m.norm.running_mean)),
      "; norm affine trainable:", all(p.requires_grad for p in m.norm.parameters()))

print("\n== the bank after de-duplication (one entry per name), EnMAP and AVIRIS-NG band lists ==")
import os, tempfile
from app.spectral_match.namer.bank import build_bank
from app.spectral_match.namer.classes import MANMADE_CLASSES
from app.utils.band_alignment import load_band_list
for f in ("enmap_224", "aviris_ng_372"):
    wl = np.asarray(load_band_list(str(ROOT / f"research/experiments/indradhanu-finetune/band_lists/{f}.json")), float)
    b = build_bank(wl, Path(tempfile.gettempdir()) / "namerchk", ROOT / "data/family_library")
    c, t = b.cls.numpy(), b.test.numpy()
    print(f"{f}: {len(b.names)} entries, {int((c >= 0).sum())} man-made, {int(t.sum())} held out; names unique: {len(set(b.names)) == len(b.names)}")
    for i, k in enumerate(MANMADE_CLASSES):
        n, h = int((c == i).sum()), int(((c == i) & t).sum())
        print(f"    {k:18s} {n:3d} {h:2d}  {100 * h / n:.0f}%")
