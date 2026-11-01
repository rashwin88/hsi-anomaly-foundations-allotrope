"""Numbers for chapter 42 (the spectral namer as built, full model) - rewritten 2026-10-31 (late) after level 1 by
matching. Run from the repo root:
    PYTHONPATH=. C:/av/Scripts/python.exe research/textbook/tools/compute_namer_full.py > research/textbook/tools/compute_namer_full_output.txt
Uses the real code wherever it can (SpectralNamer, SpectralReducedCompressor, band_widths, wavelength_encoding,
ring_and_noise, plant, class_scores, level1_logit, build_bank). Numbers that only illustrate are marked [A] in the
book; everything printed here is [C].
"""
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.foundation_models.components.spectral_reduced import (BIN_END_NM, BIN_ORIGIN_NM, MAX_GAP_NM,
                                                                SpectralReducedCompressor, band_widths)
from app.foundation_models.components.spectral_tokens import wavelength_encoding
from app.spectral_match.namer.bank import LibraryBank
from app.spectral_match.namer.generator import plant, ring_and_noise
from app.spectral_match.namer.model import SpectralNamer

ROOT = Path(__file__).resolve().parents[3]
torch.set_num_threads(8)


def h(t):
    print("\n== " + t + " ==")


# ---------------------------------------------------------------------------------------------------- parameters
h("1. parameter count of the namer as the code builds it today (D = 64, d = 32, 4 heads, hidden 256, embed 128)")
m = SpectralNamer(np.linspace(400, 2500, 50).tolist(), [0.2] * 50, [0.1] * 50)
for name, mod in m.named_children():
    print(f"  {name:10s} {sum(p.numel() for p in mod.parameters()):>8,}")
for name in ("log_tau", "l1_bias"):
    print(f"  {name:10s} {getattr(m, name).numel():>8,}")
tot = sum(p.numel() for p in m.parameters())
print(f"  total      {tot:>8,}   (347,684 before l1_bias was added for level 1 by matching)")
print("  stem by tensor:")
for n, p in m.stem.named_parameters():
    print(f"    {n:24s} {str(tuple(p.shape)):10s} {p.numel():>6,}")
print("  stem by hand: band_weight (32*64+64)+(64*32+32) =", 32 * 64 + 64 + 64 * 32 + 32,
      "; skip_weight (32*64+64)+(64*64+64) =", 32 * 64 + 64 + 64 * 64 + 64,
      "; bin_bias 32*32+32 =", 32 * 32 + 32, "; queries 64*32 =", 64 * 32,
      "; q,k,v,o 4*(32*32+32) =", 4 * (32 * 32 + 32), "; to_channel 32+1 =", 33)
print("  sum =", 4192 + 6272 + 1056 + 2048 + 4224 + 33)
print("  share of stem:", round(17825 / tot, 4), "; pixel tower (pixel MLP + 3 heads):", 180992 + 32896 + 257 + 257,
      "; library tower:", 115328)
print("  dead under level1 = matching: to_manmade 257 parameters receive no gradient")

# ---------------------------------------------------------------------------------------------------- stem pieces
h("2. wavelength encoding: 16 periods from 10 to 10,000 nm, geometric")
per = [10.0 * (1000.0 ** (i / 15)) for i in range(16)]
print("  ratio between periods 1000^(1/15) =", round(1000 ** (1 / 15), 4))
print("  periods (nm):", [round(p, 1) for p in per])
enc = wavelength_encoding(torch.tensor([1000.0, 1010.0]), 32)
print("  enc(1000 nm) first 4 sines:", [round(float(v), 4) for v in enc[0, :4]], "; slowest pair (sin, cos):",
      round(float(enc[0, 15]), 4), round(float(enc[0, 31]), 4))
print("  by hand: sin(2 pi 1000/10) = sin(200 pi) = 0; sin(2 pi 1000/15.85) =",
      round(math.sin(2 * math.pi * 1000 / per[1]), 4), "; sin(2 pi 1000/10000) =", round(math.sin(2 * math.pi * 0.1), 4),
      "cos =", round(math.cos(2 * math.pi * 0.1), 4))
print("  cosine of enc(1000) and enc(1010):", round(float(F.cosine_similarity(enc[:1], enc[1:])), 4))

h("3. bins: 40 nm from 400 nm")
G = math.ceil((BIN_END_NM - BIN_ORIGIN_NM) / 40.0)
print("  G = ceil((2500-400)/40) = ceil(", (BIN_END_NM - BIN_ORIGIN_NM) / 40.0, ") =", G, "; last bin", 400 + 40 * 52, "-",
      400 + 40 * 53, "nm")
stem = SpectralReducedCompressor(64, 32, 4, 40.0, "width")
for wl in (405.0, 439.9, 440.0, 1000.0, 2450.0, 2510.0, 380.0):
    print(f"  lambda {wl}: floor(({wl}-400)/40) = {math.floor((wl - 400) / 40)} -> bin {int(stem.bin_index(torch.tensor([wl]))[0])}")
print("  bin centres: 420, 460, ..., ", 400 + 40 * 52.5)

h("4. band widths (half the gaps, gaps capped at 20 nm)")
wl4 = torch.tensor([410.0, 430.0, 445.0, 470.0])
print("  bands", wl4.tolist(), "-> widths", band_widths(wl4).tolist())
wlg = torch.tensor([1320.0, 1330.0, 1340.0, 1460.0, 1470.0])
print("  across a water gap", wlg.tolist(), "-> widths", band_widths(wlg).tolist(), "(cap", MAX_GAP_NM, ")")

h("5. token averaging as one matmul, by hand (4 bands, 2 bins, d = 2 for paper) [k vectors assumed]")
v = np.array([0.5, -0.2, 1.0, 0.4])                   # normalised values
w = band_widths(wl4).numpy().astype(float)            # 20, 17.5, 20, 25
k = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, -1.0]])   # k(lambda_b), assumed
onehot = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
vals = v * w
cnt = w @ onehot
kmat = (onehot[:, :, None] * k[:, None, :]).reshape(4, 2 * 2)
tok = (vals @ kmat).reshape(2, 2) / cnt[:, None]
print("  w*v =", vals.tolist(), "; cnt per bin =", cnt.tolist())
print("  kmat (N, G*d) =\n", kmat)
print("  vals @ kmat =", (vals @ kmat).tolist(), "-> tokens", np.round(tok, 4).tolist())
direct = [sum(vals[b] * k[b] for b in (0, 1)) / cnt[0], sum(vals[b] * k[b] for b in (2, 3)) / cnt[1]]
print("  direct loop over bands:", [np.round(t, 4).tolist() for t in direct])

h("6. attention over bins, by hand: 1 query, 3 bins, head dim 2, bin 3 empty")
q = np.array([1.0, 0.5]); K = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]); V = np.array([[0.2, 0.4], [0.6, -0.2], [9.0, 9.0]])
sc = K @ q / math.sqrt(2)
print("  q.k / sqrt(2):", np.round(sc, 4).tolist())
sc_m = sc.copy(); sc_m[2] = -np.inf
a = np.exp(sc_m - sc_m[:2].max()); a /= a.sum()
print("  softmax with bin 3 masked:", np.round(a, 4).tolist(), "; output", np.round(a @ V, 4).tolist())
qt = torch.tensor(q, dtype=torch.float32).view(1, 1, 1, 2)
kt = torch.tensor(K, dtype=torch.float32).view(1, 1, 3, 2)
vt = torch.tensor(V, dtype=torch.float32).view(1, 1, 3, 2)
print("  torch SDPA with bool mask:",
      [round(float(x), 4) for x in F.scaled_dot_product_attention(qt, kt, vt, attn_mask=torch.tensor([True, True, False]).view(1, 1, 1, 3)).flatten()])

h("7. frozen BatchNorm, one channel")
bn = torch.nn.BatchNorm1d(1)
with torch.no_grad():
    bn.running_mean.fill_(0.4); bn.running_var.fill_(0.25); bn.weight.fill_(1.5); bn.bias.fill_(-0.1)
bn.eval()
print("  y = 1.5 (0.9 - 0.4)/sqrt(0.25 + 1e-5) - 0.1 =", round(float(bn(torch.tensor([[0.9]]))), 5),
      "; by hand", round(1.5 * 0.5 / math.sqrt(0.25 + 1e-5) - 0.1, 5))

# ---------------------------------------------------------------------------------------------------- real EnMAP list
h("8. EnMAP's 224-band list in the stem's bins")
enm = json.loads((ROOT / "research/experiments/indradhanu-finetune/band_lists/enmap_224.json").read_text())
wl_en = np.asarray(enm["wavelengths_nm"] if isinstance(enm, dict) else enm, float)
print("  bands", len(wl_en), "range", wl_en.min(), "-", wl_en.max())
bi = stem.bin_index(torch.tensor(wl_en, dtype=torch.float32)).numpy()
cnts = np.bincount(bi, minlength=G)
print("  bands per bin:", cnts.tolist())
print("  occupied bins:", int((cnts > 0).sum()), "of", G, "; most bands in one bin:", int(cnts.max()))
for lo, hi in ((1340, 1460), (1790, 1960)):
    pass
keep = np.ones(len(wl_en), bool)
for lo, hi in ((1340, 1460), (1790, 1960)):
    keep &= ~((wl_en >= lo) & (wl_en <= hi))
print("  bands left if 1340-1460 and 1790-1960 nm are dead [A: windows]:", int(keep.sum()))
cn2 = np.bincount(bi[keep], minlength=G)
print("  occupied bins then:", int((cn2 > 0).sum()))

# ---------------------------------------------------------------------------------------------------- shape walk
h("9. one training step for one sensor at illustrative EnMAP sizes: B=16, 128x128, N=200, L=378 (real code, random data)")
B, N, Hh, Ww, L, C = 16, 200, 128, 128, 378, 7
wl = torch.linspace(420, 2440, N)
g = torch.Generator().manual_seed(0)
base = torch.rand(B, N, 1, 1, generator=g) * 0.3 + 0.05
pixels = base + 0.01 * torch.randn(B, N, Hh, Ww, generator=g)
valid = torch.ones_like(pixels, dtype=torch.bool)
n_nat = 219
cls = torch.cat([torch.full((n_nat,), -1), torch.arange(L - n_nat) % C]).long()
test = torch.zeros(L, dtype=torch.bool); test[n_nat::5] = True
bank = LibraryBank(torch.rand(L, N, generator=g) * 0.5, cls, test, [str(i) for i in range(L)])
ring, count, sigma = ring_and_noise(pixels, valid)
print("  pixels", tuple(pixels.shape), f"= {pixels.numel():,} numbers = {pixels.numel() * 4 / 2**20:.0f} MiB fp32")
print("  ring", tuple(ring.shape), "count", tuple(count.shape), "sigma", tuple(sigma.shape))
print("  interior pixels allowed per tile: (128 - 2*7)^2 =", (128 - 14) ** 2)
t0 = time.time()
s = plant(pixels, valid, bank, per_tile=512, generator=g)
print("  plant:", {k_: tuple(v_.shape) for k_, v_ in s.items()}, f"{time.time() - t0:.1f}s")
kinds = torch.where(s["manmade"] > 0, 0, torch.where(s["entry"] >= 0, 1, 2))
print("  kinds drawn (man-made, natural mix, plain):", [int((kinds == i).sum()) for i in range(3)], "of", len(kinds))
model = SpectralNamer(np.linspace(400, 2500, 50).tolist(), [0.2] * 50, [0.1] * 50)
shapes = {}
def hook(name):
    def f(mod, inp, out):
        shapes.setdefault(name, []).append(tuple(out.shape))
    return f
for name in ("band_weight", "skip_weight", "bin_bias", "q", "k", "v", "o", "to_channel"):
    getattr(model.stem, name).register_forward_hook(hook("stem." + name))
model.stem.register_forward_hook(hook("stem(out)"))
model.norm.register_forward_hook(hook("norm"))
model.pixel.register_forward_hook(hook("pixel MLP"))
model.library.register_forward_hook(hook("library MLP"))
model.to_embed.register_forward_hook(hook("to_embed"))
model.to_fill.register_forward_hook(hook("to_fill"))
model.train()
t0 = time.time()
z, logit_head, fill = model.forward_pixel(s["x"], s["r"], wl)
e = model.forward_library(bank.refl, wl)
lib_cls = torch.where(bank.manmade & ~bank.test, bank.cls, torch.full_like(bank.cls, -1))
lib_cls = torch.where(bank.manmade, lib_cls, torch.full_like(lib_cls, C))
scores = model.class_scores(z, e, lib_cls, C + 1, exclude=s["entry"])
logit = model.level1_logit(scores)
print(f"  forward on CPU: {time.time() - t0:.1f}s")
for k_, v_ in shapes.items():
    print(f"  {k_:14s} {v_}")
print("  z", tuple(z.shape), "e", tuple(e.shape), "S (P, L) =", (len(z), L), "-> class scores", tuple(scores.shape),
      "level-1 logit", tuple(logit.shape), "fill", tuple(fill.shape))
print("  kmat elements N*G*d =", N * 53 * 32, "; tok elements P*G*d =", 8192 * 53 * 32, "; (P,N,d) avoided =", 8192 * N * 32)
print("  attention weights per stem call P*heads*D*G =", f"{8192 * 4 * 64 * 53:,}", f"= {8192 * 4 * 64 * 53 * 4 / 2**20:.0f} MiB fp32")
print("  SDPA chunks of 32,768 rows: P=8192 -> 1 chunk; library L=378 -> 1")
print("  score matrix elements P*L =", f"{8192 * L:,}")
excl = int((s["entry"] >= 0).sum())
print("  own-entry exclusions:", excl, "(every planted sample, man-made and natural mix)")
mm = s["cls"] >= 0
l1 = F.binary_cross_entropy_with_logits(logit, s["manmade"])
l2 = F.cross_entropy(scores[:, :C][mm], s["cls"][mm])
lf = (fill - s["fill"]).abs().mean()
print(f"  untrained losses: BCE {float(l1):.4f}, CE {float(l2):.4f} (ln 7 = {math.log(7):.4f}), fill L1 {float(lf):.4f}")
print("  classes with -inf (no train entry):", int(torch.isinf(scores[0]).sum()))

# ---------------------------------------------------------------------------------------------------- level 1
h("10. level 1 by matching, 3 library entries: plastic s=6.0, metal s=5.0, soil (natural) s=5.5; bias 0; truth man-made")
sc3 = torch.tensor([[6.0, 5.0, 5.5]], requires_grad=True)
mod = SpectralNamer(np.linspace(400, 2500, 20).tolist(), [0.2] * 20, [0.1] * 20, compressed_channels=8, token_dim=8,
                    attention_heads=2, hidden=8, embed=4)
ell = mod.level1_logit(sc3)
lossv = F.binary_cross_entropy_with_logits(ell, torch.ones(1))
lossv.backward()
print("  logit = LSE(6, 5) - 5.5 + 0 =", round(float(ell), 4), "; LSE(6,5) = 6 + ln(1 + e^-1) =", round(6 + math.log(1 + math.exp(-1)), 4))
print("  P(man-made) = sigmoid =", round(float(torch.sigmoid(ell)), 4), "; BCE =", round(float(lossv), 4))
print("  dL/dlogit = p - y =", round(float(torch.sigmoid(ell)) - 1, 4))
print("  autograd dL/ds:", [round(float(x), 4) for x in sc3.grad[0]], "; dL/dbias =", round(float(torch.sigmoid(ell)) - 1, 4))
print("  softmax weights within man-made: e^6/(e^6+e^5) =", round(math.exp(6) / (math.exp(6) + math.exp(5)), 4))
print("  sum of grads:", round(float(sc3.grad.sum()), 6))
sc3n = torch.tensor([[6.0, 5.0, 5.5]], requires_grad=True)
l_n = F.binary_cross_entropy_with_logits(mod.level1_logit(sc3n), torch.zeros(1)); l_n.backward()
print("  same scores, truth natural: BCE", round(float(l_n), 4), "grads", [round(float(x), 4) for x in sc3n.grad[0]])
print("  the test's pair: logits", [round(float(x), 4) for x in mod.level1_logit(torch.tensor(
    [[2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0], [-3.0] * 7 + [4.0]]))])

h("11. why one hyperplane fails: XOR on the unit circle; natural at 0 and 180 deg, man-made at 90 and 270 deg")
ang = {"n1": 0, "n2": 180, "m1": 90, "m2": 270}
vec = {k_: np.array([math.cos(math.radians(a_)), math.sin(math.radians(a_))]) for k_, a_ in ang.items()}
x80 = np.array([math.cos(math.radians(80)), math.sin(math.radians(80))])
cosv = {k_: float(x80 @ v_) for k_, v_ in vec.items()}
print("  pixel at 80 deg, cosines:", {k_: round(v_, 4) for k_, v_ in cosv.items()})
tau = 0.1
smm = math.log(math.exp(cosv["m1"] / tau) + math.exp(cosv["m2"] / tau))
snat = math.log(math.exp(cosv["n1"] / tau) + math.exp(cosv["n2"] / tau))
print(f"  tau 0.1: S_mm = LSE({cosv['m1']/tau:.4f}, {cosv['m2']/tau:.4f}) = {smm:.4f}; S_nat = LSE({cosv['n1']/tau:.4f}, {cosv['n2']/tau:.4f}) = {snat:.4f}; logit = {smm - snat:.4f}")
x190 = np.array([math.cos(math.radians(190)), math.sin(math.radians(190))])
c2 = {k_: float(x190 @ v_) for k_, v_ in vec.items()}
smm2 = math.log(sum(math.exp(c2[k_] / tau) for k_ in ("m1", "m2"))); snat2 = math.log(sum(math.exp(c2[k_] / tau) for k_ in ("n1", "n2")))
print(f"  pixel at 190 deg: logit = {smm2:.4f} - {snat2:.4f} = {smm2 - snat2:.4f}")

h("12. the entry-count offset: LSE of m equal scores = s + ln m")
bank_counts = {"natural": 219, "man-made train (N001b)": 125}
print("  EnMAP natural entries 378 - 159 =", 378 - 159, "; man-made train 159 - 34 =", 159 - 34)
print("  ln 219 =", round(math.log(219), 4), "; ln 125 =", round(math.log(125), 4), "; offset ln(219/125) =",
      round(math.log(219 / 125), 4))

h("13. evidence: what the losses mean (BCE -> geometric-mean probability of the truth)")
for name, b in (("chance", math.log(2)), ("linear head, 150 steps", 0.64), ("matching, 150 steps", 0.43)):
    print(f"  {name}: BCE {b:.4f} -> exp(-BCE) = {math.exp(-b):.4f}")

# ---------------------------------------------------------------------------------------------------- training law
h("14. the fill law and the sample law")
print("  E[a] man-made/natural mix = (1 - 0.05)/ln 20 = 0.95 /", round(math.log(20), 4), "=", round(0.95 / math.log(20), 4))
print("  E[fill target] over all samples = 0.75 *", round(0.95 / math.log(20), 4), "=", round(0.75 * 0.95 / math.log(20), 4))
print("  median = 0.05^0.5 =", round(0.05 ** 0.5, 4))
print("  E[a^2] = (1 - 0.05^2)/(2 ln 20) =", round((1 - 0.0025) / (2 * math.log(20)), 4))
print("  per step per sensor 16*512 =", 16 * 512, "; expected man-made 4096, natural mix 2048, plain 2048")
print("  eval: 40 batches x 8192 =", 40 * 8192, "samples per sensor per fill (kinds 0.5/0.5/0)")
print("  P(a <= 0.1, 0.25, 0.5):", [round(math.log(q_ / 0.05) / math.log(20), 4) for q_ in (0.1, 0.25, 0.5)])

h("15. learning-rate schedule (LambdaLR, total 20,000 steps, warmup 500, base 3e-4, stem 0.3x)")
def lam(s_):
    return min(1.0, (s_ + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s_ / 20000)))
for s_ in (0, 1, 249, 499, 500, 5000, 10000, 15000, 19999):
    print(f"  step {s_:>5}: factor {lam(s_):.5f}  lr {3e-4 * lam(s_):.3e}  stem {0.3 * 3e-4 * lam(s_):.3e}")

h("16. the bank, N001b and N002, on EnMAP and AVIRIS-NG band lists")
from app.spectral_match.namer.bank import build_bank
from app.spectral_match.namer.classes import MANMADE_CLASSES
from app.utils.band_alignment import load_band_list
for f in ("enmap_224", "aviris_ng_372"):
    wlb = np.asarray(load_band_list(str(ROOT / f"research/experiments/indradhanu-finetune/band_lists/{f}.json")), float)
    b = build_bank(wlb, Path(tempfile.gettempdir()) / "namerchk", ROOT / "data/family_library")
    c_, t_ = b.cls.numpy(), b.test.numpy()
    print(f"  {f}: {len(b.names)} entries, natural {int((c_ < 0).sum())}, man-made {int((c_ >= 0).sum())}, test {int(t_.sum())}, "
          f"train man-made {int(((c_ >= 0) & ~t_).sum())}")
    for i, k_ in enumerate(MANMADE_CLASSES):
        n_, h_ = int((c_ == i).sum()), int(((c_ == i) & t_).sum())
        print(f"    {k_:18s} {n_:3d} {h_:2d}  {100 * h_ / n_:.0f}%")
    tx = MANMADE_CLASSES.index("textile_fabric")
    unseen = t_ | (c_ == tx)
    print(f"    N002 (textile held out): train man-made pool {int(((c_ >= 0) & ~unseen).sum())}, "
          f"test pool {int(((c_ >= 0) & unseen).sum())} (textile {int((c_ == tx).sum())})")
    print(f"    level-1 matching classes in training: natural entries {int((c_ < 0).sum())} as class 7")

h("17. hash split examples")
for name in ("Copper Metal", "Aluminum Metal"):
    hx = hashlib.sha1(name.encode()).hexdigest()[:8]
    print(f"  {name!r}: {hx} = {int(hx, 16):,} / {0xFFFFFFFF:,} = {int(hx, 16) / 0xFFFFFFFF:.4f}")

h("18. pass bars at fill 0.25 (separability JSONs)")
for s_ in ("prisma", "enmap", "aviris_ng"):
    j = json.loads((ROOT / f"research/experiments/spectral-namer/separability_{s_}.json").read_text())["0.25"]
    print(f"  {s_}: f = {1 - j['level1_natural_called_natural']:.4f}, ACE recall {j['level1_manmade_called_manmade']:.4f}, "
          f"bar {j['level1_manmade_called_manmade'] + 0.1:.4f}")

h("19. inference: ring_means on a 9x9 single-band toy is the generator's ring")
from app.spectral_match.namer.infer import ring_means
cube = np.random.default_rng(1).random((2, 20, 20)).astype(np.float32)
vv = np.ones_like(cube, bool); vv[0, 3, 4] = False
got = ring_means(cube, vv, np.array([10]), np.array([10]))
rg, _, _ = ring_and_noise(torch.from_numpy(cube)[None], torch.from_numpy(vv)[None])
print("  infer ring", np.round(got[0], 5).tolist(), "generator ring", [round(float(x), 5) for x in rg[0, :, 10, 10]])
print("  bands dropped at inference when valid at <= 50% of requested pixels: 3 of 5 valid = 0.6 keep; 2 of 4 = 0.5 drop")

h("20. ACE at fill 1.0 (the laptop test's fill) and the LSE-of-LSE identity")
for s_ in ("prisma", "enmap", "aviris_ng"):
    j = json.loads((ROOT / f"research/experiments/spectral-namer/separability_{s_}.json").read_text())["1.0"]
    print(f"  {s_}: fill 1.0 f = {1 - j['level1_natural_called_natural']:.4f}, ACE man-made recall {j['level1_manmade_called_manmade']:.4f}")
ea, eb, ec = math.exp(6.0), math.exp(5.0), math.exp(5.5)
print(f"  e^6 = {ea:.4f}, e^5 = {eb:.4f}, e^5.5 = {ec:.4f}; man-made mass share (e^6+e^5)/(e^6+e^5+e^5.5) = {(ea + eb) / (ea + eb + ec):.4f}")
x = torch.tensor([[1.0, 2.0, 0.5, 3.0, -1.0]])
grp = [[0, 1], [2], [3, 4]]
lse_of = torch.logsumexp(torch.stack([torch.logsumexp(x[0, g_], 0) for g_ in grp]), 0)
print("  LSE over class LSEs", round(float(lse_of), 4), "= LSE over all entries", round(float(torch.logsumexp(x[0], 0)), 4))
print("  with bias b = 0.5: P(M) = e^b A / (e^b A + N) =",
      round(math.exp(0.5) * (ea + eb) / (math.exp(0.5) * (ea + eb) + ec), 4), "; sigmoid(0.8133 + 0.5) =",
      round(1 / (1 + math.exp(-(0.8133 + 0.5))), 4))

h("21. answers: hidden 128 / embed 64 count; level 1 with a second natural entry; widths at 1790-1970; equal scores")
m2 = SpectralNamer(np.linspace(400, 2500, 50).tolist(), [0.2] * 50, [0.1] * 50, hidden=128, embed=64)
print("  hidden 128, embed 64: total", f"{sum(p.numel() for p in m2.parameters()):,}")
sc4 = torch.tensor([[6.0, 5.0, 5.5]], requires_grad=True)   # plastic, metal | natural class pooled below
nat = torch.logsumexp(torch.tensor([5.5, 5.5]), 0)
s4 = torch.tensor([6.0, 5.0, 5.5, 5.5], requires_grad=True)
ell4 = torch.logsumexp(s4[:2], 0) - torch.logsumexp(s4[2:], 0)
l4 = F.binary_cross_entropy_with_logits(ell4[None], torch.ones(1)); l4.backward()
print("  S_nat = LSE(5.5, 5.5) =", round(float(nat), 4), "; logit", round(float(ell4), 4), "; P", round(float(torch.sigmoid(ell4)), 4),
      "; BCE", round(float(l4), 4), "; grads", [round(float(x), 4) for x in s4.grad])
print("  widths 1790,1800,1960,1970:", band_widths(torch.tensor([1790.0, 1800.0, 1960.0, 1970.0])).tolist(),
      "bins", stem.bin_index(torch.tensor([1790.0, 1800.0, 1960.0, 1970.0])).tolist())
print("  equal scores: P(M) = 125/(125+219) =", round(125 / 344, 4), "; b = ln(219/125) =", round(math.log(219 / 125), 4))
print("  a_min 0.02: P(a<=0.25) =", round(math.log(12.5) / math.log(50), 4), "; mean =", round(0.98 / math.log(50), 4))
print("  pool example: S_nat = ln(e^2.5 + 1) =", round(math.log(math.exp(2.5) + 1), 4), "; LSE(3,2) =",
      round(math.log(math.exp(3) + math.exp(2)), 4), "; logit", round(math.log(math.exp(3) + math.exp(2)) - math.log(math.exp(2.5) + 1), 4),
      "; P(M)", round(1 / (1 + math.exp(-(math.log(math.exp(3) + math.exp(2)) - math.log(math.exp(2.5) + 1)))), 4),
      "; P(plastic|M)", round(math.exp(3) / (math.exp(3) + math.exp(2)), 4))
print("  attention by hand: e^0.7071 =", round(math.exp(0.7071), 4), "e^0.3536 =", round(math.exp(0.3536), 4))
print("  lerp example: v = (0.35-0.308)/0.108 =", round((0.35 - 0.308) / 0.108, 4))
print("  new library 500 spectra MLP mult-adds:", 500 * (64 * 256 + 256 * 256 + 256 * 128))
