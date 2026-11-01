"""Numbers and figure data for Part X (chapters 44-hslwhat ... 49-hslverdict): why not HyperSL.

Run from anywhere with the product venv:  C:/av/Scripts/python.exe research/textbook/tools/compute_hypersl.py
Writes tools/compute_hypersl_output.txt and figdata/hsl_*.dat. Every [C] number in Part X comes from here.

Inputs (all read-only):
  figdata/hypersl_ckpt_shapes.json        tensor shapes of the released checkpoint (tools/hypersl_ckpt_shapes.py, WSL)
  the Set-1 PRISMA HE5 band attributes     C:/Users/FITT/Documents/stage1_datasets (the 149- and 234-band axes)
  app/models/dataset/spectralearth band table (EnMAP minus its water windows, 202 bands)
  app/models/dataset/vendables DEFAULT_COMMON_WAVELENGTH_GRID (the product's 165-band grid)
  research/experiments/hypersl-anomaly-bench/results.json (the detection run, if present)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parent.parent
REPO = T.parents[1]
sys.path.insert(0, str(REPO))
FIG = T / "figdata"
FIG.mkdir(exist_ok=True)
OUT: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def angle(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return c, math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ------------------------------------------------------------------ 1. parameter count
say("== 1. HyperSL parameters, by hand and from the checkpoint")
D, F = 256, 1024
block = 2 * D + (D * 3 * D + 3 * D) + (D * D + D) + 2 * D + (D * F + F) + (F * D + D)
say("one transformer block:", block)
enc = 8 * block
dec = 3 * block
cross = 2 * (D * D + D)
convs = (5 * D + D) + (9 * D + D) + (13 * D + D) + (3 * D * D + D)
bn_affine = 4 * 2 * D
tokens = 2 * D
norms = 2 * 2 * D
head = 3 * D + 1
trainable = enc + dec + cross + convs + bn_affine + tokens + norms + head
say("encoder 8 blocks", enc, "decoder 3 blocks", dec, "cross-attn", cross, "tokenizer convs", convs,
    "BN affine", bn_affine, "tokens", tokens, "norms", norms, "head", head)
say("trainable total", trainable)
ck = json.loads((FIG / "hypersl_ckpt_shapes.json").read_text())
buf = sum(r["numel"] for r in ck["tensors"] if any(s in r["name"] for s in ("running_", "num_batches", "pe", "div_term")))
say("checkpoint total entries", ck["total_numel"], "buffers (pe, div_term, BN stats)", buf,
    "-> trainable", ck["total_numel"] - buf, "epoch", ck["epoch"], "sha256", ck["sha256"][:8], "...", ck["sha256"][-7:])
say("tokenizer share of trainable %.2f%%" % (100 * (convs + bn_affine) / trainable),
    "| conv1x1 768->256 alone", 3 * D * D + D)
# positional encoding frequencies
div = np.exp(np.arange(0, D, 2) * -(math.log(100.0) / D))
say("div_term first, second, last: %.6f %.6f %.6f" % (div[0], div[1], div[-1]))
for lam in (450.0, 1000.0, 2400.0):
    p = (lam - 400) / 10
    say(f"  lambda {lam:.0f}: p={(p):.1f}  sin(p*div0)={math.sin(p * div[0]):+.4f}  cos={math.cos(p * div[0]):+.4f}"
        f"  sin(p*div_last)={math.sin(p * div[-1]):+.4f}")
say("wavelength in um instead of nm: p for 1.0 um =", (1.0 - 400) / 10)
say("mask 0.95 on 149 bands: visible int(149*0.05) =", int(149 * 0.05), "hidden", 149 - int(149 * 0.05))
say("mask 0.95 on 224-7=217 EnMAP bands: visible", int(217 * 0.05))

# ------------------------------------------------------------------ 2. per-scene min-max toy
say("\n== 2. per-scene min-max on a 3-pixel, 2-band toy scene")


def minmax(X):
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    return (X - lo) / (hi - lo), lo, hi


S1 = np.array([[0.10, 0.30], [0.20, 0.50], [0.30, 0.40]])
S2 = np.array([[0.05, 0.20], [0.20, 0.50], [0.40, 0.60]])
for name, S in (("S1", S1), ("S2", S2)):
    Xs, lo, hi = minmax(S)
    say(name, "min", lo, "max", hi, "span", hi - lo)
    for x, xs in zip(S, Xs):
        say("   ", x, "->", np.round(xs, 4))
say("(a) the same spectrum (0.20, 0.50) is", np.round(minmax(S1)[0][1], 4), "in S1 and", np.round(minmax(S2)[0][1], 4), "in S2")
S1b = np.vstack([S1, [0.60, 0.90]])
Xb, lo, hi = minmax(S1b)
say("(b) S1 + bright anomaly (0.60, 0.90): min", lo, "max", hi)
for x, x0, xs in zip(S1, minmax(S1)[0], Xb[:3]):
    say("   ", x, "was", np.round(x0, 4), "now", np.round(xs, 4))
say("    anomaly ->", np.round(Xb[3], 4))
t = np.array([0.70, 0.20])
_, lo, hi = minmax(S1)
ts = (t - lo) / (hi - lo)
say("(c) library t=(0.70, 0.20) in S1's frame:", np.round(ts, 4), "- outside [0, 1]; clipped:", np.clip(ts, 0, 1))
S1t = np.vstack([S1, t])
Xt, lo2, hi2 = minmax(S1t)
say("    if t is in the scene, the frame becomes min", lo2, "max", hi2, "and t ->", np.round(Xt[3], 4),
    "; p1 ->", np.round(Xt[0], 4))
say("(d) spectral angle before and after the per-band map, S1")
Xs = minmax(S1)[0]
for i, j in ((0, 1), (1, 2), (0, 2)):
    c0, a0 = angle(S1[i], S1[j])
    c1, a1 = angle(Xs[i], Xs[j])
    say(f"    p{i + 1},p{j + 1}: reflectance cos {c0:.4f} angle {a0:.2f} deg | min-max cos {c1:.4f} angle {a1:.2f} deg")
# scaling only (no offset) also changes the angle
u, v, s = np.array([0.20, 0.50]), np.array([0.30, 0.40]), np.array([1 / 0.20, 1 / 0.20])
say("    pure scaling by 1/span:", angle(u, v), angle(u / np.array([0.20, 0.20]), v / np.array([0.20, 0.20])),
    angle(u / np.array([0.20, 0.10]), v / np.array([0.20, 0.10])))
# WavelengthStats is fixed: same map for pixel and library
mean, std = np.array([0.25, 0.35]), np.array([0.10, 0.12])
say("    fixed z-score (mean 0.25/0.35, std 0.10/0.12): (0.20,0.50) ->", np.round(((np.array([0.20, 0.50]) - mean) / std), 4),
    "in every scene; t ->", np.round((t - mean) / std, 4))

# ------------------------------------------------------------------ 3. conv over band index
say("\n== 3. a kernel over band index: how many nm it spans")


def prisma_axes():
    import h5py
    he5 = Path(r"C:\Users\FITT\Documents\stage1_datasets\Set-1\Hypersepctral Datasets\PRS_L2D_STD_20201214060713_20201214060717_0001.he5")
    with h5py.File(he5, "r") as f:
        a = f.attrs
        cw = np.concatenate([np.ravel(a["List_Cw_Vnir"]), np.ravel(a["List_Cw_Swir"])])
        fl = np.concatenate([np.ravel(a["List_Cw_Vnir_Flags"]), np.ravel(a["List_Cw_Swir_Flags"])])
    good = fl.astype(bool) & (cw > 0)
    idx = np.flatnonzero(good)[np.argsort(cw[good], kind="stable")]
    full = cw[idx].astype(float)
    sel = full.copy()
    drop = np.zeros(len(idx), bool)
    for lo, hi in ((400.0, 485.0), (1130.0, 1175.0), (1325.0, 1495.0), (1790.0, 2030.0), (2350.0, 2500.0)):
        drop |= (sel >= lo) & (sel <= hi)
    drop |= (sel >= 915.0) & (sel <= 985.0) & (idx >= 66)
    return full, sel[~drop]


def spans(wl, k):
    h = (k - 1) // 2
    wl = np.sort(wl)
    return np.array([wl[min(i + h, len(wl) - 1)] - wl[max(i - h, 0)] for i in range(len(wl))])


from app.models.dataset.spectralearth import load_band_table  # noqa: E402
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID  # noqa: E402

p234, p149 = prisma_axes()
axes = {"PRISMA 234 (all good bands)": p234, "PRISMA 149 (HyperSL runs here)": p149,
        "EnMAP via SpectralEarth 202 (water windows removed)": np.sort(load_band_table().wavelengths_nm),
        "product 165-band grid": np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, float)}
for name, wl in axes.items():
    wl = np.sort(wl)
    gaps = np.diff(wl)
    big = np.flatnonzero(gaps > 25)
    say(f"{name}: {len(wl)} bands, {wl[0]:.1f}-{wl[-1]:.1f} nm, median step {np.median(gaps):.2f} nm; gaps > 25 nm:",
        "; ".join(f"{wl[i]:.1f}->{wl[i + 1]:.1f} ({gaps[i]:.1f})" for i in big))
    for k in (5, 9, 13):
        s = spans(wl, k)
        h = (k - 1) // 2
        inner = s[h:len(s) - h]
        i = int(np.argmax(inner)) + h
        say(f"   kernel {k:2d}: span min {inner.min():.1f}  median {np.median(inner):.1f}  max {inner.max():.1f} nm"
            f"  (centred at {wl[i]:.1f}: {wl[i - h]:.1f} .. {wl[i + h]:.1f})")
p = np.sort(p149)
s5 = spans(p, 5)
with open(FIG / "hsl_kernel_span_prisma149.dat", "w") as f:
    f.write("wl span5 span13\n")
    s13 = spans(p, 13)
    for i in range(2, len(p) - 2):
        f.write(f"{p[i]:.2f} {s5[i]:.2f} {s13[i] if 6 <= i < len(p) - 6 else float('nan'):.2f}\n")
# the worked 5-band windows across the 1325-1495 gap
i = int(np.flatnonzero(np.diff(p) > 100)[0])
say("PRISMA149 bands around the first gap > 100 nm:", np.round(p[i - 4:i + 5], 2))
say("   window before gap", np.round(p[i - 6:i - 1], 2), "span", round(p[i - 2] - p[i - 6], 2))
say("   window across gap", np.round(p[i - 2:i + 3], 2), "span", round(p[i + 2] - p[i - 2], 2))
gaps_nm = np.diff(p)
say("   steps inside that window:", np.round(np.diff(p[i - 2:i + 3]), 2))

# ------------------------------------------------------------------ 4. self-completion toy
say("\n== 4. self-completion vs spatial completion, 3 bands")
b = np.array([0.20, 0.30, 0.40])
tt = np.array([0.60, 0.50, 0.40])
dip = np.array([0.60, 0.30, 0.60])
for nm, x in (("background", b), ("smooth anomaly", tt), ("anomaly with a dip", dip)):
    self_pred = (x[0] + x[2]) / 2
    r_self = x[1] - self_pred
    r_sp = x - b
    say(f"{nm} {x}: self-completion of band 2 = ({x[0]:.2f}+{x[2]:.2f})/2 = {self_pred:.3f}, residual {r_self:+.3f};"
        f" spatial residual {np.round(r_sp, 3)}, squared sum {float(r_sp @ r_sp):.4f}")
# least-squares learned predictor of band 2 from bands 1 and 3 on noisy linear backgrounds
rng = np.random.default_rng(0)
n = 2000
off = rng.uniform(0.05, 0.4, n)
slope = rng.uniform(-0.1, 0.15, n)
B = np.stack([off, off + slope, off + 2 * slope], 1) + rng.normal(0, 0.005, (n, 3))
A = np.c_[B[:, 0], B[:, 2], np.ones(n)]
coef, *_ = np.linalg.lstsq(A, B[:, 1], rcond=None)
say("least-squares band2 ~ w1*band1 + w3*band3 + c on 2000 noisy linear backgrounds [A]: w1 %.4f w3 %.4f c %.5f" % tuple(coef))
for nm, x in (("smooth anomaly", tt), ("dip", dip)):
    say(f"   learned predictor on {nm}: {coef[0] * x[0] + coef[1] * x[2] + coef[2]:.4f} (true {x[1]:.2f})")

# ------------------------------------------------------------------ 5. sub-pixel smoothing and TV
say("\n== 5. 3x3 smoothing and a TV penalty on an abundance map")
for a in (1.0, 0.51, 0.13):
    say(f"single pixel at fill {a}: 3x3 mean at centre = {a}/9 = {a / 9:.4f}; at a neighbour also {a / 9:.4f}")
# TV on one pixel: 4 edges of height a
c = 0.20          # ||t - b||^2 from section 4's smooth anomaly, per unit fill squared
for lam in (0.005, 0.013, 0.02):
    thr = 2 * lam / c
    say(f"lambda {lam}: J(a') = c (a - a')^2 + 4 lambda a'; minimiser a' = max(0, a - 2 lambda / c) = max(0, a - {thr:.3f});"
        f" a=0.13 -> {max(0.0, 0.13 - thr):.3f}; a=0.51 -> {max(0.0, 0.51 - thr):.3f}; a=1 -> {max(0.0, 1 - thr):.3f}")
for side in (1, 2, 3, 5):
    area, perim = side * side, 4 * side
    say(f"square target {side}x{side}: area {area}, perimeter edges {perim}, perimeter/area {perim / area:.2f}")

# ------------------------------------------------------------------ 6. recorded results -> figdata
say("\n== 6. recorded results turned into figure data")
cloud = {"C": [[341395, 36215, 4955, 909], [22708, 244246, 11452, 2042], [21979, 28341, 20472, 1], [1619, 5025, 36, 42725]],
         "B": [[342688, 25158, 12901, 2727], [27357, 225595, 25510, 1986], [6861, 15888, 48037, 7], [406, 1135, 16, 47848]]}
names = ["clear", "cloud", "cirrus", "snow"]
rec = {}
for k, m in cloud.items():
    m = np.array(m)
    say(f"cloud variant {k}: total {m.sum():,}; row sums {m.sum(1).tolist()}")
    r = np.diag(m) / m.sum(1)
    pr = np.diag(m) / m.sum(0)
    rec[k] = (r, pr)
    say("   recall", np.round(100 * r, 1), "precision", np.round(100 * pr, 1),
        "overall accuracy %.2f%%" % (100 * np.trace(m) / m.sum()))
    say("   cirrus row: correct %d of %d; called clear %d, cloud %d" % (m[2, 2], m[2].sum(), m[2, 0], m[2, 1]))
with open(FIG / "hsl_cloud_recall.dat", "w") as f:
    f.write("cls idx recallC recallB precC precB\n")
    for i, nm in enumerate(names):
        f.write(f"{nm} {i} {100 * rec['C'][0][i]:.1f} {100 * rec['B'][0][i]:.1f} {100 * rec['C'][1][i]:.1f} {100 * rec['B'][1][i]:.1f}\n")
hysu = {"side": [0.25, 0.5, 1, 2, 3], "area": [0.13, 0.51, 2.0, 8.2, 18.4],
        "ace": [0.97, 1.00, 1.00, 1.00, 1.00], "ace_lib": [0.88, 0.99, 1.00, 1.00, 1.00],
        "hsl_anch": [0.68, 0.77, 1.00, 1.00, 1.00], "hsl_blind": [0.72, 0.62, 0.80, 0.88, 0.92]}
with open(FIG / "hsl_hysu_auc.dat", "w") as f:
    f.write("i side area ace acelib hslanch hslblind\n")
    for i in range(5):
        f.write(f"{i} {hysu['side'][i]} {hysu['area'][i]} {hysu['ace'][i]} {hysu['ace_lib'][i]} {hysu['hsl_anch'][i]} {hysu['hsl_blind'][i]}\n")
say("HySU true areas vs 3x3 smoothing: area/9 =", [round(a / 9, 3) for a in hysu["area"]])
msad = {"DC1": (0.000, 0.092, 0.084, 0.173, 0.173), "DC2": (0.000, 0.085, 0.082, 0.182, 0.175),
        "DC3": (0.069, 0.124, 0.164, 0.199, 0.153), "EGU": (0.004, 0.106, 0.083, 0.264, 0.249)}
with open(FIG / "hsl_msad.dat", "w") as f:
    f.write("i set vca upstream noR paper scratch\n")
    for i, (k, v) in enumerate(msad.items()):
        f.write(f"{i} {k} " + " ".join(f"{x:.3f}" for x in v) + "\n")
say("mSAD rad -> deg for DC3 VCA 0.069 and paper 0.199:", round(math.degrees(0.069), 2), round(math.degrees(0.199), 2))
fp = {"baseline": [0.339, 0.128, 0.198, 0.262], "scores/B": [0.059, 0.219, 0.223, 0.249], "hypersl/B": [0.615, 0.464, 0.188, 0.318],
      "both/B": [0.611, 0.466, 0.159, 0.365]}
for k, v in fp.items():
    say(f"fp-filter oracle share of false-positive pixels removed at 90% object recall, {k}: {v}, mean {np.mean(v):.3f}")
recon = {"set1": (3.176, 6.174, 0.094), "set2": (4.021, 5.160, 0.220), "set3": (3.261, 4.267, 0.242), "set4": (2.741, 3.988, 0.104)}
for k, (a, bg, auc) in recon.items():
    say(f"splib run, mask 0, per-curve: {k} median recon angle anomalous {a:.2f} vs background {bg:.2f} deg, AUC {auc:.3f}")
say("mean recon AUC", round(np.mean([v[2] for v in recon.values()]), 3))
chik = {"raw": (39.60, 42.69), "PCA": (37.68, 41.11), "HyperSL band 0-1": (59.60, 60.23), "HyperSL raw": (48.19, None)}
for k, (o, p_) in chik.items():
    say(f"Chikusei K-means OA {k}: ours {o}" + (f" paper {p_} diff {o - p_:+.2f}" if p_ else ""))
say("HyperSL raw vs 0-1 input OA loss", round(59.60 - 48.19, 2))
ft = {"BOS": (99.01, 99.66, 97.55, 97.05), "IP": (97.73, 98.48, 94.94, 95.04)}
for k, (pre, pp, rnd, pr) in ft.items():
    say(f"fine-tune {k}: gain ours {pre - rnd:+.2f} paper {pp - pr:+.2f}; ratio {(pre - rnd) / (pp - pr):.2f}")

# ------------------------------------------------------------------ 7. the new detection run
res_path = REPO / "research/experiments/hypersl-anomaly-bench/results.json"
if res_path.exists():
    say("\n== 7. HyperSL on the labelled PRISMA bench (research/experiments/hypersl-anomaly-bench)")
    r = json.loads(res_path.read_text())
    S = r["summary"]
    order = ["FTH3_ep5 RM, cloud excl.", "FTH3_ep5 RM", "PCA-RX, cloud excl.", "PCA-RX", "MNF-RX, cloud excl.", "MNF-RX",
             "HyperSL RM, cloud excl.", "HyperSL RM", "HyperSL L2, cloud excl.", "HyperSL L2", "HyperSL loss, cloud excl.",
             "HyperSL loss"]
    with open(FIG / "hsl_detect.dat", "w") as f:
        f.write("i label pauc caught1 caught01\n")
        for i, k in enumerate(order):
            s = S[k]
            say(f"{k:28s} pAUC {s['pauc_0.01']:.4f}  ROC {s['roc_auc']:.3f}  TPR@0.001 {s['tpr_at_0.001']:.3f}  "
                f"caught1 {s['caught_1_n']} ({100 * s['caught_1']:.1f}%)  caught0.1 {100 * s['caught_0.1']:.1f}%  "
                f"tiers {[round(100 * s['tier_caught_1'][t], 1) for t in ('easy', 'medium', 'subtle')]}  per-scene pAUC "
                f"{[round(v, 4) for v in s['per_scene_pauc'].values()]}")
            f.write(f"{i} {{{k}}} {s['pauc_0.01']:.4f} {100 * s['caught_1']:.2f} {100 * s['caught_0.1']:.2f}\n")
    say("chance caught top 1%:", round(100 * S["HyperSL L2"]["chance_1"], 1))
    bar = ["HyperSL loss", "HyperSL L2", "MNF-RX", "HyperSL RM", "PCA-RX", "FTH3_ep5 RM"]
    with open(FIG / "hsl_detect_bar.dat", "w") as f:
        f.write("pos pauc caught1 paucx caught1x\n")
        for i, k in enumerate(bar):
            s, x = S[k], S[k + ", cloud excl."]
            f.write(f"{i} {s['pauc_0.01']:.4f} {100 * s['caught_1']:.2f} {x['pauc_0.01']:.4f} {100 * x['caught_1']:.2f}\n")
    n_px = sum(v["pixels"] for v in r["step1"]["scenes"].values())
    exp = sum(v["pixels"] * 149 * (7 / 149) ** 4 for v in r["step1"]["scenes"].values())
    say(f"(pixel, band) pairs never hidden in 4 draws: expected {exp:.0f} = N x 149 x (7/149)^4 summed; measured "
        f"{sum(v['bands_never_hidden'] for v in r['step1']['scenes'].values())}; (7/149)^4 = {(7 / 149) ** 4:.4e}")
    s1 = r["step1"]["scenes"]["set1"]
    say(f"set1 expected never-hidden: {s1['pixels']} x 149 x (7/149)^4 = {s1['pixels'] * 149 * (7 / 149) ** 4:.0f}; measured {s1['bands_never_hidden']}")
    for k, b in r["bootstrap"].items():
        say(f"{k}: pAUC {b['pauc_diff']:+.4f} [{b['pauc_ci'][0]:+.4f}, {b['pauc_ci'][1]:+.4f}], caught1 "
            f"{100 * b['caught_diff']:+.1f} [{100 * b['caught_ci'][0]:+.1f}, {100 * b['caught_ci'][1]:+.1f}] pts")
    for k, v in r["step1"]["scenes"].items():
        say(f"step1 {k}: {v}")
    gpu = sum(v["gpu_s"] for v in r["step1"]["scenes"].values())
    px = sum(v["pixels"] for v in r["step1"]["scenes"].values())
    say(f"GPU time total {gpu:.0f} s for {px:,} pixels x {r['step1']['draws']} draws: {1e6 * gpu / px:.0f} us per pixel")

(T / "tools" / "compute_hypersl_output.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
