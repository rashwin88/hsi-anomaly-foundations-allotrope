"""AVIRIS-NG 600-line window, full resolution, pixel-index pairing (same instrument grid): seg 200 vs seg 50 vs JPL corr.
Wavelength ranges, surface type (NDVI from JPL), brightness, spectral angle, per-pixel change, atmosphere maps."""
import glob, numpy as np, rasterio
from spectral.io import envi
O = "/content/t11"
def prod(case):
    with rasterio.open(glob.glob(f"{O}/{case}/*_L2_SR.tif")[0]) as d:
        wl = np.array([float(d.tags(i + 1)["wavelength"]) for i in range(d.count)])
        x = np.stack([d.read(i + 1) for i in range(d.count)], -1).astype(np.float32)
    x[x == -9999] = np.nan
    return x / 1e4, wl
s50, wl = prod("ang600_s50b"); s200, _ = prod("ang600_s200")
h = "/content/data/t11/ang600/l2/ang20151219t081738_corr_v2m2_img.hdr"
rwl = np.array(envi.read_envi_header(h)["wavelength"], float)
ref = np.asarray(envi.open(h).open_memmap(interleave="bip"), np.float32)[..., [int(np.argmin(abs(rwl - w))) for w in wl]]
ref[(ref < -0.2) | (ref > 2.5) | (ref <= -9990)] = np.nan
core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
L, S, B = s50.shape
P = lambda a: a.reshape(-1, B)
a50, a200, r = P(s50), P(s200), P(ref)
ok = np.all(np.isfinite(a50[:, core]), 1) & np.all(np.isfinite(a200[:, core]), 1) & (np.isfinite(r[:, core]).mean(1) > 0.95)
a50, a200, r = a50[ok], a200[ok], r[ok]
print(f"pixels compared: {ok.sum():,} of {L * S:,}")
b = lambda nm, x: x[:, int(np.argmin(abs(wl - nm)))]
ndvi = (b(860, r) - b(660, r)) / (b(860, r) + b(660, r) + 1e-6)
water = (b(860, r) < 0.05) & (b(560, r) > b(860, r))
cls = np.where(water, "water", np.where(ndvi > 0.5, "veg dense", np.where(ndvi > 0.2, "veg sparse", "bare/built")))
def stats(x, sel, cols):
    d = (x - r)[sel][:, cols]
    return np.nanmean(abs(d)), np.nanmean(d)
print("\n1. core MAE / bias vs JPL by wavelength range (all pixels):")
print(f"{'range nm':>12s} | {'seg50 MAE':>9s} {'bias':>8s} | {'seg200 MAE':>10s} {'bias':>8s}")
for lo, hi in ((460, 550), (550, 700), (700, 900), (900, 1100), (1100, 1340), (1460, 1780), (1990, 2200), (2200, 2450)):
    c = core & (wl >= lo) & (wl < hi)
    m1, b1 = stats(a50, slice(None), c); m2, b2 = stats(a200, slice(None), c)
    print(f"{lo:5d}-{hi:<5d}  | {m1:9.4f} {b1:+8.4f} | {m2:10.4f} {b2:+8.4f}")
m1, b1 = stats(a50, slice(None), core); m2, b2 = stats(a200, slice(None), core)
print(f"{'all core':>12s} | {m1:9.4f} {b1:+8.4f} | {m2:10.4f} {b2:+8.4f}")
print("\n2. core MAE by surface type (NDVI from JPL):")
for k in ("veg dense", "veg sparse", "bare/built", "water"):
    sel = cls == k
    if sel.sum() > 100:
        m1, b1 = stats(a50, sel, core); m2, b2 = stats(a200, sel, core)
        print(f"  {k:10s} {sel.mean():6.1%} | seg50 {m1:.4f} ({b1:+.4f}) | seg200 {m2:.4f} ({b2:+.4f})")
print("\n3. core MAE by brightness quartile (JPL mean core reflectance):")
br = np.nanmean(r[:, core], 1)
qs = np.nanpercentile(br, [0, 25, 50, 75, 100])
for i in range(4):
    sel = (br >= qs[i]) & (br <= qs[i + 1])
    m1, _ = stats(a50, sel, core); m2, _ = stats(a200, sel, core)
    print(f"  Q{i + 1} ({qs[i]:.3f}-{qs[i + 1]:.3f}) | seg50 {m1:.4f} | seg200 {m2:.4f} | relative change {100 * (m2 - m1) / m1:+.1f}%")
def sam(x, y):
    x, y = x[:, core], y[:, core]
    ok2 = np.all(np.isfinite(x), 1) & np.all(np.isfinite(y), 1)
    c = np.sum(x[ok2] * y[ok2], 1) / (np.linalg.norm(x[ok2], axis=1) * np.linalg.norm(y[ok2], axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
s1, s2, s12 = sam(a50, r), sam(a200, r), sam(a200, a50)
print(f"\n4. spectral angle (core bands, degrees): vs JPL seg50 median {np.median(s1):.2f} p95 {np.percentile(s1, 95):.2f} | "
      f"seg200 median {np.median(s2):.2f} p95 {np.percentile(s2, 95):.2f}\n   seg200 vs seg50: median {np.median(s12):.3f} "
      f"p95 {np.percentile(s12, 95):.3f} p99 {np.percentile(s12, 99):.3f} max {s12.max():.2f}")
dm = np.nanmean(abs(a200[:, core] - a50[:, core]), 1)
print(f"\n5. per-pixel mean |seg200 - seg50| over core bands: median {np.median(dm):.4f} p95 {np.percentile(dm, 95):.4f} "
      f"p99 {np.percentile(dm, 99):.4f} | > 0.005: {np.mean(dm > 0.005):.2%} | > 0.01: {np.mean(dm > 0.01):.2%}")
print("\n6. atmosphere maps:")
def atm(case):
    f = [q for q in glob.glob(f"{O}/{case}/_work/isofit/output/*_atm_interp") if "_subs_" not in q][0]
    with rasterio.open(f) as s:
        n = [(x or "").upper() for x in s.descriptions]
        return {k: s.read(n.index(k) + 1).astype(float) for k in ("AOT550", "H2OSTR")}
A50, A200 = atm("ang600_s50b"), atm("ang600_s200")
for k in ("AOT550", "H2OSTR"):
    x, y = A50[k], A200[k]
    v = (x > -9990) & (y > -9990)
    print(f"  {k}: seg50 sd {x[v].std():.4f} | seg200 sd {y[v].std():.4f} | pixel corr {np.corrcoef(x[v], y[v])[0, 1]:.3f} | "
          f"mean |diff| {np.mean(abs(x[v] - y[v])):.4f} p99 {np.percentile(abs(x[v] - y[v]), 99):.4f}")
print("DEEP_DONE")
