"""Empirical-line tail: which pixels change spectral shape vs the analytical line, and is that closer to or farther
from JPL? Same 600-line window, full resolution, pixel index."""
import glob, numpy as np, rasterio
from spectral.io import envi
O = "/content/t11"
def prod(c):
    with rasterio.open(glob.glob(f"{O}/{c}/*_L2_SR.tif")[0]) as d:
        wl = np.array([float(d.tags(i + 1)["wavelength"]) for i in range(d.count)])
        x = np.stack([d.read(i + 1) for i in range(d.count)], -1).astype(np.float32)
    x[x == -9999] = np.nan
    return x.reshape(-1, len(wl)) / 1e4, wl
an, wl = prod("ang600_s200"); el, _ = prod("ang600_s200_el"); e4, _ = prod("ang600_s400_el"); a50, _ = prod("ang600_s50b")
h = "/content/data/t11/ang600/l2/ang20151219t081738_corr_v2m2_img.hdr"
rwl = np.array(envi.read_envi_header(h)["wavelength"], float)
r = np.asarray(envi.open(h).open_memmap(interleave="bip"), np.float32)[..., [int(np.argmin(abs(rwl - w))) for w in wl]].reshape(-1, len(wl))
r[(r < -0.2) | (r > 2.5)] = np.nan
core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
ok = (np.isfinite(r[:, core]).mean(1) > 0.95) & np.all(np.isfinite(an[:, core]), 1) & np.all(np.isfinite(el[:, core]), 1) & np.all(np.isfinite(e4[:, core]), 1) & np.all(np.isfinite(a50[:, core]), 1)
R = np.nan_to_num(r[ok][:, core])
def sam(x, y):
    c = np.sum(x * y, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
A, E, E4, A50 = an[ok][:, core], el[ok][:, core], e4[ok][:, core], a50[ok][:, core]
print(f"pixels {ok.sum():,}")
for name, x in (("seg50 analytical", A50), ("seg200 analytical", A), ("seg200 empirical", E), ("seg400 empirical", E4)):
    s = sam(x, R)
    print(f"  {name:18s} vs JPL angle: median {np.median(s):.2f} p95 {np.percentile(s, 95):.2f} p99 {np.percentile(s, 99):.2f} p99.9 {np.percentile(s, 99.9):.2f}")
d = sam(E, A)
tail = d > np.percentile(d, 99)
mean = R.mean(1)
b = lambda nm: R[:, int(np.argmin(abs(wl[core] - nm)))]
ndvi = (b(860) - b(660)) / (b(860) + b(660) + 1e-6)
print(f"\nseg200 empirical vs analytical angle: median {np.median(d):.2f} p99 {np.percentile(d, 99):.2f} (tail = top 1%, {tail.sum():,} px)")
print(f"  tail pixels: JPL mean core reflectance median {np.median(mean[tail]):.3f} (all {np.median(mean):.3f}) | "
      f"darkest 5% share {np.mean(mean[tail] < np.percentile(mean, 5)):.1%} | NDVI median {np.median(ndvi[tail]):.2f} (all {np.median(ndvi):.2f})")
sa, se = sam(A, R)[tail], sam(E, R)[tail]
print(f"  on the tail, angle vs JPL: analytical median {np.median(sa):.2f} | empirical median {np.median(se):.2f} | "
      f"empirical closer to JPL on {np.mean(se < sa):.1%} of them")
L, S = 600, 722
rows = np.flatnonzero(ok)[tail] // S
print(f"  tail rows spread: p5 {np.percentile(rows, 5):.0f} p50 {np.percentile(rows, 50):.0f} p95 {np.percentile(rows, 95):.0f} | "
      f"cols p5 {np.percentile(np.flatnonzero(ok)[tail] % S, 5):.0f} p95 {np.percentile(np.flatnonzero(ok)[tail] % S, 95):.0f}")
