"""T7 full line: split the spectral-angle tail by surface (JPL reflectance): water (NIR860 < 0.05 and green > NIR),
dark land (mean core < 0.06), other land. Old = seg 50 analytical, new = seg 400 empirical."""
import glob, numpy as np, rasterio
from spectral.io import envi
O = "/content/t11"
h = "/content/data/t11/ang/l2/ang20151219t081738_corr_v2m2_img.hdr"
ref = envi.open(h).open_memmap(interleave="bip"); rwl = np.array(envi.read_envi_header(h)["wavelength"], float)
a, b = rasterio.open(glob.glob(f"{O}/ang_fast/*_L2_SR.tif")[0]), rasterio.open(glob.glob(f"{O}/ang/*_L2_SR.tif")[0])
wl = np.array([float(a.tags(i + 1)["wavelength"]) for i in range(a.count)]); ri = [int(np.argmin(abs(rwl - w))) for w in wl]
core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
i860, i560 = int(np.argmin(abs(wl - 860))), int(np.argmin(abs(wl - 560)))
sam = lambda p, q: np.degrees(np.arccos(np.clip(np.sum(p * q, 1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-12), -1, 1)))
cls, so, sn, mo, mn, rows = [], [], [], [], [], []
for r0 in range(0, a.height, 4 * 128):
    r1 = min(a.height, r0 + 128); w = ((r0, r1), (0, a.width))
    x = np.stack([a.read(i + 1, window=w) for i in range(a.count)], -1).reshape(-1, a.count).astype(np.float32)
    y = np.stack([b.read(i + 1, window=w) for i in range(b.count)], -1).reshape(-1, b.count).astype(np.float32)
    z = np.asarray(ref[r0:r1], np.float32)[..., ri].reshape(-1, len(ri))
    x[x == -9999] = np.nan; y[y == -9999] = np.nan; x, y = x / 1e4, y / 1e4
    z[(z < -0.2) | (z > 2.5)] = np.nan
    ok = np.all(np.isfinite(x[:, core]), 1) & np.all(np.isfinite(y[:, core]), 1) & (np.isfinite(z[:, core]).mean(1) > 0.95)
    X, Y, Z = x[ok][:, core], y[ok][:, core], np.nan_to_num(z[ok][:, core])
    zn, zg, zm = z[ok][:, i860], z[ok][:, i560], np.nanmean(z[ok][:, core], 1)
    cls.append(np.where((zn < 0.05) & (zg > zn), 0, np.where(zm < 0.06, 1, 2)))
    so.append(sam(Y, Z)); sn.append(sam(X, Z)); mo.append(np.nanmean(abs(Y - Z), 1)); mn.append(np.nanmean(abs(X - Z), 1))
    rows.append(np.repeat(np.arange(r0, r1), a.width)[ok])
C, SO, SN, MO, MN, R = map(np.concatenate, (cls, so, sn, mo, mn, rows))
print(f"pixels {C.size:,}")
for k, name in ((0, "water"), (1, "dark land (<0.06)"), (2, "other land")):
    s = C == k
    if s.sum():
        print(f"  {name:18s} {s.mean():6.2%} | angle vs JPL med old {np.median(SO[s]):5.2f} new {np.median(SN[s]):5.2f} | p99 old {np.percentile(SO[s], 99):5.2f} new {np.percentile(SN[s], 99):5.2f} | MAE old {MO[s].mean():.4f} new {MN[s].mean():.4f}")
t = SN > np.percentile(SN, 99)
print(f"  top-1% angle (new): water {np.mean(C[t] == 0):.1%} dark land {np.mean(C[t] == 1):.1%} other {np.mean(C[t] == 2):.1%} | MAE there old {MO[t].mean():.4f} new {MN[t].mean():.4f} | rows p5-p95 {np.percentile(R[t], 5):.0f}-{np.percentile(R[t], 95):.0f}")
