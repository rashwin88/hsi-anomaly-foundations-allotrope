import glob, numpy as np, rasterio
from spectral.io import envi
sys_path = None
o = rasterio.open(glob.glob("/content/t11/ang/*_L2_SR.tif")[0])
wl = np.array([float(o.tags(i + 1)["wavelength"]) for i in range(o.count)])
ref = envi.open("/content/data/t11/ang/l2/ang20151219t081738_corr_v2m2_img.hdr").open_memmap(interleave="bip")
rwl = np.array(envi.read_envi_header("/content/data/t11/ang/l2/ang20151219t081738_corr_v2m2_img.hdr")["wavelength"], float)
ri = [int(np.argmin(abs(rwl - w))) for w in wl]
core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
rows = np.arange(100, 5500, 60); cols = np.arange(10, 712, 8)
ours = o.read(window=((0, 1), (0, 1)))  # warm
res = {}
for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
        d = []
        for r in rows[::3]:
            a = o.read(window=((r, r + 1), (0, o.width)))[:, 0, cols].T / 1e4
            b = np.asarray(ref[r + dr, cols + dc, :], float)[:, ri]
            ok = (a > -0.5).all(1) & (b > -0.2).all(1) & (b < 2.5).all(1)
            d.append(np.abs(a[ok][:, core] - b[ok][:, core]))
        res[(dr, dc)] = float(np.mean(np.concatenate(d)))
for k, v in sorted(res.items(), key=lambda x: x[1]):
    print("row/col shift", k, "core MAE", round(v, 4))
print("ours transform", o.transform, o.crs)
with rasterio.open("/content/data/t11/ang/l2/ang20151219t081738_corr_v2m2_img.bin") as d:
    print("corr transform", d.transform)
with rasterio.open(glob.glob("/content/data/t11/ang/ang20151219t081738/*_rdn_v2m2_img.bin")[0]) as d:
    print("rdn  transform", d.transform)
