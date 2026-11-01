"""T7 full line: seg 400 + empirical (ang_fast) vs seg 50 analytical (ang) vs JPL corr, full resolution, pixel index,
read in row blocks (4 Mpix x 380 bands). Plus output checks and stage times."""
import glob, json, re, numpy as np, rasterio
from spectral.io import envi
O = "/content/t11"
f_new, f_old = glob.glob(f"{O}/ang_fast/*_L2_SR.tif")[0], glob.glob(f"{O}/ang/*_L2_SR.tif")[0]
h = "/content/data/t11/ang/l2/ang20151219t081738_corr_v2m2_img.hdr"
ref = envi.open(h).open_memmap(interleave="bip")
rwl = np.array(envi.read_envi_header(h)["wavelength"], float)
with rasterio.open(f_new) as a, rasterio.open(f_old) as b:
    wl = np.array([float(a.tags(i + 1)["wavelength"]) for i in range(a.count)])
    ri = [int(np.argmin(abs(rwl - w))) for w in wl]
    core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
    blue = core & (wl < 550)
    acc = {k: [] for k in ("mae_new", "mae_old", "b_new", "b_old", "blue_new", "blue_old", "sam_new", "sam_old", "sam_no", "d_no", "dark")}
    L = a.height
    for r0 in range(0, L, 4 * 128):   # every 4th block of 128 rows: ~1 Mpix sampled across the whole line
        r1 = min(L, r0 + 128)
        w = ((r0, r1), (0, a.width))
        x = np.stack([a.read(i + 1, window=w) for i in range(a.count)], -1).reshape(-1, a.count).astype(np.float32)
        y = np.stack([b.read(i + 1, window=w) for i in range(b.count)], -1).reshape(-1, b.count).astype(np.float32)
        z = np.asarray(ref[r0:r1], np.float32)[..., ri].reshape(-1, len(ri))
        x[x == -9999] = np.nan; y[y == -9999] = np.nan
        x, y = x / 1e4, y / 1e4
        z[(z < -0.2) | (z > 2.5)] = np.nan
        ok = np.all(np.isfinite(x[:, core]), 1) & np.all(np.isfinite(y[:, core]), 1) & (np.isfinite(z[:, core]).mean(1) > 0.95)
        X, Y, Z = x[ok][:, core], y[ok][:, core], np.nan_to_num(z[ok][:, core])
        sam = lambda p, q: np.degrees(np.arccos(np.clip(np.sum(p * q, 1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-12), -1, 1)))
        acc["mae_new"].append(np.nanmean(abs(X - Z), 1)); acc["mae_old"].append(np.nanmean(abs(Y - Z), 1))
        acc["b_new"].append(np.nanmean(X - Z, 1)); acc["b_old"].append(np.nanmean(Y - Z, 1))
        acc["blue_new"].append(np.nanmean(abs(x[ok][:, blue] - np.nan_to_num(z[ok][:, blue])), 1))
        acc["blue_old"].append(np.nanmean(abs(y[ok][:, blue] - np.nan_to_num(z[ok][:, blue])), 1))
        acc["sam_new"].append(sam(X, Z)); acc["sam_old"].append(sam(Y, Z)); acc["sam_no"].append(sam(X, Y))
        acc["d_no"].append(np.nanmean(abs(X - Y), 1)); acc["dark"].append(Z.mean(1))
    A = {k: np.concatenate(v) for k, v in acc.items()}
print(f"\nfull line, {A['mae_new'].size:,} sampled px (every 4th 128-row block), core bands, vs JPL by pixel index:")
for lab, s in (("seg 50 analytical (T7)", "old"), ("seg 400 empirical (new default)", "new")):
    print(f"  {lab:32s} MAE {A['mae_' + s].mean():.4f} bias {A['b_' + s].mean():+.4f} | 460-550 {A['blue_' + s].mean():.4f} | "
          f"angle med {np.median(A['sam_' + s]):.2f} p95 {np.percentile(A['sam_' + s], 95):.2f} p99 {np.percentile(A['sam_' + s], 99):.2f} p99.9 {np.percentile(A['sam_' + s], 99.9):.2f}")
d, s = A["d_no"], A["sam_no"]
print(f"  new vs old per pixel: mean|diff| median {np.median(d):.4f} p99 {np.percentile(d, 99):.4f} >0.005 {np.mean(d > 0.005):.2%} >0.01 {np.mean(d > 0.01):.2%} | angle med {np.median(s):.2f} p99 {np.percentile(s, 99):.2f}")
dk = A["dark"] < np.percentile(A["dark"], 5)
print(f"  darkest 5%: angle vs JPL old {np.median(A['sam_old'][dk]):.2f} new {np.median(A['sam_new'][dk]):.2f} | rest old {np.median(A['sam_old'][~dk]):.2f} new {np.median(A['sam_new'][~dk]):.2f}")
p = json.load(open(glob.glob(f"{O}/ang_fast/*_provenance.json")[0]))
print("\noutputs:", sorted(x.split("/")[-1] for x in p["product"].values()), "| bands", p["bands_kept"], "| georef", p["georeference"],
      "| write s", p["write_times_s"], "| atm", {k: round(p["atm_post"][k], 3) for k in ("scene_aot550", "scene_h2o", "scene_elev_km")})
with rasterio.open(f_new) as a, rasterio.open(glob.glob("/content/data/t11/ang/ang20151219t081738/*_rdn_v2m2_img.bin")[0]) as r:
    print("transform equals source:", a.transform == r.transform, "| crs", a.crs)
page = open(glob.glob(f"{O}/ang_fast/*_report.html")[0]).read()
print("report:", page.count("<img"), "imgs,", page.count("Not available"), "not available | MAE", re.search(r"MAE \(all bands\)</td><td>([^<]*)", page).group(1))
st = [l.split() for l in open(f"{O}/ang_fast/_work/stage_times.txt")]
print("stages:", " | ".join(f"{x} {(int(c) - int(bb)) / 60:.1f} min" for (x, bb), (_, c) in zip(st, st[1:])))
