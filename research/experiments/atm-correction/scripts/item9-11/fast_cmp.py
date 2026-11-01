"""600-line window: seg 50 analytical / seg 200 empirical / seg 400 empirical / seg 200 empirical no-presolve, all vs
JPL corr at full resolution by pixel index; each vs seg 50; ISOFIT stage times; atmosphere."""
import glob, re, numpy as np, rasterio
from spectral.io import envi
O = "/content/t11"
CASES = [("ang600_s50b", "seg 50 analytical"), ("ang600_s200_el", "seg 200 empirical"),
         ("ang600_s400_el", "seg 400 empirical"), ("ang600_s200_el_nops", "seg 200 empirical, no presolve")]
def prod(c):
    with rasterio.open(glob.glob(f"{O}/{c}/*_L2_SR.tif")[0]) as d:
        wl = np.array([float(d.tags(i + 1)["wavelength"]) for i in range(d.count)])
        x = np.stack([d.read(i + 1) for i in range(d.count)], -1).astype(np.float32)
    x[x == -9999] = np.nan
    return x.reshape(-1, len(wl)) / 1e4, wl
P = {}
for c, _ in CASES:
    P[c], wl = prod(c)
h = "/content/data/t11/ang600/l2/ang20151219t081738_corr_v2m2_img.hdr"
rwl = np.array(envi.read_envi_header(h)["wavelength"], float)
r = np.asarray(envi.open(h).open_memmap(interleave="bip"), np.float32)[..., [int(np.argmin(abs(rwl - w))) for w in wl]].reshape(-1, len(wl))
r[(r < -0.2) | (r > 2.5)] = np.nan
core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
blue = core & (wl < 550)
ok = (np.isfinite(r[:, core]).mean(1) > 0.95)
for c, _ in CASES:
    ok &= np.all(np.isfinite(P[c][:, core]), 1)
rr = np.nan_to_num(r[ok][:, core])
def sam(x, y):
    cth = np.sum(x * y, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cth, -1, 1)))
base = P["ang600_s50b"][ok]
print(f"pixels compared: {ok.sum():,}")
print(f"{'case':32s} {'ISOFIT min':>10s} {'core MAE':>9s} {'bias':>8s} {'460-550':>8s} {'SAM med':>8s} | vs seg50: {'med':>6s} {'p99':>6s} {'>0.005':>7s} {'SAM p99':>7s}")
for c, label in CASES:
    x = P[c][ok]
    d = (x - r[ok])[:, core]
    st = [l.split() for l in open(f"{O}/{c}/_work/stage_times.txt")]
    iso = next((int(b2) - int(b1)) / 60 for (a, b1), (_, b2) in zip(st, st[1:]) if a == "isofit")
    dm = np.nanmean(abs(x[:, core] - base[:, core]), 1)
    s12 = sam(x[:, core], base[:, core])
    print(f"{label:32s} {iso:10.1f} {np.nanmean(abs(d)):9.4f} {np.nanmean(d):+8.4f} {np.nanmean(abs((x - r[ok])[:, blue])):8.4f} "
          f"{np.median(sam(x[:, core], rr)):8.2f} | {np.median(dm):6.4f} {np.percentile(dm, 99):6.4f} {np.mean(dm > 0.005):7.2%} {np.percentile(s12, 99):7.2f}")
    log = open(f"{O}/{c}.log").read()
    ts = re.findall(r"INFO:2026-\d\d-\d\d,(\d\d:\d\d:\d\d) \|\| \S+ \| (Beginning \d+ inversions|Pixel class: uniform_surface inversions complete|Empirical line inference|Analytical line inference)", log)
    print("    " + " ; ".join(f"{t} {s[:24]}" for t, s in ts))
for c, label in CASES:
    f = sorted(glob.glob(f"{O}/{c}/_work/isofit/output/*_atm_interp")) or sorted(glob.glob(f"{O}/{c}/_work/isofit/output/*_subs_state"))
    with rasterio.open(f[0]) as s:
        n = [(q or "").upper() for q in s.descriptions]
        out = []
        for k in ("AOT550", "H2OSTR"):
            v = s.read(n.index(k) + 1).ravel()
            v = v[v > -9990]
            out.append(f"{k} median {np.median(v):.3f} p5-p95 {np.percentile(v, 5):.3f}-{np.percentile(v, 95):.3f}")
    print(f"{label:32s} " + " | ".join(out))
