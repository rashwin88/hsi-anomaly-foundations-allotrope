#!/usr/bin/env bash
# AVIRIS-NG 600-line window: rule default (seg 200) with --line empirical vs analytical (ang600_s200), after the pip T1.
cd /content/modU && python -m pytest -p no:cacheprovider 2>&1 | tail -2; cd /content
E=/content/mamba/envs/atm; M=/content/modU; T=/content/data/t11; O=/content/t11
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$M/src:$M
export CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-el
rm -rf $O/ang600_s200_el; t=$(date +%s)
bash $M/entrypoint.sh $T/ang600/l1/ang20151219t081738 $O/ang600_s200_el --keep-work --line empirical --no-report > $O/ang600_s200_el.log 2>&1
echo "== ang600_s200_el exit $? $(( ($(date +%s) - t) / 60 )) min $(date -u +%H:%M)"
grep -h "superpixels of\|Empirical\|empirical" $O/ang600_s200_el.log | cut -c1-150 | head -5
python -W ignore - <<'PY'
import glob, numpy as np, rasterio
from spectral.io import envi
def prod(c):
    with rasterio.open(glob.glob(f"/content/t11/{c}/*_L2_SR.tif")[0]) as d:
        wl = np.array([float(d.tags(i + 1)["wavelength"]) for i in range(d.count)])
        x = np.stack([d.read(i + 1) for i in range(d.count)], -1).astype(np.float32)
    x[x == -9999] = np.nan
    return x.reshape(-1, len(wl)) / 1e4, wl
an, wl = prod("ang600_s200"); el, _ = prod("ang600_s200_el")
h = "/content/data/t11/ang600/l2/ang20151219t081738_corr_v2m2_img.hdr"
rwl = np.array(envi.read_envi_header(h)["wavelength"], float)
r = np.asarray(envi.open(h).open_memmap(interleave="bip"), np.float32)[..., [int(np.argmin(abs(rwl - w))) for w in wl]].reshape(-1, len(wl))
r[(r < -0.2) | (r > 2.5)] = np.nan
core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
ok = np.all(np.isfinite(an[:, core]), 1) & np.all(np.isfinite(el[:, core]), 1) & (np.isfinite(r[:, core]).mean(1) > 0.95)
for name, x in (("analytical", an), ("empirical", el)):
    d = (x - r)[ok][:, core]
    c = np.sum(x[ok][:, core] * np.nan_to_num(r[ok][:, core]), 1) / (np.linalg.norm(x[ok][:, core], axis=1) * np.linalg.norm(np.nan_to_num(r[ok][:, core]), axis=1) + 1e-12)
    print(f"seg200 {name:10s}: core MAE vs JPL {np.nanmean(abs(d)):.4f} bias {np.nanmean(d):+.4f} | SAM median {np.degrees(np.median(np.arccos(np.clip(c, -1, 1)))):.2f} deg")
dm = np.nanmean(abs(el[ok][:, core] - an[ok][:, core]), 1)
print(f"empirical vs analytical per pixel: median {np.median(dm):.4f} p99 {np.percentile(dm, 99):.4f} > 0.005 {np.mean(dm > 0.005):.2%} > 0.01 {np.mean(dm > 0.01):.2%}")
for c in ("ang600_s200", "ang600_s200_el"):
    st = [l.split() for l in open(f"/content/t11/{c}/_work/stage_times.txt")]
    print(c, " | ".join(f"{a} {(int(b2) - int(b1)) / 60:.1f}" for (a, b1), (_, b2) in zip(st, st[1:])))
PY
echo EL_DONE
