#!/usr/bin/env bash
E=/content/mamba/envs/atm; M=/content/modV; T=/content/data/t11; O=/content/t11
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$M/src:$M
export CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-s400an
rm -rf $O/ang600_s400; t=$(date +%s)
bash $M/entrypoint.sh $T/ang600/l1/ang20151219t081738 $O/ang600_s400 --keep-work --no-report --segmentation-size 400 --line analytical > $O/ang600_s400.log 2>&1
echo "== ang600_s400 exit $? $(( ($(date +%s) - t) / 60 )) min $(date -u +%H:%M)"
grep -h "Rule avirisng" $O/ang600_s400.log | cut -c1-200
sed -i 's#("ang600_s400_el", "seg 400 empirical"), #("ang600_s400_el", "seg 400 empirical"), ("ang600_s400", "seg 400 analytical"), #' /content/fast_cmp.py
python -W ignore /content/fast_cmp.py | grep -v "^    "
python -W ignore - <<'PY'
import glob, numpy as np, rasterio
from spectral.io import envi
O = "/content/t11"
def prod(c):
    with rasterio.open(glob.glob(f"{O}/{c}/*_L2_SR.tif")[0]) as d:
        wl = np.array([float(d.tags(i + 1)["wavelength"]) for i in range(d.count)])
        x = np.stack([d.read(i + 1) for i in range(d.count)], -1).astype(np.float32)
    x[x == -9999] = np.nan
    return x.reshape(-1, len(wl)) / 1e4, wl
x, wl = prod("ang600_s400")
h = "/content/data/t11/ang600/l2/ang20151219t081738_corr_v2m2_img.hdr"
rwl = np.array(envi.read_envi_header(h)["wavelength"], float)
r = np.asarray(envi.open(h).open_memmap(interleave="bip"), np.float32)[..., [int(np.argmin(abs(rwl - w))) for w in wl]].reshape(-1, len(wl))
r[(r < -0.2) | (r > 2.5)] = np.nan
core = ((wl > 460) & (wl < 1340)) | ((wl > 1460) & (wl < 1780)) | ((wl > 1990) & (wl < 2450))
ok = (np.isfinite(r[:, core]).mean(1) > 0.95) & np.all(np.isfinite(x[:, core]), 1)
X, R = x[ok][:, core], np.nan_to_num(r[ok][:, core])
c = np.sum(X * R, 1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(R, axis=1) + 1e-12)
s = np.degrees(np.arccos(np.clip(c, -1, 1)))
print(f"seg400 analytical vs JPL angle: median {np.median(s):.2f} p95 {np.percentile(s, 95):.2f} p99 {np.percentile(s, 99):.2f} p99.9 {np.percentile(s, 99.9):.2f}")
PY
echo S400AN_DONE
