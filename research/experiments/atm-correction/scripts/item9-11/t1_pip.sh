#!/usr/bin/env bash
# T1 PRISMA in the clean pip-only venv (what the image ships): patch ISOFIT inside the venv as the Dockerfile does,
# run the same entrypoint (fixed module modT), score raw / harmonised, and diff against the conda run (t11/prs_l4).
until grep -q SEG2_DONE /content/t11_seg2.log; do sleep 60; done
V=/content/venv312; M=/content/modT; T=/content/data/t11; O=/content/t11
export PATH=$V/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$M/src:$M
python /opt/build/isofit_patches.py
EL=$V/lib/python3.12/site-packages/isofit/utils/empirical_line.py
sed -i "s/\(output_reflectance_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/; s/\(output_uncertainty_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/" "$EL"
python -c "import importlib, fast_al, isofit, torch; A = importlib.import_module('isofit.utils.apply_oe'); assert A.ALAlg is fast_al.analytical_line; print('pip venv: fast_al wired, isofit', isofit.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
export CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-pip
rm -rf $O/prs_pip; s=$(date +%s)
bash $M/entrypoint.sh "$(ls $T/prs_train/PRS_L1_*.he5)" $O/prs_pip --keep-work --ref "$(ls $T/prs_train/PRS_L2C*.he5)" > $O/prs_pip.log 2>&1
echo "== prs_pip exit $? $(( ($(date +%s) - s) / 60 )) min $(date -u +%H:%M)"
K=$O/prs_pip/_work
RD=$(python -c "import json; print(json.load(open('$K/isofit/atm_post.json'))['reflectance'])")
for v in raw harm; do r=$RD; [ $v = harm ] && r=${RD}_harm
  /content/mamba/envs/atm/bin/python /content/pairs_sample.py prisma x $(ls $T/prs_train/PRS_L2C*.he5) $r $(ls $K/PRS_*_obs) $K/isofit/atm_post.json /tmp/pp_$v.npz 2>&1 | grep "land  *MAE" | head -1 | sed "s/^/pip $v /"; done
python - <<'PY'
import glob, numpy as np, rasterio
a = rasterio.open(glob.glob("/content/t11/prs_l4/*_L2_SR.tif")[0]).read().astype(np.int32)
b = rasterio.open(glob.glob("/content/t11/prs_pip/*_L2_SR.tif")[0]).read().astype(np.int32)
ok = (a != -9999) & (b != -9999)
d = np.abs(a - b)[ok]
print(f"pip vs conda product (int16 x1e4): identical {np.mean(d == 0):.4%} of values, max |diff| {d.max()} ({d.max() / 1e4:.4f} refl), "
      f"mean |diff| {d.mean() / 1e4:.6f}, nodata agree {np.mean((a == -9999) == (b == -9999)):.4%}")
PY
echo PIP_DONE
