#!/usr/bin/env bash
# Item 11 trials on colab-atm (L4 GPU, 12 cores): the all-sensor entrypoint end to end, report with --ref.
#   T1 prs_l4     PRISMA 20260514055637 (training): regression vs AC26-E7 (0.0097 raw / 0.0057 harm) + GPU timing
#   T2 tan_thess  Tanager Thessaloniki 2025-07-12 (AERONET Thessaloniki)       ref: Planet basic SR
#   T3 tan_rocx   Tanager ROCX2025 2025-09-19 (AERONET ROCX2025)               ref: Planet basic SR
#   T4 prs_snow   PRISMA Himachal 2026-03-05 (snow)                            ref: ASI L2D
#   T5 en_coast   EnMAP 056279 Sundarbans 2024-01-02 (coast, winter haze)      ref: DLR L2A
#   T6 tan_flare  Tanager Permian 2025-05-28 (gas flares)                      ref: Planet basic SR
#   T7 ang        AVIRIS-NG ang20151219t081738 full flight line                ref: JPL corr_v2m2
# Usage: t11_run.sh [trial ...]   (default: all, in the order above)
set -u
E=/content/mamba/envs/atm; M=/content/modH; T=/content/data/t11; O=/content/t11
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$M/src:$M
export CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-trial
mkdir -p $O
run() { n=$1; shift; rm -rf $O/$n; s=$(date +%s); bash $M/entrypoint.sh "$@" $O/$n --keep-work "${REF[@]}" > $O/$n.log 2>&1
  echo "== $n exit $? $(( ($(date +%s) - s) / 60 )) min $(date -u +%H:%M)"
  grep -h "Scene:\|Rule \|Per-pixel\|Final reflectance\|ERROR\|WARNING\|report:" $O/$n.log | cut -c1-220 | head -12; }
trial() { case $1 in
  prs_l4)    REF=(--ref "$(ls $T/prs_train/PRS_L2C*.he5)"); run $1 "$(ls $T/prs_train/PRS_L1_*.he5)" ;;
  tan_thess) s=20250712_100036_58_4001; REF=(--ref $T/tan/$s/${s}_basic_sr_hdf5.h5); run $1 $T/tan/$s/${s}_basic_radiance_hdf5.h5 ;;
  tan_rocx)  s=20250919_170238_88_4001; REF=(--ref $T/tan/$s/${s}_basic_sr_hdf5.h5); run $1 $T/tan/$s/${s}_basic_radiance_hdf5.h5 ;;
  prs_snow)  REF=(--ref "$(ls $T/prs_snow/PRS_L2D*.he5)"); run $1 "$(ls $T/prs_snow/PRS_L1_*.he5)" ;;
  en_coast)  REF=(--ref "$(ls -d $T/enmap/ENMAP01-____L2A-*)"); run $1 "$(ls -d $T/enmap/ENMAP01-____L1C-*)" ;;
  tan_flare) s=20250528_180922_74_4001; REF=(--ref $T/tan/$s/${s}_basic_sr_hdf5.h5); run $1 $T/tan/$s/${s}_basic_radiance_hdf5.h5 ;;
  ang)       REF=(--ref $T/ang/l2/ang20151219t081738_corr_v2m2_img.hdr); run $1 $T/ang/ang20151219t081738 ;;
  esac; }
for t in ${@:-prs_l4 tan_thess tan_rocx prs_snow en_coast tan_flare ang}; do trial $t; done
# T1 regression with the AC26-ACC scorer (same numbers as AC26-E7)
K=$O/prs_l4/_work
if [ -f $K/isofit/atm_post.json ]; then
  RD=$(python -c "import json; print(json.load(open('$K/isofit/atm_post.json'))['reflectance'])")
  for v in raw harm; do r=$RD; [ $v = harm ] && r=${RD}_harm
    python /content/pairs_sample.py prisma x $(ls $T/prs_train/PRS_L2C*.he5) $r $(ls $K/PRS_*_obs) $K/isofit/atm_post.json $O/prs_l4_$v.npz 2>&1 \
      | grep "land  *MAE" | head -1 | sed "s/^/  prs_l4 $v /"; done
fi
python /content/time_splits.py $O/*.log 2>/dev/null | cut -f1,15-17
echo T11_DONE
