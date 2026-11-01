#!/usr/bin/env bash
# T7 again with the pushed module (1e6885d): AVIRIS-NG full line on the sensor rule's defaults (seg 400 + empirical).
E=/content/mamba/envs/atm; M=/content/modW; T=/content/data/t11; O=/content/t11
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$M/src:$M
export CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-1e6885d
rm -rf $O/ang_fast; s=$(date +%s)
bash $M/entrypoint.sh $T/ang/ang20151219t081738 $O/ang_fast --keep-work --ref $T/ang/l2/ang20151219t081738_corr_v2m2_img.hdr > $O/ang_fast.log 2>&1
echo "== ang_fast exit $? $(( ($(date +%s) - s) / 60 )) min $(date -u +%H:%M)"
grep -h "Rule avirisng\|Per-pixel\|WARNING\|ERROR" $O/ang_fast.log | grep -v "Token auth\|surface model" | cut -c1-200 | head -8
grep -h "Beginning\|inversions complete\|line inference\|line inversions complete" $O/ang_fast.log | cut -c1-110
python -W ignore /content/t7b_cmp.py
echo T7B_DONE
