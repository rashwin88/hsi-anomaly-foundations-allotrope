#!/usr/bin/env bash
# AVIRIS-NG 600-line window: seg 200, then seg 50 again (clean timing, nothing else running), then compare 50/100/200.
E=/content/mamba/envs/atm; M=/content/modS; T=/content/data/t11; O=/content/t11
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$M/src:$M CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-seg
for c in "200 ang600_s200" "50 ang600_s50b"; do set -- $c
  rm -rf $O/$2; t=$(date +%s)
  bash $M/entrypoint.sh $T/ang600/l1/ang20151219t081738 $O/$2 --keep-work --segmentation-size $1 --no-report > $O/$2.log 2>&1
  echo "== $2 exit $? $(( ($(date +%s) - t) / 60 )) min $(date -u +%H:%M)"
done
python -W ignore /content/seg_cmp3.py
echo SEG2_DONE
