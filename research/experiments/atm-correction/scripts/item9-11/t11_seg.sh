#!/usr/bin/env bash
# AVIRIS-NG 600-line window (rows 2496-3096): --segmentation-size 50 vs 100, after the T11 queue ends.
E=/content/mamba/envs/atm; M=/content/modS; T=/content/data/t11; O=/content/t11
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -W ignore /content/ang_crop.py
until grep -q T11_DONE /content/t11_run.log; do sleep 60; done
rm -rf $M && cp -r /content/modH $M && cp /content/entrypoint_seg.sh $M/entrypoint.sh && chmod +x $M/entrypoint.sh
export PYTHONPATH=$M/src:$M CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-seg
for s in 50 100; do
  rm -rf $O/ang600_s$s; t=$(date +%s)
  bash $M/entrypoint.sh $T/ang600/l1/ang20151219t081738 $O/ang600_s$s --keep-work --segmentation-size $s \
    --ref $T/ang600/l2/ang20151219t081738_corr_v2m2_img.hdr > $O/ang600_s$s.log 2>&1
  echo "== ang600_s$s exit $? $(( ($(date +%s) - t) / 60 )) min $(date -u +%H:%M)"
done
python -W ignore /content/seg_cmp.py
echo SEG_DONE
