#!/usr/bin/env bash
# 1 calibration window to Drive, 2 tests, 3 seg 400 (rule default: empirical), 4 seg 200 no presolve (empirical),
# 5 compare with ang600_s200_el (seg 200 empirical, presolve) and ang600_s50b (seg 50 analytical) vs JPL.
E=/content/mamba/envs/atm; M=/content/modV; T=/content/data/t11; O=/content/t11
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/content/drive/MyDrive/PS_11/atm_trial_scenes/aviris_ng_600line_calibration
mkdir -p $D && cp -r $T/ang600/l1 $D/input && cp -r $T/ang600/l2 $D/reference && du -sh $D && echo CALIB_COPIED
rm -rf $M && mkdir -p $M && tar xzf /content/modV.tgz -C $M && (cd $M && python -m pytest -p no:cacheprovider 2>&1 | tail -2)
export PYTHONPATH=$M/src:$M CONFIG_ROOT=$M/configs DEM_VRT=/content/dem_all/cop30_mosaic.vrt N_CORES=12 PIPELINE_VERSION=1.3.0-fast
for c in "ang600_s400_el --segmentation-size 400" "ang600_s200_el_nops --no-presolve"; do set -- $c; n=$1; shift
  rm -rf $O/$n; t=$(date +%s)
  bash $M/entrypoint.sh $T/ang600/l1/ang20151219t081738 $O/$n --keep-work --no-report "$@" > $O/$n.log 2>&1
  echo "== $n exit $? $(( ($(date +%s) - t) / 60 )) min $(date -u +%H:%M)"; grep -h "Rule avirisng" $O/$n.log | cut -c1-200
done
python -W ignore /content/fast_cmp.py
echo FAST_DONE
