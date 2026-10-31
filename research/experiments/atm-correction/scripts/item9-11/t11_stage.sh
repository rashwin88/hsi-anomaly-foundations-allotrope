#!/usr/bin/env bash
# Item 11: stage the trial inputs (local copies) and their Copernicus DEM tiles into /content/dem_all.
set -x
E=/content/mamba/envs/atm; export PATH=$E/bin:$PATH
M=/content/modH; T=/content/data/t11; mkdir -p $T /content/dem_all
P=/content/drive/MyDrive/PS_11; PT=/content/drive/.shortcut-targets-by-id/1XhTlJ-NSfjnZjwUIDWkzoEwI9PDo9PN2/PRISMA_Training_Set
TS=$P/tanager_datasets/scenes
cpy() { [ -e "$2/$(basename $1)" ] || cp -r "$1" "$2/"; }
mkdir -p $T/prs_train $T/prs_snow $T/enmap $T/tan $T/ang
cpy "$(ls $PT/20260514055637/PRS_L1_STD*.he5)" $T/prs_train; cpy "$(ls $PT/20260514055637/PRS_L2C_STD*.he5)" $T/prs_train
S3=$P/stage3_datasets/set3_20260906
cpy "$(ls $S3/PRS_L1_STD_OFFL_20260305054623_*/*.he5 | head -1)" $T/prs_snow
cpy "$(ls $S3/PRS_L2D_STD_20260305054623_*/*.he5)" $T/prs_snow
cpy "$(ls -d $P/EnMAP_datasets/enmap_L1C/ENMAP01-____L1C-DT0000056279_*)" $T/enmap
cpy "$(ls -d $P/EnMAP_datasets/stage_3/ENMAP01-____L2A-DT0000056279_*)" $T/enmap
for s in 20250712_100036_58_4001 20250919_170238_88_4001 20250528_180922_74_4001; do
  mkdir -p $T/tan/$s; for k in basic_radiance_hdf5 basic_sr_hdf5; do cpy $TS/$s/${s}_$k.h5 $T/tan/$s; done; cpy $TS/$s/${s}_stac_item.json $T/tan/$s
done
cpy $P/aviris_datasets_L1/ang20151219t081738 $T/ang
mkdir -p $T/ang/l2; for f in $P/aviris_datasets/ang20151219t081738/ang20151219t081738_corr_v2m2_img*; do cpy $f $T/ang/l2; done
bb() { python - "$1" <<'PY'
import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r"); la = lo = None
def walk(n, o):
    global la, lo
    if isinstance(o, h5py.Dataset) and n.endswith("Geolocation Fields/Latitude") and la is None: la = o[::20, ::20]
    if isinstance(o, h5py.Dataset) and n.endswith("Geolocation Fields/Longitude") and lo is None: lo = o[::20, ::20]
f.visititems(walk)
ok = (la > -90) & (lo > -180) & (la != 0)
print(int(np.floor(lo[ok].min())), int(np.floor(la[ok].min())), int(np.floor(lo[ok].max())), int(np.floor(la[ok].max())))
PY
}
SD=$M/scripts/stage_dem.sh
for h in $T/prs_train/PRS_L2C*.he5 $T/prs_snow/PRS_L1_*.he5 $T/tan/*/*_basic_radiance_hdf5.h5; do
  bash $SD bbox $(bb $h) /content/dem_all >> /content/t11_dem.log 2>&1 || echo "DEM FAIL $h"
done
bash $SD scene $T/enmap/ENMAP01-____L1C-DT0000056279_* /content/dem_all >> /content/t11_dem.log 2>&1 || echo "DEM FAIL enmap"
ls /content/dem_all | wc -l; gdalinfo /content/dem_all/cop30_mosaic.vrt | grep -A1 "Size is\|Upper Left\|Lower Right"
du -sh $T/*
echo STAGE_DONE
