#!/usr/bin/env bash
# Item 9 test: report.py on the item 8 outputs (/content/t8/{w,out}), light env.
#   enmap_l2a  + fake ISOFIT atm_interp, atm_post record, outlier/snow sidecars, stage times; ref = the same L2A
#              folder (self-comparison: MAE must be ~0, pairing by location on the same UTM grid)
#   ang        ref = AVIRIS-NG L2 corr_v2m2 (rotated grid, location pairing; values are rdn vs rfl, pairing only)
#   prisma     ref = PRISMA L2D .he5 (map-projected vs swath: location pairing via _latlon.tif)
#   tan_ortho  ref = Tanager ortho SR .h5
#   nometa     no location anywhere, no ref
set -u
M=/content/t8/mod; export PYTHONPATH=$M/src
D=/content/drive/MyDrive/PS_11; W=/content/t8/w; O=/content/t8/out
python -W ignore /content/t8/t9_fake.py $W/enmap_l2a $O/enmap_l2a
run() { local c=$1; shift; local t=$SECONDS; python -W ignore -m report --out $O/$c --work $W/$c "$@" 2>&1 | grep -v "^$" | tail -4; echo "$c report $((SECONDS - t)) s"; }
run enmap_l2a --ref $D/EnMAP_datasets/stage_3/ENMAP01-____L2A-DT0000056279_20240102T052511Z_001_V010502_20260127T164258Z
run ang --ref $D/aviris_datasets/ang20151219t081738/ang20151219t081738_corr_v2m2_img.hdr
run prisma --ref $D/stage3_datasets/set3_20260906/PRS_L2D_STD_20260305054623_20260305054628_0001/PRS_L2D_STD_20260305054623_20260305054628_0001.he5
run tan_ortho --ref $D/tanager_datasets/scenes/20250321_054913_40_4001/20250321_054913_40_4001_ortho_sr_hdf5.h5
run nometa
ls -la $O/*/*_report.html
python - <<'PY'
import glob, re
for p in sorted(glob.glob("/content/t8/out/*/*_report.html")):
    s = open(p).read()
    na = re.findall(r'class="na">([^<]*)', s)
    m = re.search(r"MAE \(all bands\)</td><td>([^<]*)", s); n = re.search(r"Pixels compared</td><td>([^<]*)", s)
    pr = re.search(r"Pairing</td><td>([^<]*)", s)
    print(p.split("/")[-2], "imgs", s.count("<img"), "| NA:", na, "| MAE", m and m.group(1), "|", n and n.group(1), "|", pr and pr.group(1))
PY
echo T9_DONE
