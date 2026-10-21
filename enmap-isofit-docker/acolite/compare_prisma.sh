#!/usr/bin/env bash
# Score every PRISMA ACOLITE run under <run_root>/<timestamp> against ASI's L2D
# with validate_prisma_vs_asi.py. Usage: compare_prisma.sh <run_root> <csv_name>
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"; REPO="$(dirname "$HERE")"
DATA="${DATA:-/mnt/c/Users/FITT/Documents/gaic_stage3_datasets/stage3_20260906}"
RUN="$(readlink -f "$1")"
docker run --rm --entrypoint bash \
    -v "$REPO/src:/src:ro" -v "$DATA:/data:ro" -v "$RUN:/run" \
    "${IMG:-acolite:8e04b97}" -c '
asi=(); ours=()
for d in /run/2*/; do
    ts=$(basename "$d")
    l2r=$(ls "$d"*_L2R.nc 2>/dev/null | head -1) || true
    [ -z "$l2r" ] && continue
    asi+=("$(ls "/data/with ac/PRS_L2D_STD_${ts}"_*/*.he5)"); ours+=("$l2r")
done
cd /src && python -m validate_prisma_vs_asi --asi "${asi[@]}" --ours "${ours[@]}" --csv /run/'"$2"'
'
