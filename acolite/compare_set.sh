#!/usr/bin/env bash
# Repack every ACOLITE run under <run_root>/DT* as PACO layout, then score it
# against DLR PACO with validate_vs_paco.py. Usage: compare_set.sh <run_root> <csv_name>
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"; REPO="$(dirname "$HERE")"
DATA="${DATA:-/mnt/c/Users/FITT/Documents/gaic_stage3_datasets/stage3_20260906}"
RUN="$(readlink -f "$1")"
docker run --rm --entrypoint bash \
    -v "$REPO/src:/src:ro" -v "$DATA:/data:ro" -v "$RUN:/run" \
    "${IMG:-acolite:8e04b97}" -c '
set -e
for d in /run/DT*/; do
    dt=$(basename "$d")
    l2r=$(ls "$d"*_L2R.nc | head -1)
    l1c=$(ls -d "/data/without ac/ENMAP01-____L1C-$dt"_*)
    python /src/acolite_to_paco.py --l2r "$l2r" --l1c "$l1c" --out /run/paco_layout
done
cd /src && python -m validate_vs_paco --paco "/data/with ac" --ours /run/paco_layout --csv /run/'"$2"'
'
