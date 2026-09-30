#!/usr/bin/env bash
# PRISMA L1 -> ACOLITE L2R for every PRS_L1_STD_OFFL_* product in a folder.
# Usage: run_prisma.sh <settings.txt> <out_root> [timestamp filter]
# Geometry comes from src/prisma_l1_geometry.py, since ACOLITE otherwise
# requires ASI's L2C product next to the L1.
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"; REPO="$(dirname "$HERE")"
SRC="${SRC:-/mnt/c/Users/FITT/Documents/gaic_stage3_datasets/stage3_20260906/without ac}"
IMG="${IMG:-acolite:8e04b97}"
C="$HOME/acolite_work/cache"; mkdir -p "$C/LUT" "$C/MET"
NETRC_ARGS=(); [ -f "$HOME/.netrc" ] && NETRC_ARGS=(-v "$HOME/.netrc:/root/.netrc:ro")
for d in "$SRC"/PRS_L1_STD_OFFL_*; do
    id=$(basename "$d"); ts=${id#PRS_L1_STD_OFFL_}; ts=${ts%%_*}
    [ -n "${3:-}" ] && [ "$ts" != "$3" ] && continue
    out="$2/$ts"; mkdir -p "$out"; out="$(readlink -f "$out")"
    ls "$out"/*_L2R.nc >/dev/null 2>&1 && { echo "$ts already done"; continue; }
    echo "[$(date -u +%FT%TZ)] $ts"
    { cat "$1"; echo "l2cfile=/out/PRS_L2C_STD_${id#PRS_L1_STD_OFFL_}.he5"; } > "$out/settings.txt"
    docker run --rm -v "$d:/in:ro" -v "$out:/out" -v "$REPO/src:/src:ro" \
        -v "$C/LUT:/opt/acolite/data/LUT" -v "$C/MET:/opt/acolite/data/MET" "${NETRC_ARGS[@]}" \
        --entrypoint bash "$IMG" -c "
            python /src/prisma_l1_geometry.py --l1 /in/$id.he5 --out /out &&
            python launch_acolite.py --cli --settings /out/settings.txt --inputfile /in/$id.he5 --output /out" \
        > "$out.log" 2>&1 || echo "  FAILED, see $out.log"
done
