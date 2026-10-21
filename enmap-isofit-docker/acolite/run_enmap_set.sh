#!/usr/bin/env bash
# Run ACOLITE with one settings file over every EnMAP L1C scene in a folder.
# Usage: run_enmap_set.sh <settings.txt> <out_root> [DT id filter]
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
SRC="${SRC:-/mnt/c/Users/FITT/Documents/gaic_stage3_datasets/stage3_20260906/without ac}"
for d in "$SRC"/ENMAP01-____L1C-DT*; do
    dt=$(basename "$d" | grep -o 'DT[0-9]\{10\}')
    [ -n "${3:-}" ] && [ "$dt" != "$3" ] && continue
    ls "$2/$dt/"*_L2R.nc >/dev/null 2>&1 && { echo "$dt already done"; continue; }
    echo "[$(date -u +%FT%TZ)] $dt"
    "$HERE/run_acolite.sh" "$d" "$2/$dt" "$1" > "$2/$dt.log" 2>&1 || echo "  FAILED, see $2/$dt.log"
done
