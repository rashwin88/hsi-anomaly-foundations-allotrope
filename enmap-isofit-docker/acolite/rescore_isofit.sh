#!/usr/bin/env bash
# Re-score the ISOFIT 1.1.4 L2A products with the current validate_vs_paco.py,
# so they sit next to the ACOLITE numbers. Usage: rescore_isofit.sh <out_dir> <csv_name>
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"; REPO="$(dirname "$HERE")"
DATA="${DATA:-/mnt/c/Users/FITT/Documents/gaic_stage3_datasets/stage3_20260906}"
mkdir -p "$1"; OUT="$(readlink -f "$1")"
docker run --rm --entrypoint bash -v "$REPO/src:/src:ro" -v "$DATA:/data:ro" -v "$OUT:/out" \
    "${IMG:-acolite:8e04b97}" -c 'cd /src && python -m validate_vs_paco --paco "/data/with ac" --ours "/data/without ac" --csv /out/'"$2"
