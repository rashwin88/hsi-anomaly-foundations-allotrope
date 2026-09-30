#!/usr/bin/env bash
# Run ACOLITE on one L1 product dir.
# Usage: run_acolite.sh <l1_dir> <out_dir> <settings.txt>
# LUTs and MERRA-2 ancillary files are cached under ~/acolite_work/cache so they
# download once. ~/.netrc (EarthData login) is mounted if present; ancillary_fixed
# needs it.
set -euo pipefail
mkdir -p "$2"
L1="$(readlink -f "$1")"; OUT="$(readlink -f "$2")"; SET="$(readlink -f "$3")"
C="$HOME/acolite_work/cache"
mkdir -p "$C/LUT" "$C/MET"
NETRC_ARGS=()
[ -f "$HOME/.netrc" ] && NETRC_ARGS=(-v "$HOME/.netrc:/root/.netrc:ro")
docker run --rm \
    -v "$L1:/in:ro" -v "$OUT:/out" -v "$SET:/settings.txt:ro" \
    -v "$C/LUT:/opt/acolite/data/LUT" -v "$C/MET:/opt/acolite/data/MET" \
    "${NETRC_ARGS[@]}" \
    "${IMG:-acolite:8e04b97}" \
    --settings /settings.txt --inputfile /in --output /out
