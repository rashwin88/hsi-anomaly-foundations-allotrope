#!/usr/bin/env bash
# build_sixs.sh <6S source dir> <out dir>
# Builds the fast 6S binaries from the source `isofit download sixs` leaves behind (left untouched, it stays the
# stock reference). Each build gets both patches (sixs_noaer_patch.py, sixs_cache_patch.py):
#   <out>/check/sixsV2.1   -O, ISOFIT's own flags: must match stock exactly (sixs_check.py), then deleted
#   <out>/v3/sixsV2.1      -O3 -march=x86-64-v3 (AVX2/FMA: any x86 server since ~2015, the build PC)
#   <out>/v4/sixsV2.1      -O3 -march=x86-64-v4 (AVX-512: the VM's Xeon 6760P)
# Never -march=native: it would bake in whatever CPU the image happens to be built on.
# The 6S wrapper (item 3) picks v4 or v3 at run time from /proc/cpuinfo.
set -euo pipefail
SRC=$1; OUT=$2; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
build() {  # name flags
  local d="$OUT/$1.src"
  rm -rf "$d"; cp -r "$SRC" "$d"; rm -f "$d"/*.o "$d"/sixsV2.1 "$d"/sixsV2.1-lutaero
  grep -q '^EXTRA   = -O -ffixed-line-length-132 -std=legacy$' "$d/Makefile"   # the last EXTRA line wins in make
  sed -i "s/^EXTRA   = -O -ffixed-line-length-132 -std=legacy$/EXTRA   = $2 -ffixed-line-length-132 -std=legacy/" "$d/Makefile"
  python "$HERE/sixs_noaer_patch.py" "$d"
  python "$HERE/sixs_cache_patch.py" "$d"
  make -C "$d" -j"$(nproc)" sixs > "$d/make.log" 2>&1 || { tail -30 "$d/make.log"; exit 1; }
  mkdir -p "$OUT/$1"; mv "$d/sixsV2.1" "$OUT/$1/sixsV2.1"; rm -rf "$d"
  echo "built $OUT/$1/sixsV2.1 ($2)"
}
build check "-O"
build v3 "-O3 -march=x86-64-v3"
build v4 "-O3 -march=x86-64-v4"
