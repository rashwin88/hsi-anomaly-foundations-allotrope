#!/usr/bin/env bash
# 6S wrapper, installed under ISOFIT's 6S executable name (/root/.isofit/sixs/sixsV2.1). ISOFIT runs
# `"<exe>" < LUT_<point>.inp > <point>`, so this sees exactly what 6S would and passes it through.
#   binary   x86-64-v3 build (AVX2/FMA) by default; SIXS_ARCH=v4 picks the AVX-512 build when the CPU has it.
#            v3 is the default because it measured faster: 1.29 s vs 1.38 s per point on Colab's AVX-512 Xeon
#            (AVX-512 lowers the clock). A CPU without AVX2 gets stock. (build_sixs.sh)
#   cache    with SIXS_CACHE_DIR set (entrypoint.sh, one dir per run), points that differ only in water vapour
#            share one scattering cache file (sixs_cache_patch.py). The key hashes the binary and the input with
#            the H2O value masked; 6S carries H2O on two lines (target and sensor level), the mask covers both.
#            The first point of a group takes the group's lock and fills the file. A point that finds the lock
#            taken does not wait: it runs without the cache, so the cache never makes a run slower.
# Off switches: SIXS_FAST=0 runs the stock binary with no cache; SIXS_CACHE=0 keeps the fast binary, no cache.
# `sixsV2.1 --selftest` checks the installed builds against stock on this CPU (sixs_check.py).
set -u
D=/opt/sixs
if [[ "${1:-}" == "--selftest" ]]; then
    exec python /opt/build/sixs_check.py "$D/stock/sixsV2.1" "$D" /opt/build/sixs_fixtures
fi

flags=" $(grep -m1 '^flags' /proc/cpuinfo) "
has() { for f in "$@"; do [[ "$flags" == *" $f "* ]] || return 1; done; }
if [[ "${SIXS_FAST:-1}" == "0" ]]; then exec "$D/stock/sixsV2.1"
elif [[ "${SIXS_ARCH:-v3}" == "v4" ]] && has avx512f avx512bw avx512cd avx512dq avx512vl avx2 fma bmi2; then bin=$D/v4/sixsV2.1
elif has avx2 fma bmi2; then bin=$D/v3/sixsV2.1
else exec "$D/stock/sixsV2.1"; fi
[[ "${SIXS_CACHE:-1}" == "0" || -z "${SIXS_CACHE_DIR:-}" ]] && exec "$bin"

mkdir -p "$SIXS_CACHE_DIR"
inp=$(mktemp "$SIXS_CACHE_DIR/inp.XXXXXX")
trap 'rm -f "$inp"' EXIT
cat > "$inp"
h2o=$(awk 'f { sub(/,.*/, ""); gsub(/ /, ""); print; exit } /User defined H2O/ { f = 1 }' "$inp")
[[ -z "$h2o" ]] && { "$bin" < "$inp"; exit; }     # no user-defined H2O: nothing to share
mask='BEGIN { FS = OFS = "," } { x = $1; gsub(/ /, "", x); if (x == t || x == "-" t) $1 = "@H2O@" } 1'
key=$({ echo "$bin"; awk -v t="$h2o" "$mask" "$inp"; } | sha256sum | cut -c1-32)
c=$SIXS_CACHE_DIR/$key
if [[ -s "$c" ]]; then SIXS_DISCOM_CACHE=$c "$bin" < "$inp"; exit; fi
exec 9> "$c.lock"
if flock -n 9; then SIXS_DISCOM_CACHE=$c "$bin" < "$inp"   # reads the file if a sibling just wrote it
else "$bin" < "$inp"; fi
