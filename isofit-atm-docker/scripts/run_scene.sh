#!/usr/bin/env bash
# =============================================================================
# Process one scene (any supported sensor / format) through the container.
#
#   scripts/run_scene.sh <input file or folder> <output dir> [entrypoint options]
#
# Environment:
#   IMG       image (default isofit-atm-docker:1.3.0)
#   DEM_DIR   folder holding cop30_mosaic.vrt + tiles (optional: without it the
#             run uses a flat 0 m and flags it)
#   REF       reference product for the report's comparison (optional; mounted
#             read-only and passed as --ref)
#   GPUS      docker --gpus value (default all; "none" for a CPU-only host)
#   SHM_SIZE  /dev/shm for Ray's object store (default 16g; Docker's 64 MB
#             default makes Ray spill to disk)
#   N_CORES   parallel workers (default: all CPUs)
# The container runs as the caller's uid, so outputs belong to the caller.
# =============================================================================
set -euo pipefail

IMG="${IMG:-isofit-atm-docker:1.3.0}"
IN="${1:?Usage: $0 <input file or folder> <output dir> [entrypoint options]}"
OUT="${2:?output dir required}"
shift 2

IN=$(readlink -f "$IN")
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")
ARGS=(--rm --user "$(id -u):$(id -g)" --shm-size "${SHM_SIZE:-16g}"
      -e N_CORES="${N_CORES:-$(nproc)}"
      -v "$(dirname "$IN"):/data/in:ro" -v "${OUT}:/data/out")
[[ "${GPUS:-all}" != "none" ]] && ARGS+=(--gpus "${GPUS:-all}")
[[ -n "${DEM_DIR:-}" ]] && ARGS+=(-v "$(readlink -f "$DEM_DIR"):/aux/dem:ro")
EXTRA=()
if [[ -n "${REF:-}" ]]; then
    REF=$(readlink -f "$REF")
    ARGS+=(-v "$(dirname "$REF"):/data/ref:ro")
    EXTRA=(--ref "/data/ref/$(basename "$REF")")
fi

docker run "${ARGS[@]}" "$IMG" "/data/in/$(basename "$IN")" /data/out "${EXTRA[@]}" "$@"

echo "Done. Outputs in: $OUT"
