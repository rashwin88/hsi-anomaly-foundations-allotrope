#!/usr/bin/env bash
# =============================================================================
# Process one EnMAP L1C scene through the container.
# =============================================================================
set -euo pipefail

IMG="${IMG:-enmap-isofit:1.1.3}"
L1C_DIR="${1:?Usage: $0 <l1c_dir> <l2a_out_dir> <dem_dir> [--season summer|winter|auto] [--n-cores N]}"
L2A_DIR="${2:?output dir required}"
DEM_DIR="${3:?DEM dir required}"
shift 3

# Absolute paths for the volume mounts
L1C_DIR=$(readlink -f "$L1C_DIR")
L2A_DIR=$(readlink -f "$L2A_DIR")
DEM_DIR=$(readlink -f "$DEM_DIR")

mkdir -p "$L2A_DIR"

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${L1C_DIR}:/data/l1c:ro" \
    -v "${L2A_DIR}:/data/l2a" \
    -v "${DEM_DIR}:/aux/dem:ro" \
    -e N_CORES="${N_CORES:-$(nproc)}" \
    "${IMG}" \
    /data/l1c /data/l2a "$@"

echo "Done. Outputs in: $L2A_DIR"
