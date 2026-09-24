#!/usr/bin/env bash
# =============================================================================
# Deploy the EnMAP-ISOFIT image on an air-gapped host.
#
# Run on the target after transferring:
#   enmap-isofit-<VERSION>.tar.gz
#   enmap-isofit-<VERSION>.sha256
# =============================================================================
set -euo pipefail

TARBALL="${1:-}"
SHA_FILE="${2:-}"

if [[ -z "$TARBALL" || -z "$SHA_FILE" ]]; then
    cat <<EOF
Usage: $0 <tarball.tar.gz> <sha256_file>

Example:
    $0 enmap-isofit-1.1.3.tar.gz enmap-isofit-1.1.3.sha256
EOF
    exit 1
fi

[[ -f "$TARBALL" ]]  || { echo "ERROR: tarball not found: $TARBALL"; exit 1; }
[[ -f "$SHA_FILE" ]] || { echo "ERROR: sha256 file not found: $SHA_FILE"; exit 1; }

echo "==> Verifying checksum"
if sha256sum -c "$SHA_FILE"; then
    echo "    Checksum OK"
else
    echo "ERROR: checksum mismatch - aborting"
    exit 1
fi

echo "==> Loading image into Docker"
docker load -i "$TARBALL"

echo "==> Loaded images:"
docker images | grep -i enmap-isofit || true

echo
echo "==> Smoke test: verify entrypoint and aux data"
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i enmap-isofit | head -n1)
echo "    Testing image: $IMG"

docker run --rm "$IMG" --help
docker run --rm --entrypoint python "$IMG" -c "
import isofit, tensorflow, rasterio, numpy, scipy
import os
print(f'ISOFIT:             {isofit.__version__}')
print(f'TensorFlow:         {tensorflow.__version__}')
print(f'Rasterio:           {rasterio.__version__}')
import glob
h5s = glob.glob('/root/.isofit/srtmnet/**/*.h5', recursive=True)
print(f'sRTMnet h5:         {h5s[0]} ({os.path.getsize(h5s[0])/1e6:.1f} MB)')
sixs = [p for p in glob.glob('/root/.isofit/sixs/sixsV*') if 'lutaero' not in p]
print(f'6S binary:          {sixs[0] if sixs else \"MISSING - LUTs will be all-NaN\"}')
print(f'Surface priors:     {os.listdir(\"/opt/aux/surface\")}')
print(f'isofit-data files:  {len(os.listdir(\"/opt/aux/isofit-data\"))}')
"

echo
echo "==> Deployment complete."
echo
echo "Next steps:"
echo "  1. Stage Copernicus DEM tiles for your AOIs into /path/to/dem/"
echo "     (see scripts/stage_dem.sh - run on connected node, transfer output)"
echo "  2. Run a scene with scripts/run_scene.sh"
