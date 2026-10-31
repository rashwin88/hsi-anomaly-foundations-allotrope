#!/usr/bin/env bash
# =============================================================================
# Deploy the isofit-atm-docker image on an air-gapped host.
#
# Run on the target after transferring:
#   isofit-atm-docker-<VERSION>.tar.gz
#   isofit-atm-docker-<VERSION>.sha256
# =============================================================================
set -euo pipefail

TARBALL="${1:-}"
SHA_FILE="${2:-}"

if [[ -z "$TARBALL" || -z "$SHA_FILE" ]]; then
    cat <<EOF
Usage: $0 <tarball.tar.gz> <sha256_file>

Example:
    $0 isofit-atm-docker-1.3.0.tar.gz isofit-atm-docker-1.3.0.sha256
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
docker images | grep -i isofit-atm-docker || true

echo
echo "==> Smoke test: verify entrypoint and aux data"
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i isofit-atm-docker | head -n1)
echo "    Testing image: $IMG"

docker run --rm "$IMG" --help
docker run --rm --entrypoint python "$IMG" -c "
import isofit, torch, rasterio, numpy, scipy
import os
print(f'ISOFIT:             {isofit.__version__}')
print(f'PyTorch:            {torch.__version__}')
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
echo "==> 6S builds vs stock on this CPU, and the GPU as the per-pixel solve sees it (non-root uid)"
docker run --rm --user 1000:1000 --entrypoint bash "$IMG" -c '/root/.isofit/sixs/sixsV2.1 --selftest'
docker run --rm --user 1000:1000 --gpus all --entrypoint python "$IMG" -c \
    "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
    || echo "    (no GPU visible: install nvidia-container-toolkit, or run CPU-only with GPUS=none)"

echo
echo "==> Deployment complete."
echo
echo "Next steps:"
echo "  1. Stage Copernicus DEM tiles for your AOIs into /path/to/dem/"
echo "     (see scripts/stage_dem.sh - run on connected node, transfer output)"
echo "  2. Run a scene with scripts/run_scene.sh"
