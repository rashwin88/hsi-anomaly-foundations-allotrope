#!/usr/bin/env bash
# =============================================================================
# Build EnMAP-ISOFIT image on the connected node and export for air-gap transfer.
#
# Run on the connected build node. Requires: docker (or podman), pigz.
# Produces:
#   dist/enmap-isofit-<VERSION>.tar.gz     (Docker image)
#   dist/enmap-isofit-<VERSION>.sha256     (checksum)
#   dist/aux-manifest.txt                  (bill of materials)
# =============================================================================
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-enmap-isofit}"
IMAGE_VERSION="${IMAGE_VERSION:-1.1.3}"
ISOFIT_VERSION="${ISOFIT_VERSION:-3.7.7}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
mkdir -p "$DIST_DIR"

TAG="${IMAGE_NAME}:${IMAGE_VERSION}"

echo "==> Building ${TAG}"
docker build \
    --build-arg ISOFIT_VERSION="${ISOFIT_VERSION}" \
    -t "${TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"

echo "==> Recording image metadata"
docker inspect "${TAG}" > "${DIST_DIR}/enmap-isofit-${IMAGE_VERSION}.inspect.json"

echo "==> Exporting image tarball (this can take several minutes)"
TARBALL="${DIST_DIR}/enmap-isofit-${IMAGE_VERSION}.tar"
docker save "${TAG}" -o "${TARBALL}"

echo "==> Compressing with pigz"
if command -v pigz >/dev/null 2>&1; then
    pigz -9 -f "${TARBALL}"
    TARBALL="${TARBALL}.gz"
else
    gzip -9 -f "${TARBALL}"
    TARBALL="${TARBALL}.gz"
fi

echo "==> Computing SHA256"
( cd "${DIST_DIR}" && sha256sum "$(basename "${TARBALL}")" > "enmap-isofit-${IMAGE_VERSION}.sha256" )

echo "==> Extracting aux data bill of materials"
docker run --rm --entrypoint /bin/bash "${TAG}" -c "
    echo '# sRTMnet weights';
    ls -l /root/.isofit/srtmnet/;
    echo;
    echo '# 6S binary';
    ls -l /root/.isofit/sixs/sixsV*;
    echo;
    echo '# Surface priors';
    ls -l /opt/aux/surface/;
    echo;
    echo '# ISOFIT version';
    python -c 'import isofit; print(isofit.__version__)';
    echo;
    echo '# Key package versions';
    pip freeze | grep -iE '^(isofit|tensorflow|ray|rasterio|gdal|numpy|scipy)==';
" > "${DIST_DIR}/aux-manifest.txt"

echo
echo "==> Build complete."
echo "    Image:   ${TAG}"
echo "    Tarball: ${TARBALL}"
echo "    SHA256:  ${DIST_DIR}/enmap-isofit-${IMAGE_VERSION}.sha256"
echo
echo "Transfer these to the air-gap host:"
echo "    ${TARBALL}"
echo "    ${DIST_DIR}/enmap-isofit-${IMAGE_VERSION}.sha256"
echo "    ${DIST_DIR}/aux-manifest.txt"
echo
echo "Then run scripts/deploy_airgap.sh on the target."
