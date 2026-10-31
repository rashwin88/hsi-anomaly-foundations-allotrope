#!/usr/bin/env bash
# =============================================================================
# Build the isofit-atm-docker image on the connected node and export for air-gap transfer.
#
# Run on the connected build node. Requires: docker (or podman), pigz.
# Produces:
#   dist/isofit-atm-docker-<VERSION>.tar.gz     (Docker image)
#   dist/isofit-atm-docker-<VERSION>.sha256     (checksum)
#   dist/aux-manifest.txt                  (bill of materials)
# =============================================================================
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-isofit-atm-docker}"
IMAGE_VERSION="${IMAGE_VERSION:-1.3.0}"
ISOFIT_VERSION="${ISOFIT_VERSION:-4.1.5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
mkdir -p "$DIST_DIR"

TAG="${IMAGE_NAME}:${IMAGE_VERSION}"

echo "==> Unit tests on the runtime image (target test: fails on a test or coverage < 30 %)"
docker build \
    --build-arg ISOFIT_VERSION="${ISOFIT_VERSION}" \
    --target test \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"

echo "==> Building ${TAG} (target runtime; layers cached from the step above)"
docker build \
    --build-arg ISOFIT_VERSION="${ISOFIT_VERSION}" \
    --target runtime \
    -t "${TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"

echo "==> Smoke test as a non-root uid (how scripts/run_scene.sh runs it)"
docker run --rm --user 1000:1000 --entrypoint bash "${TAG}" -c '
    /root/.isofit/sixs/sixsV2.1 --selftest &&
    python -c "import isofit, torch, report, writers, readers.prep; print(\"isofit\", isofit.__version__, \"torch\", torch.__version__)"'

echo "==> Vulnerability scan (target: < 100 CRITICAL)"
if command -v trivy >/dev/null 2>&1; then
    trivy image --scanners vuln --severity CRITICAL --format table \
        --output "${DIST_DIR}/trivy-critical-${IMAGE_VERSION}.txt" "${TAG}"
    trivy image --scanners vuln --severity CRITICAL,HIGH --format json \
        --output "${DIST_DIR}/trivy-${IMAGE_VERSION}.json" "${TAG}"
    python3 - "${DIST_DIR}/trivy-${IMAGE_VERSION}.json" <<'PY'
import collections, json, sys
r = json.load(open(sys.argv[1]))
n = collections.Counter()
for t in r.get("Results", []):
    for v in t.get("Vulnerabilities") or []:
        n[(v["Severity"], t.get("Type", "?"))] += 1
for (sev, typ), c in sorted(n.items()):
    print(f"  {sev:8s} {typ:12s} {c}")
print("  CRITICAL total:", sum(c for (s, _), c in n.items() if s == "CRITICAL"))
PY
else
    echo "    trivy not installed: skipped (https://trivy.dev, then rerun or: trivy image ${TAG})"
fi

echo "==> Recording image metadata"
docker inspect "${TAG}" > "${DIST_DIR}/isofit-atm-docker-${IMAGE_VERSION}.inspect.json"

echo "==> Exporting image tarball (this can take several minutes)"
TARBALL="${DIST_DIR}/isofit-atm-docker-${IMAGE_VERSION}.tar"
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
( cd "${DIST_DIR}" && sha256sum "$(basename "${TARBALL}")" > "isofit-atm-docker-${IMAGE_VERSION}.sha256" )

echo "==> Extracting aux data bill of materials"
docker run --rm --entrypoint bash "${TAG}" -c "
    echo '# sRTMnet weights';
    ls -l /root/.isofit/srtmnet/;
    echo;
    echo '# 6S wrapper + builds';
    ls -l /root/.isofit/sixs/sixsV2.1 /opt/sixs/*/sixsV2.1;
    echo;
    echo '# Surface priors';
    ls -l /opt/aux/surface/;
    echo;
    echo '# ISOFIT version';
    python -c 'import isofit; print(isofit.__version__)';
    echo;
    echo '# Key package versions';
    pip freeze | grep -iE '^(isofit|torch|ray|rasterio|gdal|numpy|scipy|matplotlib|h5py)==';
" > "${DIST_DIR}/aux-manifest.txt"

echo
echo "==> Build complete."
echo "    Image:   ${TAG}"
echo "    Tarball: ${TARBALL}"
echo "    SHA256:  ${DIST_DIR}/isofit-atm-docker-${IMAGE_VERSION}.sha256"
echo
echo "Transfer these to the air-gap host:"
echo "    ${TARBALL}"
echo "    ${DIST_DIR}/isofit-atm-docker-${IMAGE_VERSION}.sha256"
echo "    ${DIST_DIR}/aux-manifest.txt"
echo
echo "Then run scripts/deploy_airgap.sh on the target."
