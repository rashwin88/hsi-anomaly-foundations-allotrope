#!/usr/bin/env bash
# Run the app/ test suite inside the worker container.
# tests/ is not baked into the image, so it is bind-mounted here at run time.
# Usage: ./scripts/run_tests.sh [extra pytest args]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$root/docker/.env" ]; then
    echo "docker/.env is missing - copy docker/.env.example and fill it in." >&2
    exit 1
fi

if [ $# -eq 0 ]; then set -- -q; fi
marks="not large_files and not large_benchmarks and not network_access"

# Mount the working tree over the image's baked copies, so tests exercise the
# code you just edited rather than whatever was current at the last build.
docker compose -f "$root/docker/docker-compose.yml" run --rm --no-deps \
    -v "$root/tests:/srv/tests" \
    -v "$root/pytest.ini:/srv/pytest.ini" \
    -v "$root/app:/srv/app" \
    -v "$root/backend/allotrope:/srv/allotrope" \
    -v "$root/backend/allotrope_worker:/srv/allotrope_worker" \
    worker python -m pytest tests -m "$marks" "$@"
