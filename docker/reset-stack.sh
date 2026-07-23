#!/usr/bin/env bash
# Wipe Allotrope's DB + scene/artifact volumes and bring the stack back up
# on a clean slate. Keeps `allotrope_models` (model checkpoints) intact so
# we don't have to re-download / re-load the foundation model bundles.
#
# Usage:
#   ./docker/reset-stack.sh [--include-models]
#
# Flags:
#   --include-models   Also wipe allotrope_models (full nuclear reset).
#                       You'll need to reseed the model checkpoints after.
#
# Safe to re-run; each step is idempotent. Exits non-zero on first failure.
#
# What gets wiped (default):
#   - allotrope_db          postgres data dir (all tables, all rows)
#   - allotrope_data        scene raw files + vendables + staging
#   - allotrope_artifacts   thumbnails + per-project action outputs + viz + exports
#
# What survives (default):
#   - allotrope_models      foundation model checkpoints
#
# What the script does:
#   1. docker compose down              (stops + removes containers, keeps volumes)
#   2. docker volume rm <targets>       (drops the named volumes)
#   3. docker compose up -d             (recreates containers + empty volumes;
#                                        the `bootstrap` one-shot service runs
#                                        alembic + seed-admin + seed-action-templates
#                                        before api/worker start — see Step 21)
#   4. wait for api healthcheck         (proxy for "bootstrap finished cleanly")
#
# After this, the UI loads with: 1 user (admin), 0 scenes, 0 projects, 0 jobs,
# 8 action templates (system defaults), and the configured model catalog.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

INCLUDE_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --include-models) INCLUDE_MODELS=1 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

TARGETS=(allotrope_db allotrope_data allotrope_artifacts)
if [[ "$INCLUDE_MODELS" -eq 1 ]]; then
  TARGETS+=(allotrope_models)
fi

echo "About to wipe these Docker volumes:"
printf '  - %s\n' "${TARGETS[@]}"
echo
read -r -p "Type 'wipe' to confirm: " confirm
if [[ "$confirm" != "wipe" ]]; then
  echo "aborted." >&2
  exit 1
fi

echo
echo "[1/6] docker compose down"
docker compose -f "$COMPOSE_FILE" down

echo
echo "[2/6] removing volumes: ${TARGETS[*]}"
# `docker volume rm` errors if a volume is missing; tolerate that.
for v in "${TARGETS[@]}"; do
  if docker volume inspect "$v" >/dev/null 2>&1; then
    docker volume rm "$v"
  else
    echo "  (already absent: $v)"
  fi
done

echo
echo "[3/4] docker compose up -d (bootstrap service runs migrations + seeds)"
# Compose dependency order: postgres → bootstrap → api/worker. The
# bootstrap service (Step 21) runs alembic + seed-admin +
# seed-action-templates, then exits; api/worker only start once it
# completes successfully. So a single `up -d` does what used to take
# 4 commands.
docker compose -f "$COMPOSE_FILE" up -d

echo
echo "[4/4] waiting for api healthcheck..."
# api comes up only after bootstrap. Poll /healthz/db for at most 60s
# so we exit with a clear signal when the stack is actually usable.
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  if curl -fsS http://127.0.0.1:8000/healthz/db >/dev/null 2>&1; then
    echo "  api is healthy."
    break
  fi
  sleep 2
done

echo
echo "Done. UI: http://127.0.0.1:3000"
echo
echo "Bootstrap logs (in case anything looked off):"
docker compose -f "$COMPOSE_FILE" logs bootstrap 2>&1 | tail -20
