#!/usr/bin/env bash
# Host-side loader. Lives INSIDE the bundle (as bootstrap.sh) and is also
# kept as a standalone script in the repo for development.
#
# Run on the REMOTE host after scp + tar-extract:
#
#     scp -P <port> -i <key> allotrope-bundle-*.tar.zst root@<host>:/root/
#     ssh -p <port> -i <key> root@<host>
#     cd /root
#     tar --use-compress-program=zstd -xf allotrope-bundle-*.tar.zst
#     bash bootstrap.sh
#
# What this script does:
#   1. Installs Docker + NVIDIA Container Toolkit if missing.
#   2. Smoke-tests GPU pass-through.
#   3. `docker load`s the four image tarballs.
#   4. Re-creates the named volumes and restores their contents.
#   5. Drops the compose files into /root/allotrope/.
#   6. `docker compose up -d` (no --build needed — images are pre-loaded).
#
# Idempotent — re-running on an already-loaded host is fast.

set -euo pipefail

log()  { printf "\n\033[1;34m▶\033[0m  %s\n" "$*"; }
ok()   { printf "   \033[1;32m✓\033[0m  %s\n" "$*"; }
warn() { printf "   \033[1;33m!\033[0m  %s\n" "$*"; }
err()  { printf "   \033[1;31m✗\033[0m  %s\n" "$*" >&2; }

# Resolve where we are. The bundle's contents must be in $PWD or in the
# directory containing this script.
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$BUNDLE_DIR/images" || ! -d "$BUNDLE_DIR/volumes" || ! -d "$BUNDLE_DIR/compose" ]]; then
  err "Bundle layout not found in $BUNDLE_DIR (need images/, volumes/, compose/)"
  err "Did you tar-extract the bundle in $BUNDLE_DIR ?"
  exit 1
fi
cd "$BUNDLE_DIR"

# Where the running stack will live on the host.
TARGET_DIR="/root/allotrope"

# ── 1. Disk space sanity ───────────────────────────────────────────────────
log "Checking disk space"
NEED_GB=60
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
  err "Less than ${NEED_GB} GB free on /. Have ${AVAIL_GB} GB. Free space or re-rent a bigger host."
  exit 1
fi
ok "${AVAIL_GB} GB free (need ≥ ${NEED_GB})"

# ── 2. Docker + nvidia-container-toolkit ───────────────────────────────────
log "Checking docker"
if ! command -v docker >/dev/null 2>&1; then
  log "  installing docker via convenience script"
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi
ok "docker $(docker --version)"

if ! docker compose version >/dev/null 2>&1; then
  log "  installing docker compose plugin"
  apt-get update -qq
  apt-get install -y -qq docker-compose-plugin
fi
ok "compose $(docker compose version --short)"

log "Checking NVIDIA Container Toolkit"
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  log "  installing nvidia-container-toolkit"
  distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi
ok "nvidia-container-toolkit present"

log "Host-level GPU check"
# Quick pre-flight on the host (driver visible to bare-metal). The real
# container-level test runs further down after we've loaded docker-worker.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  ok "host nvidia-smi reports a GPU"
else
  warn "host nvidia-smi not available — GPU might not be usable; continuing anyway"
fi

# ── 3. Bring down any existing stack ───────────────────────────────────────
if [[ -f "$TARGET_DIR/docker-compose.yml" ]]; then
  log "Bringing down any existing stack at $TARGET_DIR"
  ( cd "$TARGET_DIR" && docker compose down --remove-orphans >/dev/null 2>&1 || true )
fi

# ── 4. Load images ─────────────────────────────────────────────────────────
log "Loading docker images"
for img in api worker frontend postgres; do
  src="$BUNDLE_DIR/images/${img}.tar"
  if [[ ! -f "$src" ]]; then
    warn "  $src missing — skipping (will pull at compose-up time if possible)"
    continue
  fi
  log "  docker load ← $src"
  docker load -i "$src" >/dev/null
  ok "  loaded $img"
done

log "Container-level GPU pass-through test"
# Use the worker image we just loaded — no Docker Hub round-trip, works
# air-gapped. Verifies the real image (the one we actually run) can see
# the GPU, not a different vendor-provided base image.
if docker image inspect docker-worker >/dev/null 2>&1; then
  if docker run --rm --gpus all docker-worker nvidia-smi >/dev/null 2>&1; then
    ok "docker-worker can see the GPU"
  else
    warn "docker-worker --gpus all failed — torch will fall back to CPU"
    warn "(check that nvidia-container-toolkit is configured: 'nvidia-ctk runtime configure --runtime=docker' then 'systemctl restart docker')"
  fi
else
  warn "docker-worker image not present; skipping container-level GPU test"
fi

# ── 5. Restore volumes ─────────────────────────────────────────────────────
log "Restoring volumes"
restore_volume() {
  local volname="$1"
  local src="$BUNDLE_DIR/volumes/${volname}.tar"
  if [[ ! -f "$src" ]]; then
    warn "  no $volname.tar in bundle — leaving volume empty (will be created on first up)"
    docker volume create "$volname" >/dev/null
    return
  fi
  # Always destroy and re-create so a partial previous load doesn't merge.
  docker volume rm "$volname" >/dev/null 2>&1 || true
  docker volume create "$volname" >/dev/null
  docker run --rm \
    -v "$volname":/dst \
    -v "$BUNDLE_DIR/volumes":/src:ro \
    alpine:3.20 \
    sh -c "cd /dst && tar xf /src/${volname}.tar"
  local sz
  sz=$(du -sh "$src" | cut -f1)
  ok "  $volname restored ($sz on disk in bundle)"
}

restore_volume allotrope_db
restore_volume allotrope_data
restore_volume allotrope_artifacts
restore_volume allotrope_models
restore_volume allotrope_splib07

# ── 6. Compose files + env ─────────────────────────────────────────────────
log "Installing compose files at $TARGET_DIR"
mkdir -p "$TARGET_DIR"
cp "$BUNDLE_DIR/compose/docker-compose.yml"     "$TARGET_DIR/"
cp "$BUNDLE_DIR/compose/docker-compose.gpu.yml" "$TARGET_DIR/" 2>/dev/null || true
cp "$BUNDLE_DIR/compose/.env"                   "$TARGET_DIR/" 2>/dev/null || true
ok "compose installed"

# Tweak the compose file: rebind the api/frontend ports from 127.0.0.1 to
# 127.0.0.1 (no change) but make sure they're bound *inside* the host so the
# SSH tunnel works. They already are — but if a user customised this we leave
# their override intact.

# ── 7. Up ──────────────────────────────────────────────────────────────────
log "Bringing the stack up (no --build, images are pre-loaded)"
cd "$TARGET_DIR"
COMPOSE_CMD=(docker compose -f docker-compose.yml)
if [[ -f docker-compose.gpu.yml ]]; then
  COMPOSE_CMD+=(-f docker-compose.gpu.yml)
fi
"${COMPOSE_CMD[@]}" up -d
ok "stack up"

# Quick health check.
sleep 4
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Container status:"
docker ps --format "    {{.Names}}\t{{.Status}}" | grep allotrope || true
echo
echo "  Smoke checks:"
curl -s -o /dev/null -w "    api       %{http_code}\n" http://127.0.0.1:8000/healthz || true
curl -s -o /dev/null -w "    frontend  %{http_code}\n" http://127.0.0.1:3000           || true
echo
echo "  Ports 3000 + 8000 are bound to 127.0.0.1 on this host."
echo "  To reach the UI from your laptop, on YOUR LAPTOP run:"
echo
echo "    ssh -N -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000 -p <port> -i <key> root@<host>"
echo
echo "  Then open  http://localhost:3000  in the browser."
echo "═══════════════════════════════════════════════════════════════"
