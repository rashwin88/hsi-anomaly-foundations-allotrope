#!/usr/bin/env bash
# Build a one-shot deployment bundle from the running local stack.
#
# Output: dist/allotrope-bundle-<short-sha>-<timestamp>.tar.zst
#
# Contents:
#   images/{api,worker,frontend}.tar    # cross-built for linux/amd64
#   volumes/{db,data,artifacts,models,splib07}.tar  # snapshotted via tar
#   compose/{docker-compose.yml,docker-compose.gpu.yml,.env}
#   bootstrap.sh                        # self-extract loader for the host
#   manifest.json                       # arch / sha / sizes / timestamps
#
# Usage:
#   ./scripts/snapshot_bundle.sh
#
# What the script does (in order):
#   1. Sanity-checks: postgres is `postgres:16-alpine`, docker buildx is available.
#   2. Cross-builds the three images for linux/amd64 (with cache reuse).
#   3. Saves each image as a tar inside the bundle.
#   4. Stops the entire stack (cleanly) so volume snapshots are consistent.
#   5. Tars each named volume by spinning up a throwaway alpine container that
#      bind-mounts the volume + the output dir.
#   6. Restarts the local stack.
#   7. Copies compose + .env + the host bootstrap script into the bundle.
#   8. Zstd-compresses the whole thing into a single tarball.
#
# Output path printed at the end. SCP that file to the remote host.

set -euo pipefail

# ── Helpers ────────────────────────────────────────────────────────────────
log()  { printf "\n\033[1;34m▶\033[0m  %s\n" "$*"; }
ok()   { printf "   \033[1;32m✓\033[0m  %s\n" "$*"; }
warn() { printf "   \033[1;33m!\033[0m  %s\n" "$*"; }
err()  { printf "   \033[1;31m✗\033[0m  %s\n" "$*" >&2; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DIST_DIR="$REPO_ROOT/dist"
mkdir -p "$DIST_DIR"

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d -t allotrope-bundle-XXXX)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK"/{images,volumes,compose}

BUNDLE_NAME="allotrope-bundle-${GIT_SHA}-${TS}.tar.zst"
BUNDLE_PATH="$DIST_DIR/$BUNDLE_NAME"

# ── 1. Pre-flight ──────────────────────────────────────────────────────────
log "Pre-flight"

if ! docker buildx version >/dev/null 2>&1; then
  err "docker buildx not available. Update Docker Desktop or install buildx."
  exit 1
fi
ok "docker buildx present"

if ! command -v zstd >/dev/null 2>&1; then
  err "zstd not installed. Install via 'brew install zstd' (Mac) or apt."
  exit 1
fi
ok "zstd present"

# Confirm the postgres image tag matches what's in the compose file. If the
# laptop has a different version cached, the volume bytes won't be portable.
LOCAL_PG_IMAGE="$(docker inspect --format '{{.Image}}' allotrope_postgres 2>/dev/null || true)"
EXPECTED_PG="postgres:16-alpine"
if [[ -z "$LOCAL_PG_IMAGE" ]]; then
  warn "postgres container not running; can't verify pg image tag (proceeding)"
else
  EXPECTED_IMG_ID="$(docker inspect --format '{{.Id}}' "$EXPECTED_PG" 2>/dev/null || true)"
  if [[ -n "$EXPECTED_IMG_ID" && "$LOCAL_PG_IMAGE" != "$EXPECTED_IMG_ID" ]]; then
    warn "running postgres is not $EXPECTED_PG — volume may not be portable to remote"
  else
    ok "postgres image matches $EXPECTED_PG"
  fi
fi

# Ensure a buildx builder that supports linux/amd64 cross-build.
BUILDER_NAME="allotrope-xbuild"
if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  log "Creating buildx builder '$BUILDER_NAME' for cross-build"
  docker buildx create --name "$BUILDER_NAME" --driver docker-container --bootstrap >/dev/null
  ok "builder created"
else
  ok "builder '$BUILDER_NAME' already exists"
fi
docker buildx use "$BUILDER_NAME"

# ── 2. Cross-build images for linux/amd64 ──────────────────────────────────
log "Cross-building images for linux/amd64 (slow first time)"

build_image() {
  local tag="$1"
  local dockerfile="$2"
  local context="$3"
  log "  building $tag ($dockerfile)"
  docker buildx build \
    --platform linux/amd64 \
    --file "$dockerfile" \
    --tag "$tag" \
    --load \
    --cache-from "type=local,src=$DIST_DIR/buildx-cache" \
    --cache-to   "type=local,dest=$DIST_DIR/buildx-cache,mode=max" \
    "$context"
  ok "  built $tag"
}

build_image docker-api      backend/Dockerfile           "$REPO_ROOT"
build_image docker-worker   backend/Dockerfile.worker    "$REPO_ROOT"
build_image docker-frontend frontend/Dockerfile          "$REPO_ROOT/frontend"

log "Exporting image tarballs"
docker save docker-api       -o "$WORK/images/api.tar"
docker save docker-worker    -o "$WORK/images/worker.tar"
docker save docker-frontend  -o "$WORK/images/frontend.tar"
# Also save the postgres image so the host doesn't have to pull from
# Docker Hub (in case the remote is air-gapped or rate-limited).
docker pull --platform linux/amd64 postgres:16-alpine >/dev/null
docker save postgres:16-alpine -o "$WORK/images/postgres.tar"
ok "images saved"

# ── 3. Snapshot the volumes ────────────────────────────────────────────────
log "Stopping local stack so volume snapshots are consistent"
docker compose -f docker/docker-compose.yml down --remove-orphans >/dev/null
ok "stack stopped"

snapshot_volume() {
  local volname="$1"
  local out="$WORK/volumes/${volname}.tar"
  log "  tarring volume $volname"
  if ! docker volume inspect "$volname" >/dev/null 2>&1; then
    warn "  volume $volname does not exist — skipping (empty placeholder will be created on the remote)"
    return
  fi
  # Run a throwaway alpine that bind-mounts both the source volume and the
  # output dir; tar from inside.
  docker run --rm \
    -v "$volname":/src:ro \
    -v "$WORK/volumes":/out \
    alpine:3.20 \
    sh -c "cd /src && tar cf /out/${volname}.tar ."
  # Report size.
  local sz
  sz=$(du -h "$out" | cut -f1)
  ok "  $volname → $(basename "$out") ($sz)"
}

snapshot_volume allotrope_db
snapshot_volume allotrope_data
snapshot_volume allotrope_artifacts
snapshot_volume allotrope_models
snapshot_volume allotrope_splib07

log "Restarting local stack"
docker compose -f docker/docker-compose.yml up -d >/dev/null
ok "stack back up"

# ── 4. Compose + .env ──────────────────────────────────────────────────────
log "Copying compose + .env"
cp docker/docker-compose.yml     "$WORK/compose/"
cp docker/docker-compose.gpu.yml "$WORK/compose/" 2>/dev/null || true
if [[ -f docker/.env ]]; then
  cp docker/.env "$WORK/compose/.env"
  ok "shipping docker/.env AS-IS — rotate secrets after teardown"
else
  warn "docker/.env not found; host will need to provide one"
fi

# ── 5. Manifest ────────────────────────────────────────────────────────────
log "Writing manifest"
python3 - "$WORK" "$GIT_SHA" "$TS" <<'PY'
import json, os, pathlib, platform, sys, subprocess
work = pathlib.Path(sys.argv[1])
sha = sys.argv[2]
ts = sys.argv[3]
def sz(p): return p.stat().st_size if p.is_file() else 0
manifest = {
    "schema": "allotrope-bundle/1",
    "git_sha": sha,
    "generated_at_utc": ts,
    "built_on": {"system": platform.system(), "machine": platform.machine()},
    "target_arch": "linux/amd64",
    "postgres_image": "postgres:16-alpine",
    "images": {p.name: sz(p) for p in sorted((work / "images").iterdir())},
    "volumes": {p.name: sz(p) for p in sorted((work / "volumes").iterdir())},
    "compose": sorted(p.name for p in (work / "compose").iterdir()),
}
(work / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

# ── 6. Host bootstrap script (lives INSIDE the bundle) ─────────────────────
log "Embedding bootstrap.sh"
cp "$REPO_ROOT/scripts/remote_load.sh" "$WORK/bootstrap.sh"
chmod +x "$WORK/bootstrap.sh"
ok "bootstrap.sh embedded"

# ── 7. Compress ────────────────────────────────────────────────────────────
log "Compressing bundle (zstd -19, this takes a few minutes)"
# tar+zstd with multi-threaded zstd. Level 19 trades CPU for ~30% smaller files.
(cd "$WORK" && tar cf - .) | zstd -T0 -19 -o "$BUNDLE_PATH"
ok "bundle written → $BUNDLE_PATH"

FINAL_SIZE=$(du -h "$BUNDLE_PATH" | cut -f1)
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Bundle ready: $BUNDLE_PATH"
echo "  Size:         $FINAL_SIZE"
echo
echo "  Ship it:"
echo "    scp -P <vast-port> -i <key> $BUNDLE_PATH root@<host>:/root/"
echo
echo "  Then SSH in and:"
echo "    tar --use-compress-program=zstd -xf $BUNDLE_NAME"
echo "    bash bootstrap.sh"
echo "═══════════════════════════════════════════════════════════════"
