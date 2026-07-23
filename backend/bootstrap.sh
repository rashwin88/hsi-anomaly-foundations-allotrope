#!/usr/bin/env sh
# Allotrope bootstrap one-shot.
#
# Runs ONCE per `docker compose up`. The bootstrap service in
# docker-compose.yml is set with `restart: "no"` so it exits after the
# script completes; api + worker `depends_on` it with
# `condition: service_completed_successfully`, so they boot only after
# the schema + seeds are in place.
#
# Idempotent end-to-end:
#   - alembic upgrade head     — no-ops if already at head
#   - seed-admin               — refuses to clobber an existing admin
#   - seed-action-templates    — upserts (insert-or-update by type+name)
#
# Re-running the bootstrap (e.g. `docker compose restart bootstrap`)
# is safe.

set -eu

echo "[bootstrap] waiting for postgres at $POSTGRES_HOST:$POSTGRES_PORT..."
# Loop until psycopg can open a real connection. depends_on's
# `service_healthy` already waited for postgres' own pg_isready, but
# the listener occasionally accepts pre-warmup. A short retry loop
# closes the race.
#
# Probe via a standalone .py file rather than inline -c so we don't
# fight shell quoting. Captures import errors visibly on failure.
PROBE=/tmp/allotrope-bootstrap-probe.py
cat > "$PROBE" <<'PYEOF'
import os, sys, psycopg
try:
    psycopg.connect(
        host=os.environ['POSTGRES_HOST'],
        port=int(os.environ['POSTGRES_PORT']),
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        dbname=os.environ['POSTGRES_DB'],
        connect_timeout=2,
    ).close()
except Exception as e:
    print(f"probe: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

attempt=1
while [ "$attempt" -le 60 ]; do
  if python "$PROBE"; then
    echo "[bootstrap] postgres reachable on attempt $attempt"
    break
  fi
  if [ "$attempt" = "60" ]; then
    echo "[bootstrap] FATAL: postgres unreachable after 60 attempts" >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 1
done

echo "[bootstrap] running alembic upgrade head..."
alembic upgrade head

echo "[bootstrap] seeding admin user (idempotent)..."
# seed-admin exits 0 when an admin already exists; the CLI prints a
# YELLOW notice in that case. Treat both outcomes as success.
python -m allotrope.cli seed-admin || {
  echo "[bootstrap] seed-admin returned non-zero — likely missing ADMIN_* env vars" >&2
  echo "[bootstrap] continuing; api will refuse logins until you set ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD and re-run bootstrap" >&2
}

echo "[bootstrap] seeding action templates (idempotent upsert)..."
python -m allotrope.cli seed-action-templates

echo "[bootstrap] done. api + worker may now start."
