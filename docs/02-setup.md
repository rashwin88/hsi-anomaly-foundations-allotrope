# 2. Get it running

> **Read this first.** On a clean clone, `docker compose up` **fails** — the frontend image
> can't build. See [Before you start](#before-you-start).

## Prerequisites

Docker Desktop with ≥8 GB allocated. That's it — Python and Node run inside containers.

Bash is needed for the `scripts/*.sh` tooling (WSL or Git Bash on Windows).

## Before you start

`frontend/src/lib/elkLayout.ts` is missing, so `npm run build` — and therefore the frontend
Docker build — fails. Root cause is `.gitignore:22` (`lib/`, a bare pattern that matches at
any depth). Fix it properly:

```bash
# 1. change .gitignore line 22 from  lib/  to  /lib/
# 2. write frontend/src/lib/elkLayout.ts  (wraps elkjs; feeds the /models/:arch diagram)
```

Or skip the UI for now and drive the API directly:

```bash
docker compose -f docker/docker-compose.yml up -d postgres bootstrap api worker
```

## First run

```bash
cp docker/.env.example docker/.env
```

Fill in **three** values — the first two are enforced by compose, the third is enforced by
reality:

| Variable | Why |
|---|---|
| `POSTGRES_PASSWORD` | compose refuses to start without it |
| `JWT_SECRET` | compose refuses to start without it |
| `ADMIN_PASSWORD` | documented as optional, but empty ⇒ **no login exists** |

`openssl rand -base64 32` is a fine generator for the first two.

```bash
docker compose -f docker/docker-compose.yml up -d
```

Compose auto-loads `docker/.env` (it sits beside the compose file) — no `--env-file` needed.
First build takes **30–45 minutes**: the images are Python 3.14 and cp314 wheels don't exist
yet for numpy/h5py/rasterio, so they compile from source.

Open **<http://localhost:3010>** and log in with `ADMIN_USERNAME` / `ADMIN_PASSWORD`.

## Always check bootstrap after the first up

```bash
docker compose -f docker/docker-compose.yml logs bootstrap
```

`bootstrap` runs once, in order: wait for Postgres → `alembic upgrade head` → `seed-admin` →
`seed-action-templates`. All idempotent, re-run on every `up`.

**`seed-admin` failure is swallowed.** Bootstrap still exits 0, the stack comes up healthy,
and every login returns 401. The only symptom is two stderr lines in that log. If you hit
it, fill the `ADMIN_*` vars and:

```bash
docker compose -f docker/docker-compose.yml up -d --force-recreate bootstrap
```

Note `seed-admin` keys on *"does any admin exist"*, not on username — changing
`ADMIN_USERNAME` later does nothing.

## Verify

```bash
curl http://127.0.0.1:8010/healthz         # {"status":"ok"}          liveness
curl http://127.0.0.1:8010/healthz/db      # {"status":"ok","db":"connected"}
```

The **worker has no HTTP endpoint** — it's queue-driven. Check it three ways:

```bash
docker compose -f docker/docker-compose.yml logs worker   # "worker starting (id=… poll=2.0s …)"
```

or watch `jobs.last_heartbeat_at` in Postgres (`127.0.0.1:5432`) while a job runs, or read
`GET /metrics/workload` — a growing `oldest_queued_age_seconds` with `queue_depth > 0` means
the worker is dead or stuck.

## Load your first Scene

**Through the UI only** — there is no CLI for onboarding. Log in → **Scenes** → Ingest →
pick a sensor → upload.

What each sensor expects:

| Sensor | Upload |
|---|---|
| `prisma` | a single `.he5` file |
| `landsat9` | a single `.tif` file |
| `enmap` | the **folder** containing `*-METADATA.XML` |
| `aviris_ng` | the **folder** containing `*_corr_*.bin` |
| `hotsat1` | the **folder** containing `metadata.json` + GeoTIFFs |

Upload returns `202` with a `job_id`; the work happens in the worker. Watch the Jobs page.
nginx allows 20 GB bodies with 30-minute timeouts — an AVIRIS-NG folder is ~6.5 GB.

Onboarding moves files from `/data/staging/<job_id>/` to `/data/scenes/<scene_id>/raw/`,
builds the vendable, renders a thumbnail, and inserts the Scene row. **On failure the moved
files are left in place deliberately**, for inspection.

## The dev loop

**Backend — always rebuild.** There is no bind-mount and no `--reload`; the container runs
what was copied in.

```bash
docker compose -f docker/docker-compose.yml up -d --build api      # api change
docker compose -f docker/docker-compose.yml up -d --build worker   # worker change
```

Changed something under `backend/allotrope/` or `app/`? **Both** images embed it — rebuild
both. Touched `alembic/`, `cli.py` or `bootstrap.sh`? Rebuild `bootstrap` too.

**Frontend — `npm run dev` cannot reach the API.** `vite.config.ts` has **no
`server.proxy`**, and `API_BASE` is hard-coded to `/api`, which only nginx serves. So
`npm run dev` gives you a UI that can't log in. Either add a proxy to `vite.config.ts`
pointing at `http://127.0.0.1:8010`, or rebuild the container:

```bash
docker compose -f docker/docker-compose.yml up -d --build frontend
```

Also note `npm run dev` skips `tsc -b`, so it starts despite the broken import above — the
failure only appears when you navigate to `/models/:architecture`.

## GPU

Opt in with a second `-f`:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d
```

Needs an NVIDIA driver **and** `nvidia-container-toolkit`. On Windows, macOS, or any box
without the toolkit, **just omit the override** — that's why it's a separate file. Torch
falls back to CPU with the same image and code.

## Things that will confuse you

| Symptom | Cause |
|---|---|
| Models page is empty | `allotrope_models` volume is empty on a clean clone — checkpoints only arrive via a snapshot bundle |
| Anomaly scoring always fails | It's broken independently of everything else — [issue #1](09-known-issues.md) |
| `reset-stack.sh` hangs | It polls the wrong port (`:8000`, should be `:8010`) and needs interactive bash |
| A job "disappeared" | Handler exceptions are caught and written to the job row. Check `GET /jobs/{id}`, not the worker exit code |

To wipe and start over: `./docker/reset-stack.sh` (keeps `allotrope_models` unless you pass
`--include-models`).

---

**Next:** [3. Data pipeline](03-data-pipeline.md) · [6. Backend](06-backend.md)
