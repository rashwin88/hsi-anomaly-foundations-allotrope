# 6. Backend — API, worker, and Actions

Two processes share one package. `backend/allotrope/` is imported by **both**;
`backend/allotrope_worker/` is worker-only.

| Process | Entrypoint | Job |
|---|---|---|
| **api** | `uvicorn allotrope.main:app` | HTTP, reads/writes DB, enqueues jobs |
| **worker** | `python -m allotrope_worker` | does the heavy compute |

## The queue is Postgres

There is **no Redis, Celery or RQ**. The `jobs` table *is* the queue.

- **Claim** — `SELECT … WHERE status='queued' … FOR UPDATE SKIP LOCKED`, then set
  `running` and **commit immediately**. The row lock is released at once; ownership is
  afterwards expressed by *heartbeat freshness*, not by holding a lock.
- **Heartbeat** — one daemon thread per in-flight job updates `last_heartbeat_at`.
- **Reap** — jobs whose heartbeat went stale are flipped to `failed`, and the failure is
  mirrored onto `actions.status` so the UI doesn't show an eternally-running Action.

Four job types: `scene_onboard`, `annotation_attach`, `action_run`, `project_export`.

**Handler rule:** handlers may `add`/`flush`/`execute` but **must not commit or rollback**.
The runner owns the commit. (`action_run` breaks this deliberately twice — an early
`queued→running` commit so the UI updates, and one in the except branch to save
`failure_reason`.)

## Actions are the extension point

Everything a user runs is an **Action**. To add a capability you add an action type, not a
route. Each lives in `backend/allotrope/action_types/<kind>.py` and must export:

```python
KIND = "my_action"
META = ActionTypeMeta(...)          # drives the UI picker, the Action card, AND seeded templates
def validate_config(raw_cfg, sensor_type) -> Config: ...
def run(ctx) -> None: ...           # the actual work
def summarize(ctx, out) -> dict: ...
def preview(ctx, out) -> dict: ...
TERMINAL_STATUS = "complete"        # optional; anomaly_detection_prep uses "needs_threshold"
```

Register it in `action_types/__init__.py`. `META` is the single source of truth — the
frontend action picker is generated from it, so you do not touch frontend code to add one.

The six shipped kinds: `band_filter_apply`, `scene_segmentation`, `cloud_mask`,
`anomaly_scoring`, `anomaly_detection_prep`, `spectral_library_match`.

### The lazy-import rule — important

Action modules must **not** import `app.*`, `torch` or `rasterio` at module top level, only
inside `run()`/`summarize()`/`preview()`. The api imports every action module at startup;
top-level heavy imports would load torch into the web process.

That is why you'll see paired files: `anomaly_scoring.py` (light — schema, validation,
`META`) and `_anomaly_scoring_run.py` (heavy — the real implementation, imported lazily).
Follow the pattern.

`ctx` (`ActionRunContext`) gives you `configuration`, `scene_raw_path`, `output_dir`,
`on_step(msg)` for progress, and `resolve_action_output(wire_id)` to read an upstream
Action's output — that last one is how the DAG is wired.

## Conventions you will trip over

**Wire IDs are prefixed, never bare UUIDs.** `scene_<uuid>`, `project_<uuid>`,
`action_<uuid>`, `job_<uuid>`, `export_<uuid>`. Parse and format with
`api/wireformat.py`. Returning a raw UUID is a bug.

**No `/api` prefix in FastAPI.** nginx proxies `/api/*` → `api:8000` and strips the prefix.
Same origin, so there is also **no CORS middleware**. Don't add one; don't hardcode `/api`
in a response body.

**Every handler is sync `def`.** There is no async anywhere. Don't introduce one endpoint's
worth of `async` — the DB session is sync and will block the loop.

**Status columns are `Text`, not enums**, deliberately — adding a type stays code-only with
no migration.

## Auth

Stateless HS256 JWT in an HttpOnly, `SameSite=Strict` cookie. argon2 passwords.
No refresh tokens, no session table, no revocation.

Two things to know:
- `is_admin` is **baked into the claims**, so a demoted admin keeps admin until the token
  expires (≤24 h).
- There is **no per-user authorization**. `require_admin` guards exactly one endpoint
  (`POST /admin/users`); any authenticated user can read or delete any Project, Scene or
  Action. RBAC is deliberately deferred — don't assume it exists.

## Database

Postgres 16, SQLAlchemy 2.0 + psycopg v3, **sync**. 12 entities. Alembic, linear migrations.

`database_url` is assembled in Python from the discrete `POSTGRES_*` variables — there is no
real `DATABASE_URL` env var, so special characters in the password are encoded correctly.

`Settings()` is constructed at **import time**, so a missing `POSTGRES_PASSWORD` or
`JWT_SECRET` crashes any process on import — including the worker, which doesn't otherwise
need JWT.

## Where output goes

Paths are always stored **relative**; two volumes hold the bytes.

```
/data       scenes/<id>/raw|vendable|annotations/ , staging/<job_id>/
/artifacts  scenes/<id>/thumbnail.png
            projects/<pid>/actions/<aid>/output/
            projects/<pid>/visualizations|exports/
```

Deleting a project is one `rm -rf projects/<pid>/` — the layout is designed so there are no
orphans.


## The raw-dataset inventory — a third, separate service

`backend/allotrope_inventory/` is **not** part of the api/worker pair. It shares no code,
no database and no volume with them, and nothing in the main stack depends on it. It is the
catalogue of raw scene deliveries — one row per delivery, exactly as the provider shipped
it — and the bytes live in **`s3://allotrope-raw-inventory-india`** (Prod account,
ap-south-1). Inventory only: no sharding, no vendables, no processing.

| | |
|---|---|
| Database | **Supabase** Postgres 17 + PostGIS (Mumbai), via the *session pooler* — the direct host is IPv6-only |
| Migrations | `backend/alembic_inventory/`, run with `alembic -c alembic-inventory.ini upgrade head` |
| Auth | `X-Inventory-Key` header; keys are `name:key` pairs in `INVENTORY_API_KEYS`, and the name is the actor in every audit row |
| Config | `INVENTORY_*` env vars; local runs read `backend/.env.inventory` (gitignored) |
| Port | 8020 |

**API.** All under the service root; ids are `ds_<uuid>`.

| Method | Path | Does |
|---|---|---|
| GET / POST | `/sensors` | list / add a sensor (slugs match `scenes.sensor_type`) |
| POST | `/datasets` | register a delivery → `pending`; fixes its S3 prefix |
| GET | `/datasets` | filter by `sensor_type`, `processing_level`, `status`, `tag` (repeat = AND), `q`, `bbox`, `acquired_from/to` |
| GET | `/datasets/{id}` | row + files + sources + tags |
| POST | `/datasets/{id}/uploads` | declare files → presigned multipart part URLs; **re-call to resume** |
| POST | `/datasets/{id}/finalize` | complete + verify every upload, write `manifest.json` → `available` |
| GET | `/datasets/{id}/download` | version-pinned presigned GETs |
| PUT / DELETE | `/datasets/{id}/tags/{tag}` | idempotent; tag is `name` or `key:value` |
| GET | `/tags` | tags with usage counts |

**Bucket layout.** `<sensor>/<level>/ds_<uuid>/payload/<original paths>` plus
`<sensor>/<level>/ds_<uuid>/manifest.json`. The manifest is the record of truth; the
database is an index rebuildable from the bucket. Why, and how integrity is enforced:
[`lld/inventory.md`](lld/inventory.md).

**CLI** — HTTP only, needs just `requirements-inventory-cli.txt`. Run from `backend/` with
`INVENTORY_URL` and `INVENTORY_KEY` set:

```bash
python -m allotrope_inventory.cli ingest ./PRS_L2D_…/ --sensor prisma --level L2D \
    --scene-id PRS_L2D_STD_… --acquired-at 2025-03-01T05:12:00Z --footprint fp.geojson \
    --meta cloud_cover_pct=3.2 --source gdrive://<id> --tag site:jharia
python -m allotrope_inventory.cli ls --sensor prisma --bbox 86,23,87,24 --tag site:jharia
python -m allotrope_inventory.cli download ds_<uuid> ./out/
```

`ingest` is resumable — re-run the same command after an interruption and only missing
parts are sent. Registration fields (metadata, footprint, acquisition time) are fixed at
first registration; there is no PATCH yet.

---

**Next:** [7. Frontend](07-frontend.md) · [5. Detectors](05-detectors.md)
