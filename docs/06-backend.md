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

---

**Next:** [7. Frontend](07-frontend.md) · [5. Detectors](05-detectors.md)
