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

**Worker status.** Besides the per-job heartbeat, each worker runs one more daemon thread
(`allotrope_worker/status_sampler.py`) that every `WORKER_STATUS_INTERVAL_SEC` (5 s), idle or
busy, samples `nvidia-smi` and the torch device its models use and upserts its row in
`worker_status` (migration `20261101_10`). `GET /metrics/host` reports the api host's own
nvidia-smi if it lists a GPU, else the freshest worker row (`source: "worker"`, `worker_id`,
`age_seconds`, `torch_device`), and a row older than 60 s as unavailable. The reason: the GPU
compose override gives the GPU to the worker only, so the api's own nvidia-smi always said "no
GPU" (`allotrope/gpu_status.py`; docs/proposals/class-driven-flow.md addition E).

**Plain failures.** An action that raises `PlainActionError` (`action_types/_meta.py`) gets its
message as `failure_reason` with no `ValueError:` prefix, so the card's first line is the
sentence the user acts on. Used for the 165-grid band filter under Indradhanu V2 and for
`exclude_classes` without a V2 segmentation.

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

Keep the UI text short (user request, 2026-11-01): `label` a few words, `short_description`
one plain sentence, `when_to_use` one short sentence or empty, input labels / descriptions a
few words. The UI shows `short_description`, not `description`; the long explanation belongs
in the module docstring and the LLD. `NewActionDialog` renders every `default_config` key
that is not an input behind a collapsed **Advanced** disclosure, pre-filled with the default,
unless the dialog's `BASIC_FIELDS` lists it as a visible switch or `OWNED_FIELDS` gives it to
a type-specific block. `seed-action-templates` matches system templates by type and the
" · default · <sensor>" suffix, so relabelling renames them in place.

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

## Adding a sensor

There is no sensor registry: a sensor is a `sensor_type` string (plain `Text`, no migration)
that a dozen places switch on. This is the list Tanager touched on 2026-11-01
([`lld/tanager.md`](lld/tanager.md)); grep an existing slug (`aviris_ng`) to check nothing new
has joined it.

**`app/`**
- a `FileHelper` (`app/utils/files/`) and, if it needs typed metadata, a model in
  `app/models/file_processing/` — every class and enum it imports must exist; a missing one
  killed the worker before;
- a `DatasetBuilder` (`app/utils/dataset_builder/`) returning a vendable with ascending
  `band_cw_order`, FWHM > 0 (splib07 silently skips the cache otherwise) and NaN counted
  invalid;
- `FileSourceConfig.infer_file_category` (`app/models/file_processing/sources.py`) must
  accept the path, or every builder construction fails validation;
- a branch in `app/georef/from_scene_file.py`, or exports fall back to pixel coordinates (no CRS);
- `_guess_sensor` in `app/utils/visualization/hyperspectral_visualizer.py` if band count
  would mislabel it.

A second file format for a sensor that already exists is not a new slug: it goes behind
the sensor's helper, which picks the reader from what was uploaded. Tanager reads Planet's
HDF5 or an ENVI `.hdr` + data this way (`app/utils/files/tanager_envi.py`); the switch
points are the helper, `FileSourceConfig`, the sensor's file picker (shared by
`source_path`, `_filename_for_id` and the georef branch) and `_SENSOR_FILE_HINTS`.

**`backend/`**
- `api/scenes.py` `_SUPPORTED_SENSORS`; `cli.py` `sensor_labels` (template names);
- `foundation_models/resolver.py` `_SCENE_SENSOR_FAMILY`;
- `sensors/source_path.py` `resolve_source_path`;
- `allotrope_worker/scene_onboard.py`: `_SENSOR_FILE_HINTS`, `_filename_for_id` (must equal
  the builder's `stac_item.id`), `_build`, `_cloud_cover_pct`, `_native_projection`, and the
  hyperspectral tuple that builds the splib07 cache;
- `allotrope_worker/visualizations.py` hyperspectral tuple;
- each action type's `accepted_sensor_types` + `default_config_per_sensor`
  (`band_filter_apply` also needs its `run()` builder branch), then
  `python -m allotrope.cli seed-action-templates`.

**`frontend/src/`** — `types.ts` `SensorType`; `IngestPanel` options and `accept`;
`NewActionDialog`'s sensor → model-family map (without it the model picker is empty); the
sensor label maps in `NewProjectDialog`, `OnboardingDrawer`, `ScenesPage`,
`SceneDetailPage` (and its `isHyperspectral`), `ProjectWorkspacePage`, `ProjectsPage`.

**Tests** — add the helper and builder to `SENSOR_MODULES` in
`tests/test_backend/test_imports.py`.

**Before adding a slug, check `generic_hsi` does not already cover it.** It reads any GeoTIFF,
ENVI or HDF5 / NetCDF cube with its band table wherever it lives, decides FWHM, scale, no-data
and georef, and reports every decision ([`lld/generic-hsi.md`](lld/generic-hsi.md)). A native
slug earns its place when the provider ships quality masks or a grid the generic reader cannot
see (Tanager's cloud masks, PRISMA's swath). `generic_hsi` sits in every list above; two of its
switch points are data-driven rather than per-sensor: a vendable with `units="not_reflectance"`
makes `anomaly_scoring` skip foundation models, and `spectral_library_match` adds a warning.

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

**The version lives in one place.** `allotrope/version.py` (`APP_NAME`, `__version__`, 1.0.0
since 2026-11-01, 1.0.1 the same day, 1.0.2 for the HotSat-1 fixes). Every export MANIFEST's `software_version` is `allotrope/<version>`;
`GET /version` (unauthenticated) returns `{name, version}`; `frontend/src/version.ts` and
`frontend/package.json` mirror it by hand.

## Exports

`POST /actions/{id}/export` runs synchronously in the api (`api/action_export.py`):

| action | params | bundle |
|---|---|---|
| `anomaly_detection_prep` (committed) | — | thermal / prep bundle, unchanged |
| `spectral_library_match` | none, or `subset=all` | the full bundle, schema `/1`, `/2` or `/3` (lld/spectral-namer-in-product.md) |
| `spectral_library_match` | `subset=manmade\|natural` | the namer's man-made / natural objects, boolean mask |
| `spectral_library_match` | `classes=<list>`, `confident_only`, `manmade_p` | the class-driven pack (`api/action_class_pack.py`, `app/spectral_match/class_pack.py`) |

`classes` is a comma list of `natural`, `manmade` and the man-made material keys of
`app/spectral_match/namer/classes.py` `MANMADE_CLASSES` (read per request, so a new class is a
token with no api change). `confident_only` (default false) drops tentative man-made objects.
`manmade_p` (0-1, default the action's 0.5) re-decides every object's call on its mean
P(man-made). An unknown token, `classes` with `subset`, `confident_only` / `manmade_p` without
`classes`, or an action without namer results is a 422 that says which.
`GET /actions/{id}/spectral_library_match/selection_counts` takes the same three params and
returns `{objects, pixels, per_class, manmade_p, slug}` from the same selection code, so the
match screen's "Export oil + plastic (64 objects)" is the count the zip holds.

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

Postgres 16, SQLAlchemy 2.0 + psycopg v3, **sync**. 13 entities. Alembic, linear migrations.

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
            projects/<pid>/actions/<aid>/story/   story GIF cache, never exported (lld/story-gif.md)
            projects/<pid>/visualizations|exports/
```

Deleting a project is one `rm -rf projects/<pid>/` — the layout is designed so there are no
orphans.


## The raw-dataset inventory — a third, separate service

`backend/allotrope_inventory/` is **not** part of the api/worker pair. It shares no code,
no database and no volume with them, and nothing in the main stack depends on it. It is the
catalogue of raw data we hold, and the bytes live in **`s3://allotrope-raw-inventory-india`**
(Prod account, ap-south-1). Inventory only: no sharding, no vendables, no processing.

| | |
|---|---|
| Database | **Supabase** Postgres 17 + PostGIS (Mumbai), via the *session pooler* — the direct host is IPv6-only |
| Migrations | `backend/alembic_inventory/`, run with `alembic -c alembic-inventory.ini upgrade head` (the prod container runs this on start) |
| Auth | `X-Inventory-Key` header; keys are `name:key` pairs in `INVENTORY_API_KEYS`, and the name is the actor in every audit row |
| Config | `INVENTORY_*` env vars; local runs read `backend/.env.inventory` (gitignored); prod reads SSM Parameter Store |
| Port | 8020, bound to 127.0.0.1. The browser reaches a **read-only** slice through nginx at `/inventory-api/` |

**Every dataset has a `kind`**, and filters keep them apart:

| kind | What one row is | Example |
|---|---|---|
| `scene` | one acquisition, as the provider delivered it | a PRISMA `.he5`, an EnMAP scene folder |
| `archive` | a packaged delivery kept as shipped | the 9-part SpectralEarth `.tar.gz` (now retired) |
| `collection` | many small items unpacked from an archive, each its own file | `SPECTRAL_EARTH_PATCHES`: 539,755 patches |

Statuses: `pending` → `available` (or `failed`); `retired` once a dataset's payload is
deleted — its row, file list, manifests and audit trail stay as the record.

**API.** All under the service root; ids are `ds_<uuid>`.

| Method | Path | Does |
|---|---|---|
| GET / POST | `/sensors` | list / add a sensor (slugs match `scenes.sensor_type`) |
| POST | `/datasets` | register → `pending`, fixes its S3 prefix; body carries `kind` (default `scene`) |
| GET | `/datasets` | filter by `sensor_type`, `processing_level`, `status`, `kind`, `tag` (repeat = AND), `q`, `bbox`, `acquired_from/to` |
| GET | `/datasets/{id}` | row + tags + **file counts and bytes** + sources (no file list — see `/files`) |
| GET | `/datasets/{id}/files` | the files, paged (`limit` ≤ 1000) and searchable by path (`q`, e.g. a site id) |
| GET | `/datasets/{id}/events` | audit trail, oldest first |
| POST | `/datasets/{id}/uploads` | declare files → presigned multipart part URLs; **re-call to resume** |
| POST | `/datasets/{id}/imports` | copy one object in from another S3 bucket **server-side**; resumable |
| POST | `/datasets/{id}/files/bulk` | register ≤ 5,000 already-stored single-PUT files, each verified by HEAD |
| POST | `/datasets/{id}/finalize` | complete + verify open uploads, write the manifests → `available` |
| POST | `/datasets/{id}/retire` | delete the payload (every version), keep the record; `confirm` must repeat the scene id |
| GET | `/datasets/{id}/download` | version-pinned presigned GETs |
| PUT / DELETE | `/datasets/{id}/tags/{tag}` | idempotent; tag is `name` or `key:value` |
| GET | `/tags`, `/stats` | tags with usage counts; counts and bytes by sensor × status |

**Bucket layout.** `<sensor>/<level>/ds_<uuid>/payload/<original paths>`, plus
`manifest.json` (the summary) and `files.jsonl.gz` (one line per file) beside it. Those
two are the record of truth; the database is an index rebuildable from the bucket. Why,
and how integrity is enforced at every step: [`lld/inventory.md`](lld/inventory.md).

**CLI** — talks HTTP, plus boto3 for the S3 commands. Run from `backend/` with
`INVENTORY_URL` and `INVENTORY_KEY` set (on the prod box: `/opt/inv-cli/bin/python`,
key from `/allotrope/prod/inventory_api_keys`):

```bash
# a delivery on local disk (or pulled from Drive) — presigned multipart upload
python -m allotrope_inventory.cli ingest ./PRS_L2D_…/ --sensor prisma --level L2D \
    --scene-id PRS_L2D_STD_… --acquired-at 2025-03-01T05:12:00Z --footprint fp.geojson \
    --meta cloud_cover_pct=3.2 --source gdrive://<id> --tag site:jharia

# objects already in S3 — hashed in place, provider .md5s checked, copied server-side
python -m allotrope_inventory.cli import-s3 s3://bucket/prefix/ --sensor enmap \
    --level SpectralEarth --kind archive --scene-id SPECTRAL_EARTH --tag provider:dlr

# an archive dataset → a collection, one verified object per member
python -m allotrope_inventory.cli unpack-archive ds_<archive> --scene-id SPECTRAL_EARTH_PATCHES

python -m allotrope_inventory.cli ls --kind scene --sensor prisma --bbox 86,23,87,24
python -m allotrope_inventory.cli download ds_<uuid> ./out/
```

Every write command is resumable: re-run the same command after an interruption and only
the missing work is done. Registration fields (metadata, footprint, acquisition time) are
fixed at first registration; there is no `PATCH` yet.

---

**Next:** [7. Frontend](07-frontend.md) · [5. Detectors](05-detectors.md)
