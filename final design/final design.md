---
title: Final Design — Allotrope Product (UI + Backend, Offline Demo)
status: in-progress (running notes)
started: 2026-05-08
---

# Allotrope — Final Product Design (Running Notes)

This is a running store of design decisions as we shape the product. Newest sections appended at the bottom; do not reorganize prior entries — the chronology matters.

## Companion docs
- [Demo Storyboard](storyboard.md) — chronological storyboard ideation; rationale, decisions, and discussion log.
- [Storyboard Spec](storyboard-spec.md) — **build-ready distillation** of the storyboard's locked decisions. Read this when implementing; read storyboard.md for context and history.
- [Abstractions](abstractions.md) — running discussion on entity identity, lifecycle, invariants, and relationships. Bridges the storyboard spec to the schema and API.
- [ER Diagram](er-diagram.drawio) — visual entity-relationship diagram. Updated alongside abstractions.md as entities lock.

---

## 2026-05-08 — Session 1: Repo & Container Topology

### Context
Wrapping the existing HSI anomaly pipeline into a demo-able product:
- Frontend (UI) ↔ Backend (APIs/services), classic split.
- **Demo constraint: no internet access.** Whole stack must run from a single self-contained docker-compose bring-up.

### Proposed top-level layout
```
hsi-anomaly-foundations-allotrope/
├── backend/        # FastAPI (or similar) services, model serving, pipeline orchestration
├── frontend/       # UI app (framework TBD — React/Vue/Svelte)
├── docker/         # Dockerfiles, compose, entrypoints, demo bootstrap scripts
├── app/            # (existing) library code reused by backend
├── ...             # existing dirs (configs, checkpoints, tests, etc.)
└── docker-compose.yml
```

### Storage decision (open)
User's first instinct: **one docker volume as central storage for all files & databases.**

Working through the tradeoffs before committing:

- **One-volume option (simplest):**
  - Pros: dead-simple bring-up, one place to seed demo data, one mount to back up.
  - Cons: conflates concerns. Database files, raw HSI data, processed shards, model checkpoints, and user-uploaded files all share a lifecycle. Hard to swap one (e.g. replace demo data) without touching the rest. DB corruption risk if anything writes into its dir.

- **Per-concern volumes (recommended starting point):**
  - `allotrope_db`        — Postgres/SQLite data dir (if a DB is needed at all).
  - `allotrope_data`      — raw + processed HSI files (PRISMA HE5, Landsat TIF, EnMAP folders, webdataset shards).
  - `allotrope_models`    — model checkpoints (already large; see existing `checkpoints/`).
  - `allotrope_artifacts` — run outputs, anomaly maps, exportable user artifacts.
  - All declared in `docker-compose.yml` under `volumes:`. Backend mounts what it needs; frontend talks to backend, not to the volume directly.

  - This is still "central storage" from the user's mental model — just partitioned so we can rotate one without nuking the others.

### Offline-demo implications (must not be forgotten)
1. **Pre-build images.** No `docker pull` at demo time. Images either pre-loaded via `docker load` from `.tar` archives, or built against a local registry on the demo machine.
2. **No remote pip/npm installs at runtime.** Wheels and node_modules baked into images at build time. `pip install` should fail-closed if it ever tries to reach the network during demo.
3. **Pre-seed volumes.** A `bootstrap` service (one-shot, runs once) populates `allotrope_data` and `allotrope_models` from baked-in tarballs the first time the stack comes up. After that, it's a no-op.
4. **No external API calls at runtime.** USGS M2M, S3 fetches, etc. — all of these need a local stand-in or pre-cached fixtures for the demo. Flag any code path that hits the network so we can stub it.
5. **One-command bring-up.** `docker compose up` (or a wrapper script) — the demo machine operator should not have to do anything else.

### Open questions to resolve next
- Is a database actually needed, or can the demo run from the filesystem + in-memory? (Affects whether we need the `allotrope_db` volume at all.)
- Frontend framework choice + how it talks to backend (REST? WebSocket for streaming patches/anomaly maps?).
- Backend framework — FastAPI is the obvious Python choice given the existing stack. Confirm.
- How big is the demo dataset? Drives whether we bake data into the image vs. seed a volume from a tarball.
- GPU availability on the demo machine? (Affects whether model serving is CPU-only or GPU-accelerated; existing code is MPS/CUDA-aware.)

### Tentative agreement (to confirm with user)
- ✅ `backend/` and `frontend/` as top-level folders.
- ✅ Self-contained docker stack for offline demo.
- ⚠️ **Refine** "one volume for everything" → multiple named volumes partitioned by concern, but all managed by the same compose file so it still feels like one storage layer to operate.

---

## 2026-05-08 — Session 2: Locked-in answers + shippable bundle

### Decisions (locked in this session)
1. **Database: yes.** A real DB is in scope. Likely Postgres (most flexible; runs fine in a container; supports JSONB for metadata blobs). SQLite is a fallback only if we discover the demo has no persistent state worth a server. Default to Postgres.
2. **Frontend: React.** Vite + React + TypeScript as the starter stack (fast dev server, simple prod build, no Next.js server runtime to babysit for an offline demo).
3. **Compute: GPU-preferred, CPU-fallback.** Backend must detect device at startup and degrade gracefully. Existing code is already MPS/CUDA-aware — extend that to the serving layer.
4. **Data seeding: volumes, not baked into images.** User will sometimes be handed a fresh dataset on the spot, so we need a path to drop new files into the seeded volume without rebuilding images.

### New requirement — runtime resource monitoring
Demo audiences will ask "is it using the GPU?" — we need a visible answer.

- **Backend exposes `/metrics`** with: CPU %, RAM, GPU util %, GPU memory, per-process model device, current batch device.
  - GPU metrics via `nvidia-smi` parsing or `pynvml` (NVIDIA), `torch.mps.current_allocated_memory()` (Apple), or graceful "no GPU" if neither.
- **Frontend has a small live "System" panel** (top-right, collapsible) showing those numbers, polled every ~1s via WebSocket or SSE.
- Optional: `cAdvisor` + `Prometheus` + `Grafana` containers in the compose stack for deeper monitoring. Probably overkill for a demo — start with the lightweight `/metrics` panel and only add cAdvisor if we have time.
- Log device decisions at request time so we can show "served on GPU" vs "served on CPU" in the UI.

### Shippable bundle — answering "can I tar everything and copy it over?"
**Yes, but it's two artifacts inside one outer tarball, not a single `docker save`.** This is the most-misunderstood part of offline Docker shipping, so writing it down carefully:

#### What `docker save` does and doesn't include
- `docker save` exports **images only**. It does **not** include volume contents. A volume seeded with 40 GB of HSI data is invisible to `docker save`.
- Volumes have to be exported separately by tarring their contents from inside a throwaway container that mounts them.

#### The bundle layout
```
allotrope-bundle.tar.gz
├── images/
│   ├── allotrope-backend.tar       # docker save output
│   ├── allotrope-frontend.tar
│   ├── postgres-16.tar             # base images we depend on, also saved
│   └── ...
├── volumes/
│   ├── allotrope_db.tar.gz         # tar of /var/lib/postgresql/data
│   ├── allotrope_data.tar.gz       # tar of seeded HSI files
│   ├── allotrope_models.tar.gz     # tar of checkpoints
│   └── allotrope_artifacts.tar.gz  # usually empty at ship time
├── compose/
│   ├── docker-compose.yml
│   └── .env.example
├── load.sh                         # one-shot: loads images + restores volumes
└── README-DEMO.md                  # operator instructions
```

#### `load.sh` flow on the demo machine
1. `docker load -i images/*.tar` — register every image.
2. `docker volume create allotrope_db` (and the others) — create empty named volumes.
3. For each volume: run a throwaway `busybox` container that mounts the empty volume + the tarball, and untars into the volume.
   - One-liner per volume:
     ```
     docker run --rm -v allotrope_db:/restore -v "$(pwd)/volumes":/backup \
       busybox sh -c "cd /restore && tar xzf /backup/allotrope_db.tar.gz"
     ```
4. `docker compose up -d` — stack comes up against pre-loaded images and pre-seeded volumes.

#### Build-side script (run before shipping)
A counterpart `package.sh` does the inverse:
- `docker save` each image we depend on.
- For each named volume: spin a throwaway container, tar the volume contents into `volumes/*.tar.gz`.
- Bundle everything into `allotrope-bundle.tar.gz`.

#### Caveats — must not be forgotten
- **Architecture pinning.** If we build on Apple silicon (arm64) and the demo machine is x86_64, images won't run. Build with `--platform linux/amd64` from the start, and verify the demo machine's arch before packaging. Most likely this means we set `platform: linux/amd64` in compose and accept slower builds on Mac (Rosetta/qemu).
- **Postgres volume must be tarred while the DB is shut down**, or the snapshot will be inconsistent. The build-side `package.sh` should `docker compose down` before exporting the DB volume.
- **Bundle size.** Realistically tens of GB once HSI data + model checkpoints + base images are included. Plan for an external SSD, not a USB stick. Compression helps for the volume tarballs (zstd or gzip-9); image tarballs are already compressed internally so re-compressing has marginal gain.
- **"Drop a new dataset on the spot"** flow: there should be a small CLI or backend endpoint that lets the operator point at a folder of new HSI files and triggers ingestion into `allotrope_data` + DB rows. This is separate from the seeded baseline. We'll design this in a later session.
- **First-run vs re-run.** `load.sh` should be idempotent — detect that volumes already exist and skip the restore (or prompt before overwriting). Avoid clobbering work the operator did between demos.

### Updated open questions
- ~~Is the demo machine confirmed Linux x86_64 with NVIDIA?~~ **Confirmed: Linux server, NVIDIA GPU, x86_64.**
- Do we need any auth on the frontend, or is the demo single-user trusted?
- For the "fresh dataset on the spot" path: what file formats should the ingestion endpoint accept up front? (PRISMA HE5, Landsat TIF, EnMAP folder are the existing three — confirm these cover the demo.)

### Locked from this session
- **Target machine:** Linux server, x86_64, NVIDIA GPU.
  - Build images with `--platform linux/amd64` (matters because user develops on Mac).
  - Backend container needs `--gpus all` (or compose `deploy.resources.reservations.devices`) plus the NVIDIA Container Toolkit installed on the demo host. Add a one-line check in the demo README for the operator.
  - Base image: `nvidia/cuda:<version>-runtime-ubuntu22.04` or a PyTorch CUDA image, then layer Python deps on top. Pick the CUDA version to match what's installed on the demo server (need to confirm that number before we build).

---

## 2026-05-08 — Session 3: Source code is NOT in the bundle (clarification)

User asked: "if the images are saved, why would we need source code in `app/` inside the tarball?" — correct, we don't. Pinning this down because it's the most common point of confusion when shipping offline Docker bundles.

### Two separate layouts — keep them distinct

**(A) Dev-machine source layout** — what lives in this git repo, used at *build time*:
```
hsi-anomaly-foundations-allotrope/
├── backend/        # source: API code, Dockerfile
├── frontend/       # source: React app, Dockerfile
├── app/            # existing shared library, COPY'd into backend image
├── docker/         # compose files, scripts
└── ...
```

**(B) Demo-machine bundle** — what we ship, used at *run time*:
```
allotrope-bundle.tar.gz
├── images/         # docker save output — code is already baked into images
├── volumes/        # data, models, db snapshot
├── compose/
│   └── docker-compose.prod.yml   # references image TAGS, not build contexts
├── load.sh
└── README-DEMO.md
```

**No source code crosses from (A) to (B).** The Dockerfile's `COPY app/ /srv/app/` step happened on the dev machine at build time; the resulting image is a self-contained snapshot. `docker load` brings the image (with code inside) onto the demo machine, and it just runs.

### The compose-file gotcha
A `docker-compose.yml` that says:
```yaml
backend:
  build: ./backend         # ← needs source code to exist!
```
…will **fail on the demo machine** because there's no `./backend` directory there.

The shipped compose file must reference pre-built tags only:
```yaml
backend:
  image: allotrope-backend:0.1.0   # ← already loaded via `docker load`
```

**Practical pattern:** keep two compose files in the repo.
- `docker-compose.yml` — dev-machine, has `build:` directives, used to build images.
- `docker-compose.prod.yml` — what we put in the bundle, has `image:` references only. Generated from the dev compose or hand-maintained — both are fine; small enough to keep in sync manually.

`package.sh` (the build-side script) takes care of:
1. Building images via the dev compose.
2. `docker save` each tagged image into `images/`.
3. Copying `docker-compose.prod.yml` into the bundle's `compose/` folder.
4. Tarring volumes (with the DB stopped, per Session 2).
5. Bundling everything into `allotrope-bundle.tar.gz`.

### What about config / secrets?
Config files (`.env`, model config YAML) are a gray area — sometimes you want them inside the image (frozen for the demo), sometimes outside (operator can tweak without rebuilding). Default position: bake demo config into the image; expose only a tiny `.env` file in the bundle for things the operator might genuinely need to change (port numbers, GPU device index). Revisit once we know what's actually variable.

---

## 2026-05-08 — Session 4: Worker image — Actions go through a queue

User-led decision: **Actions are submitted to a queue and consumed by a separate backend image dedicated to running Actions.**

This is a clean architectural split and the right call. Capturing the shape.

### Two backend images, one queue

```
┌──────────────┐    HTTP      ┌──────────────────┐
│   frontend   │ ───────────► │   api (light)    │
└──────────────┘              │  - auth          │
                              │  - CRUD          │
                              │  - submit Action │
                              │  - poll status   │
                              └──────┬───────────┘
                                     │  enqueue
                                     ▼
                              ┌──────────────────┐
                              │   queue (DB or   │
                              │   Redis-backed)  │
                              └──────┬───────────┘
                                     │  pick up
                                     ▼
                              ┌──────────────────┐
                              │  worker (heavy)  │
                              │  - torch + CUDA  │
                              │  - models loaded │
                              │  - executes      │
                              │    Actions       │
                              │  - writes output │
                              │    + status      │
                              └──────────────────┘
```

### Why this split is the right move

- **GPU access concentrates in the worker.** The api image stays lightweight (FastAPI + DB driver). Only the worker needs `--gpus all`, the NVIDIA Container Toolkit, the CUDA runtime base, and the multi-GB ML deps. Smaller blast radius for GPU config; smaller api image to ship.
- **Long-running work is decoupled from the request cycle.** HTTP requests stay snappy (just enqueue and return); Actions can take seconds to minutes without holding open connections. Polling for status is naturally compatible.
- **Scales cleanly later.** If the demo machine ever grows to multi-GPU, we just run more workers. The api stays the same.
- **Easier failure isolation.** Worker crashes don't take down the API. Stuck Action = restart the worker, not the whole stack.
- **The Workload tile from the storyboard fits this exactly.** Queue depth, throughput, and average duration are all natural reads off the queue + DB.

### Queue backing — three options

This is the consequential follow-up question. For an **offline, single-machine** demo, simplest wins, but each option has tradeoffs:

1. **Postgres-backed queue** (using a `jobs` table + `SELECT … FOR UPDATE SKIP LOCKED`).
   - Pros: zero new images, no Redis container, no extra config. Postgres already in the stack. Atomic with the rest of the data model — an Action's status, queue position, and output all live in the same DB. Trivial to query (the Workload tile = `SELECT count(*) WHERE status='queued'`). Survives restarts for free.
   - Cons: not optimized for very high queue throughput (millions/sec) — completely irrelevant for a single-GPU demo machine. No Celery-style ecosystem out of the box.
   - **Recommended.** This is the right choice for our scale and for the offline-bundle goal.

2. **Redis + RQ / Celery.**
   - Pros: standard Python ecosystem; lots of examples. Celery has rich features (retries, scheduling, chains).
   - Cons: another image (`redis`). Another set of config. Job state lives in Redis but Action data lives in Postgres — two sources of truth to keep in sync. None of the extra features pay off in our scale.

3. **RabbitMQ / NATS / etc.**
   - Pros: best-in-class message brokering.
   - Cons: heavy for our scale, another image, another protocol. Overkill.

**Default position: Postgres-backed queue.** Will revisit if/when scale demands it (which won't be in any demo we ship).

**Confirmed by user (2026-05-08):** Postgres-backed queue is locked. Job state lives in a `jobs` (or `actions`) table; workers pull via `SELECT … FOR UPDATE SKIP LOCKED`. No Redis, no RabbitMQ, no Celery in the stack.

### Image set and compose update

The compose stack now becomes:

```yaml
services:
  postgres:        # data + jobs queue
  api:             # FastAPI, lightweight
  worker:          # torch + CUDA, --gpus all
  frontend:        # Vite static build served behind nginx (or by api in dev)
```

Plus the `bootstrap` one-shot from Session 1 for volume seeding.

The `worker` image is the one that needs `--platform linux/amd64`, the CUDA base, and `--gpus all` at runtime. The `api` and `frontend` are small.

### Open questions to resolve when we get to backend service definitions

1. **Worker concurrency model.** One worker process pulling jobs serially (concurrency=1, fits one-GPU machine), or a small pool? My lean: concurrency=1 in v1; revisit if needed.
2. **Cancellation semantics.** A `cancelled` state in the lifecycle is easy if the Action hasn't started yet (just mark the row). Cancelling a *running* Action requires the worker to cooperatively check a flag periodically. Worth knowing: real cancellation mid-execution is genuinely harder than it sounds. Decision: support cancel-while-queued in v1; cancel-while-running is best-effort (mark cancellation requested; worker checks at boundaries).
3. **Job heartbeat / stuck detection.** If the worker dies mid-Action, the job stays `running` forever unless we add heartbeats (worker updates `last_heartbeat_at` every N seconds; reaper marks stale jobs as `failed`). Standard pattern, worth designing in early.
4. **Worker model loading.** If we have multiple model variants in the catalog, do we load all of them into GPU memory at worker startup (fast first inference, big VRAM footprint), or load on demand per Action (slower first inference, smaller idle footprint)? Likely on-demand with an LRU cache. Defer detailed decision.

### Implication for the bundle (Session 2 revisited)

The bundle now ships **four** images instead of three: `postgres`, `api`, `worker`, `frontend`. The `worker` image is the largest by far (CUDA + torch + checkpoints — possibly 5–10 GB). Just naming it so we're not surprised at packaging time.

---

## Appendix A — Concepts explained simply

### busybox (the throwaway container we use for volume export/restore)
**What it is:** a single tiny Linux binary (~5 MB) that bundles ~300 common command-line tools — `sh`, `tar`, `cp`, `ls`, `cat`, etc. — into one program. It's the "Swiss Army knife" of small Linux environments.

**Why we use it for volumes:** Docker volumes can't be tarred from outside the Docker world; you need *some* container to mount the volume and run `tar` on it. We don't want to spin up our heavy backend image just to do that — it'd be slow and wasteful. So we use `busybox` as the smallest possible container that knows what `tar` is.

**What it actually looks like in practice:**
```
docker run --rm \
  -v allotrope_db:/restore \           # mount the (empty) named volume at /restore
  -v "$(pwd)/volumes":/backup \        # mount our local backup folder at /backup
  busybox \                            # use the tiny busybox image
  sh -c "cd /restore && tar xzf /backup/allotrope_db.tar.gz"
```
Translation: "Spin up a tiny throwaway container, give it access to the empty volume and to my local tarball, untar the tarball into the volume, then disappear (`--rm`)." After it exits, the volume is populated and the container is gone. We never run a real workload in busybox — it's just the tar-runner.

You can think of busybox as a disposable janitor we hire for 2 seconds to move files between two places Docker considers separate.

### WAL (Write-Ahead Log) — why we shut Postgres down before tarring its volume
**What it is:** Postgres (and most serious databases) doesn't write your changes straight into the main data files. Instead it does this:

1. You run `INSERT INTO ...`.
2. Postgres writes a description of the change to a **log file** first (the WAL — "Write-Ahead Log") and calls that durable.
3. Some time later (seconds, sometimes longer), it actually applies the change to the real data files.

This is faster (logs are sequential writes — fast) and safer (if the power dies between step 2 and step 3, Postgres can replay the WAL on next boot and recover).

**Why this matters for our tarball:** if we tar the Postgres data directory while the database is **running**, we might catch it mid-flight — the data files are partially updated, the WAL has entries that haven't been applied yet, and snapshotting is happening file-by-file so different files get caught at different moments in time. The result is an **inconsistent** snapshot: when the demo machine starts Postgres against this tarball, it might refuse to boot, lose recent rows, or worse — appear to work but silently miss data.

**The fix is boring and reliable:** before exporting the DB volume, run `docker compose down` (or just stop the postgres container). That flushes the WAL into the data files cleanly and stops new writes. Now everything in the volume is in a consistent state. Tar it. Bring the stack back up.

Mental model: it's like photocopying a book. If you photocopy while someone is still writing in it and flipping pages, your copy will have half-written sentences and pages out of order. Close the book first, then copy.

(Postgres does have proper "online backup" tools — `pg_basebackup`, etc. — but for a demo bundle, just stopping the DB before tarring is dramatically simpler and equally correct.)

---
