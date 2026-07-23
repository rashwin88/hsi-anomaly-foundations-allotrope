# Codebase investigation spec — for a fresh agent

A self-contained brief for another agent (or human) walking into the
Allotrope repo cold. Use this as a hand-off prompt to drive a top-to-
bottom audit of the frontend and backend without re-discovering basic
facts.

The investigator should produce a written report covering each section
below in order. Cite specific files + line numbers wherever a claim
hinges on code. Keep findings concrete; flag assumptions explicitly.

---

## 0 · Orientation

**Repo root:** `hsi-anomaly-foundations-allotrope`

**Read these first, in order:**
1. `final design/CONTEXT.md` — current build state, file map, working rules.
2. `final design/ROADMAP.md` — what's shipped (🟢), what's deferred (⚪), what's next (🟡).
3. `final design/abstractions-spec.md` — the entity model + cascade rules. Authoritative.
4. `final design/storyboard-spec.md` — the user-facing screens this stack drives.
5. `final design/visuals-and-caching.md` — how images flow from worker to browser.
6. `CONTEXT.md` (project root) — legacy `app/` data pipeline (sensors, FileHelper → DatasetBuilder → VendableDataset).

**Stack at a glance:**
- **Backend:** FastAPI (api) + Postgres-queue worker, both Python 3.14.
- **Frontend:** React 19 + Vite + TypeScript, served by nginx.
- **Persistence:** Postgres 16 (Alembic migrations) + named Docker volumes for `data` (raw + vendable), `artifacts` (rendered PNGs + per-project outputs), `models` (foundation checkpoints + classical manifests).
- **Inference:** torch checkpoints under `app/foundation_models/` (deep learning) + closed-form detectors under `app/detectors/` (classical, SPy / spectral library).

**Run locally:**
```
docker compose -f docker/docker-compose.yml up -d
```
UI at http://127.0.0.1:3000. Admin user seeded from `docker/.env`.

**Wipe and reseed:**
```
./docker/reset-stack.sh             # keeps models volume
./docker/reset-stack.sh --include-models   # nuclear
```

---

## 1 · Inventory the API surface

For every router in `backend/allotrope/api/`:

- List the routes (verb + path) + the Pydantic request/response models.
- Note auth requirements (every route except `/auth/*` requires the JWT cookie via `current_user_claims`).
- Flag any route that mutates filesystem state, and confirm the mutation is symmetric on its corresponding DELETE.
- Cross-reference with the `final design/diagrams/<slug>.drawio` sequence diagram where one exists. Files without a diagram are a roadmap gap to flag.

Specific things to check:
- `/scenes`, `/scenes/onboard`, `/scenes/{id}` (+ `/visualizations`, `/spectrum`, `/bands`, `/bands/{idx}/image`, `/thumbnail`, DELETE)
- `/projects` CRUD + `/projects/{id}/result`, `/projects/{id}/exports`, `/exports/{id}`, `/exports/{id}/download`
- `/projects/{id}/actions`, `/actions/{id}` (+ files, output-files, DELETE), `/action-types`
- `/projects/{id}/annotations`, `/scenes/{id}/annotations/{ann_id}` (+ overlay, DELETE), `/annotation-types`
- `/projects/{id}/visualizations`, `/visualizations/{id}` (+ image, PATCH, DELETE)
- `/projects/{id}/notes`, `/notes/{id}` (+ PATCH, DELETE)
- `/action-templates` CRUD
- `/jobs` (list + filter, single)
- `/metrics/host`, `/metrics/workload`
- `/models` catalog + per-architecture detail
- `/admin/users`, `/auth/login`, `/auth/me`, `/auth/logout`

Output: a single Markdown table with `path | verb | router file | response model | diagram` columns.

---

## 2 · Audit the worker handlers

For each in `backend/allotrope_worker/handlers.py`'s `HANDLERS`:

- **`scene_onboard`** — `backend/allotrope_worker/scene_onboard.py`
- **`annotation_attach`** — `backend/allotrope_worker/annotation_attach.py`
- **`action_run`** — `backend/allotrope_worker/action_run.py` (dispatches via `app/action_types/` registry)
- **`project_export`** — `backend/allotrope_worker/project_export.py`

Confirm each handler:
1. Takes `(session, job)`, returns `(target_kind, target_id)` or raises.
2. Doesn't commit/rollback (runner owns transactions).
3. Issues heartbeats via `ctx.on_step` so the reaper doesn't mark it stale.
4. Cleans up after itself (the post-job `reclaim_memory()` in `cleanup.py` runs in `runner._process_one_tick`, but per-handler temp dirs are the handler's job).

Output: 4 short summaries + a list of bugs / asymmetries spotted.

---

## 3 · Trace the data lifecycle for one scene

Pick a PRISMA scene. Walk through every artifact that gets written to disk and the FK rows that get inserted:

1. Onboarding: raw upload → staging → worker copies → vendable pickle → 6 PNGs + histogram + thumbnail.
2. Annotation attach: multipart upload → worker renders overlay.
3. Action: `band_filter_apply` → `scene_segmentation` → `anomaly_scoring` chain. Show what each one writes under `/artifacts/projects/<pid>/actions/<aid>/output/`.
4. Save visualization: browser canvas screenshot → POST /projects/{id}/visualizations.
5. Export: `project_export` job builds a zip with `manifest.json` + `result.json` + every entity's subdirs.
6. Delete the project: confirm every directory above is `rmtree`d (see `delete_project` in `backend/allotrope/api/projects.py`).

Output: a numbered list of (timestamp, action, path written, DB rows inserted, DB rows deleted). Use the existing `final design/visuals-and-caching.md` as the cache-layer reference.

---

## 4 · Frontend architecture

Walk every page in `frontend/src/pages/`:

- `LoginPage`, `HomePage`, `AdminUsersPage`
- `ScenesPage`, `SceneDetailPage`
- `ProjectsPage`, `ProjectWorkspacePage` (the workspace tabs are the hottest surface)
- `ModelsPage`, `ModelDetailPage`
- `JobsPage` (recently polished — KPI strip + chips + 10-per-page)
- `MonitoringPage` (recently extended with big charts + range picker)

For each, list:
- Which api routes it calls (from `frontend/src/api/`)
- Which polling intervals are active (search for `setInterval`)
- Which components from `frontend/src/components/` it composes

Specific patterns to verify across pages:
- **Click-to-copy ids** (currently only on JobsPage) — could be extended elsewhere.
- **Delete flow** — confirm ScenesPage, SceneDetailPage, ProjectsPage, ProjectWorkspacePage (Action rows), VisualizationsPane, NotesPane all have working delete buttons that surface the 409 cases (scene_in_use, action_running, etc).
- **Live polling** — every list that watches in-flight jobs should auto-poll while anything is `queued`/`running` and stop when idle.

Output: an inventory table + a list of UX inconsistencies (e.g. "ProjectsPage doesn't surface scene_in_use detail like ScenesPage does").

---

## 5 · Cache + memory audit

The `final design/visuals-and-caching.md` doc describes the three caches:

1. Pre-rendered PNGs on `allotrope_artifacts` (immutable HTTP cache).
2. Vendable LRU in the api process (`_VENDABLE_CACHE_SIZE = 2`).
3. Browser HTTP cache.

Verify each is doing what the doc claims:

- Confirm every endpoint that serves a PNG sets `Cache-Control: public, max-age=31536000, immutable`.
- Confirm the vendable LRU only holds 2 entries and that the lookup is correct (no path-traversal issues).
- Confirm `pickle.load` happens outside the lock so concurrent requests for different scenes don't serialize.

Worker-side memory:
- `backend/allotrope_worker/cleanup.py` runs `gc.collect()` + `malloc_trim` after every job.
- Verify it's actually called from `runner._process_one_tick` in the `finally` block, and only when a job was claimed.

Output: list of any deviation from the doc + concrete benchmarks if reproducible (e.g. "RSS after 10 anomaly_scoring jobs on PRISMA: XXX MB vs the 5 GB unbounded case prior to the fix").

---

## 6 · Foundation + classical model dispatch

`backend/allotrope/foundation_models/resolver.py` defines the catalog. As of 2026-05-11:

- 7 foundation architectures (Pratibimba / Antardhana / Tirohita / Asanskrita / Drashta / Chakshu / Indradhanu).
- 3 classical detectors (Drashta-classical wait — no, Drashta is taken; the classical codenames are different: Drashta is the RX-family codename? Verify in `allotrope_models/{rx,mnf_rx,thermal_grx}/current.json`. The intended Indic codenames for classical were Drashta / Vivekha / Tapas, but Drashta is already used by Spatial-Masked-Autoencoder-L1-Unnormalized. **Investigator: confirm there's no codename collision; rename if necessary.**)

In `backend/allotrope/action_types/_anomaly_scoring_run.py`:
- The per-model loop branches on `m.family`: `"foundation"` → torch inferencer, `"classical"` → `app.detectors` factory.
- For classical, `keep_mask` is ANDed into the validity cube before `detector.detect()` (Option B, 2026-05-11).
- Reconstruction TIFF/PNG: foundation writes the actual reconstruction; classical writes the input cube as a placeholder so the viewer's three-panel layout stays uniform.

Confirm:
- The codename collision check above.
- That foundation overrides (`patch_size`, `stride`, `batch_size`, `sam_l1_alpha`) are rejected on classical models in `validate_config`.
- That the worker's `del inferencer, reconstruction` is guarded by `if not is_classical`.

Output: a one-page summary plus any collisions / dead code spotted.

---

## 7 · Build a defect ledger

A single rolling table of every bug, smell, or inconsistency you find across §1-§6, with:

- **Severity** — `critical | high | medium | low | nit`
- **File:line**
- **What's broken**
- **Suggested fix**
- **Effort estimate** — `15min | 1h | half-day | day | week`

Sort by severity desc. This is the deliverable that drives the next session's work plan.

---

## 8 · Open questions for the user

Note anything that's underspecified or where you'd want a decision before changing code. Examples:
- Should `Vivekha` (MNF-RX codename) be Sanskrit, given `Vivekha` is Pali? (`Viveka` is the Sanskrit form.)
- Should `keep_mask`'s foundation-vs-classical asymmetry be documented in the user-facing description, or just in code comments?
- Are classical detectors expected to appear in `ModelsPage`'s flow chart (they have no architecture diagram, since they're closed-form)?

Frame each as a yes/no decision the user can answer in one sentence.

---

## 9 · Logistical notes for the investigator

- **Don't run `reset-stack.sh`** without explicit user approval. It's destructive.
- **Don't rebuild the worker mid-job.** Check `SELECT count(*) FROM jobs WHERE status IN ('queued','running')` before any `docker compose up -d --build worker`. Use the `until ... do sleep 5 done` watcher pattern to drain first.
- **Don't push the api/worker to GHCR or external registries.** Local-only build.
- **Don't add new deps to `backend/requirements.txt` without confirming they're not already in `requirements-worker.txt`.** The api is meant to be lean; heavy ML deps live in the worker only.

---

## 10 · How to deliver

Single Markdown file: `final design/investigation-report.md`. Headings match the sections above. Each section ends with a "Verdict" line: `✅ healthy`, `⚠️ smell`, or `❌ broken`. The defect ledger (§7) is the most important artifact — everything else is context for it.

Use mermaid for any architecture / sequence diagrams. Don't add new images.
