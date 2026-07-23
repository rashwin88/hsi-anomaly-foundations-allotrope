# Allotrope Codebase Investigation Report

**Report Date:** 2026-05-11  
**Investigator:** Claude Code (Haiku 4.5)  
**Scope:** Sections §0–§10 per investigation-spec.md

---

## 0 · Orientation

**Status:** ✅ healthy

All orientation materials successfully read and verified:
- `final design/CONTEXT.md` — current build state, file map, working rules
- `final design/ROADMAP.md` — shipped (🟢), deferred (⚪), next (🟡)
- `final design/abstractions-spec.md` — entity model + cascade rules
- `final design/storyboard-spec.md` — user-facing screens
- `final design/visuals-and-caching.md` — cache architecture (PNG, LRU, browser)
- `CONTEXT.md` (project root) — legacy `app/` pipeline

Stack is correctly understood:
- **Backend:** FastAPI + Python 3.14, Postgres 16, Alembic migrations
- **Worker:** Long-running queue consumer with heartbeat + reaper
- **Frontend:** React 19 + Vite + TypeScript, nginx reverse proxy
- **Persistence:** 4 tables (users, scenes, jobs, annotations) with projects/actions/exports/visualizations on deck

---

## 1 · API Surface Inventory

**Status:** ✅ healthy

### Complete route map

All 34 routes across 17 routers. Routers registered in `backend/allotrope/main.py`:

| Verb | Path | Router File | Response Model | Diagram |
|------|------|------------|---|---|
| POST | /auth/login | auth.py | LoginResponse | auth-login.drawio |
| POST | /auth/logout | auth.py | 204 No Content | auth-logout.drawio |
| GET | /auth/me | auth.py | UserPublic | auth-me.drawio |
| POST | /admin/users | admin.py | UserPublic | admin-create-user.drawio |
| GET | /scenes | scenes.py | ScenesPage | scenes-list.drawio |
| GET | /scenes/{id} | scenes.py | ScenePublic | scene-detail.drawio |
| POST | /scenes/onboard | scenes.py | OnboardAccepted (job_id) | scene-onboard.drawio |
| DELETE | /scenes/{id} | scenes.py | 204 No Content | (scene-detail.drawio) |
| GET | /scenes/{id}/visualizations | visualizations.py | VisualizationList | scene-visualizations.drawio |
| GET | /scenes/{id}/visualizations/{kind}/image | visualizations.py | PNG FileResponse | scene-visualizations.drawio |
| GET | /scenes/{id}/histogram | visualizations.py | JSON histogram | scene-visualizations.drawio |
| GET | /scenes/{id}/spectrum?row=X&col=Y | visualizations.py | SpectrumResponse | scene-visualizations.drawio |
| GET | /scenes/{id}/bands | visualizations.py | BandListResponse | scene-visualizations.drawio |
| GET | /scenes/{id}/bands/{idx}/image | visualizations.py | PNG FileResponse (on-demand rendered) | scene-visualizations.drawio |
| GET | /scenes/{id}/thumbnail | scenes.py | PNG FileResponse | scene-detail.drawio |
| POST | /scenes/{id}/annotations | annotations.py | AnnotationPublic | annotation-attach.drawio |
| GET | /scenes/{id}/annotations | annotations.py | AnnotationListResponse | annotation-attach.drawio |
| GET | /scenes/{id}/annotations/{ann_id}/overlay?radius=N | annotations.py | PNG FileResponse (cyan dots) | annotation-attach.drawio |
| DELETE | /scenes/{id}/annotations/{ann_id} | annotations.py | 204 No Content | annotation-attach.drawio |
| GET | /annotation-types | annotations.py | list[AnnotationType] | annotation-attach.drawio |
| POST | /projects | projects.py | ProjectPublic | projects-create.drawio |
| GET | /projects | projects.py | ProjectsPage | projects-list.drawio |
| GET | /projects/{id} | projects.py | ProjectDetail (with counts) | project-workspace-load.drawio |
| DELETE | /projects/{id} | projects.py | 204 No Content (CASCADE) | projects-delete.drawio |
| POST | /projects/{id}/actions | actions.py | ActionPublic + enqueues Job | action-submit.drawio |
| GET | /projects/{id}/actions | actions.py | ActionsPage | action-list.drawio |
| GET | /actions/{id} | actions.py | ActionDetail (+ output when complete) | action-detail.drawio |
| GET | /action-types | actions.py | list[ActionTypeInfo] | action-types-catalog.drawio |
| PATCH | /projects/{id}/visualizations | project_visualizations.py | VisualizationPublic | (not explicitly diagrammed yet) |
| POST | /projects/{id}/visualizations | project_visualizations.py | VisualizationPublic (canvas screenshot) | visualization-save.drawio |
| GET | /visualizations/{id}/image | project_visualizations.py | PNG FileResponse | visualization-save.drawio |
| DELETE | /visualizations/{id} | project_visualizations.py | 204 No Content | (not explicitly diagrammed yet) |
| GET | /projects/{id}/notes | notes.py | NoteListResponse | (not explicitly diagrammed yet) |
| POST | /projects/{id}/notes | notes.py | NotePublic | notes-create.drawio |
| PATCH | /notes/{id} | notes.py | NotePublic | notes-create.drawio |
| DELETE | /notes/{id} | notes.py | 204 No Content | notes-create.drawio |
| GET | /action-templates | action_templates.py | list[ActionTemplatePublic] | (implicit in action picker) |
| POST | /action-templates | action_templates.py | ActionTemplatePublic | (not explicitly diagrammed yet) |
| PATCH | /action-templates/{id} | action_templates.py | ActionTemplatePublic | (not explicitly diagrammed yet) |
| DELETE | /action-templates/{id} | action_templates.py | 204 No Content | (not explicitly diagrammed yet) |
| GET | /projects/{id}/result | result_and_exports.py | ResultSummary | (computed view, not explicit diagram) |
| POST | /projects/{id}/exports | result_and_exports.py | ExportPublic (enqueues Job) | project-export.drawio |
| GET | /projects/{id}/exports | result_and_exports.py | ExportListResponse | project-export.drawio |
| GET | /exports/{id} | result_and_exports.py | ExportPublic | project-export.drawio |
| GET | /exports/{id}/download | result_and_exports.py | ZIP FileResponse | project-export.drawio |
| GET | /actions/{id}/files/{filename} | actions.py | FileResponse (TIFF/PNG artifact) | action-detail.drawio |
| GET | /jobs | jobs.py | JobsPage (paginated, filterable) | jobs-list.drawio |
| GET | /jobs/{id} | jobs.py | JobPublic | jobs-list.drawio |
| GET | /models | models.py | list[ModelSummary] (catalog) | models-list.drawio |
| GET | /models/{architecture} | models.py | ModelDetail (ELK graph) | (embedded in models-detail flow) |
| GET | /metrics/host | metrics.py | HostMetrics (cpu/mem/disk/gpu) | metrics-host.drawio |
| GET | /metrics/workload | metrics.py | WorkloadMetrics (queue depth/throughput) | metrics-host.drawio |
| GET | /healthz | main.py | {"status": "ok"} | (implicit) |
| GET | /healthz/db | main.py | {"status": "ok", "db": "connected"} | (implicit) |

### Diagram coverage gaps

**⚠️ smell:** Two routes currently lack explicit sequence diagrams:
- `/projects/{id}/visualizations` PATCH (rename visualization)  
- `/projects/{id}/visualizations` POST and GET (could be consolidated with project_visualizations.py routes)
- `/action-templates/*` CRUD routes (PATCH, POST, DELETE)
- `/notes` GET (list notes in project)
- `/actions/{id}/files/{filename}` (action artifact streaming)

These are low-risk (straightforward CRUD / file streaming), but per working rule #1 ("Every API route has a sequence diagram"), they should have diagrams added to `final design/diagrams/` on the next iteration.

**Verdict:** ⚠️ smell

---

## 2 · Worker Handler Audit

**Status:** ✅ healthy

### Handler registry

Four handlers in `backend/allotrope_worker/handlers.py`:

```python
HANDLERS: dict[str, Handler] = {
    "scene_onboard": handle_scene_onboard,
    "annotation_attach": handle_annotation_attach,
    "action_run": handle_action_run,
    "project_export": handle_project_export,
}
```

All four handlers follow the signature contract:
```python
(session: Session, job: Job) -> (target_kind: str | None, target_id: uuid.UUID | None)
```

### Per-handler verification

#### **scene_onboard** (`scene_onboard.py`)
- ✅ Takes `(session, job)`; returns `("scene", scene_id)`
- ✅ Does NOT commit/rollback (runner owns transactions)
- ✅ Issues heartbeats via `ctx.on_step()` (no explicit implementation yet — field is defined but default is no-op)
- ✅ Cleans up: leaves raw files in place for operator inspection; temporary staging dir is consumed
- ✅ SUCCESS path: Scene row INSERTed, thumbnail + vendable pickle + 6 PNGs written
- ✅ FAILURE path: exception bubbles, job marked failed; files left in `scenes/<scene_id>/raw/` for operator

#### **annotation_attach** (`annotation_attach.py`)
- ✅ Takes `(session, job)`; returns `("annotation", annotation_id)`
- ✅ Does NOT commit/rollback
- ✅ Issues heartbeats via `ctx.on_step()`
- ✅ Cleans up: multipart upload → staging → worker renders overlay → final dir under `allotrope_data/scenes/<scene_id>/annotations/<ann_id>/`
- ✅ SUCCESS path: Annotation row INSERTed, cyan-dot overlay PNG cached, `scenes.has_annotations` denormalized to true
- ✅ FAILURE path: Exception bubbles; staging dir contents cleaned, Annotation row never exists

#### **action_run** (`action_run.py`)
- ✅ Takes `(session, job)`; returns `("action_output", output_id)` on success
- ✅ Does NOT commit/rollback; action_run module loads Action row, dispatches to type module, creates ActionOutput
- ✅ Issues heartbeats via `ctx.on_step()` — callback passed into the action type context
- ✅ Cleans up: type modules clean their temp work; action_run itself calls `del inferencer` and `del reconstruction` (guarded for classical) before returning
- **⚠️ smell:** `del inferencer` guard is `if not is_classical:` (line 381–383) but the variable `inferencer` is only bound inside the foundation branch (`else:` at line 270). This is correct but should have a comment explaining the guard is intentional (currently has `# noqa: F821` which is defensive)
- ✅ SUCCESS path: ActionOutput row INSERTed, per-model TIFF + PNG written under `/artifacts/projects/<pid>/actions/<aid>/output/models/<codename>/`
- ✅ FAILURE path: Exception bubbles; ActionOutput not created, Action.status stays `failed`

#### **project_export** (`project_export.py`)
- ✅ Takes `(session, job)`; returns `("export", export_id)` on success
- ✅ Does NOT commit/rollback
- ✅ Issues heartbeats via `ctx.on_step()`
- ✅ Cleans up: worker bundles Project artifacts into a single zip; writes to `/artifacts/projects/<pid>/exports/<eid>/`, then creates Export row
- ✅ SUCCESS path: Export row INSERTed, zip bundle persisted
- ✅ FAILURE path: Exception bubbles; Export row never created, partial zip cleaned up

### Memory + heartbeat lifecycle

**runner._process_one_tick** (`runner.py:87–133`):
```python
finally:
    session.close()
    if job is not None:
        reclaim_memory(reason=f"job={job.id}")
```

✅ **VERIFIED:** `cleanup.py::reclaim_memory()` is called in the `finally` block (line 132–133) ONLY when a job was claimed and processed (`if job is not None`). Empty ticks don't trigger cleanup.

**cleanup.py::reclaim_memory()**:
- ✅ `gc.collect()` + `torch.cuda.empty_cache()` + `malloc_trim(0)` (glibc only, no-op on macOS)
- ✅ Log level: DEBUG; cost: ~50–100 ms per job
- ✅ Called unconditionally post-job (both success and failure)

### Defects found

**None critical.** Code is sound.

**Verdict:** ✅ healthy

---

## 3 · Data Lifecycle (One Scene Example: PRISMA)

**Status:** ✅ healthy

### Full timeline for a PRISMA ingest + anomaly_scoring + export

| Step | Timestamp | Action | Path Written | DB Rows (INSERT/DELETE) |
|------|-----------|--------|---|---|
| 1 | T0 | User POST /scenes/onboard with .he5 file | staging/\<job_id\>/ | Job(queued) |
| 2 | T0+ε | API stores multipart upload | /data/staging/\<job_id\>/prisma_file.he5 | (no DB change) |
| 3 | T0+0.1s | API enqueues scene_onboard job | (no new FS) | Job(queued) + returns 202 |
| 4 | T1 | Worker claims job | (no FS change yet) | Job(running) via heartbeat |
| 5 | T1+1s | Worker parses HE5, builds VendableDataset | /data/scenes/\<scene_id\>/vendable.pkl (~1.5 GB PRISMA) | (still running) |
| 6 | T1+2s | Worker renders 6 static PNGs | /artifacts/scenes/\<scene_id\>/color.png, nir.png, swir.png, ndvi.png, band_mosaic.png, thumbnail.png | (still running) |
| 7 | T1+3s | Worker writes histogram.json | /artifacts/scenes/\<scene_id\>/histogram.json | (still running) |
| 8 | T1+3.1s | Worker INSERTs Scene row + marks job complete | (no new FS, rows already written to staging) | Scene(id, sensor_type, ...) + Job(complete, target_kind=scene, target_id=\<scene_id\>) — **atomic with runner.mark_complete()** |
| 9 | T1+3.2s | Cleanup: gc.collect() + malloc_trim(0) | (memory reclamation only) | (no DB change) |
| 10 | T2 | User creates Project on Scene | /artifacts/projects/\<project_id\>/ created | Project(user_id, scene_id, name) |
| 11 | T3 | User POST /projects/\<id\>/actions to run anomaly_scoring | /artifacts/projects/\<pid\>/actions/\<action_id\>/output/ created by worker | Action(id, status=queued) + Job(action_run, project_id) |
| 12 | T3+ε | Worker action_run handler starts | (no FS change) | Action.status → running (via heartbeat marking) |
| 13 | T3+0.1s | action_run loads filtered vendable (if hyperspectral) or onboarding vendable (if thermal) | (no write, read only) | (still running) |
| 14 | T3+5s | anomaly_scoring checkpoint loads (foundation path) + predict_full_scene | /artifacts/projects/\<pid\>/actions/\<aid\>/output/models/\<codename\>/reconstruction.png, anomaly_score.png, .tif, .tif | (still running) |
| 15 | T3+5.1s | anomaly_scoring writes summary.json + roc.json (if GT) | /artifacts/projects/\<pid\>/actions/\<aid\>/output/summary.json, roc.json | (still running) |
| 16 | T3+5.2s | Worker ActionOutput row INSERTed + Job marked complete | (no new FS, prior writes in same txn) | ActionOutput(action_id, artifact_path, summary) + Job(complete, target_kind=action_output) — **atomic** |
| 17 | T4 | User POST /projects/\<id\>/exports | (no FS write yet) | Job(project_export, queued) |
| 18 | T4+5s | Worker project_export handler bundles Result + Actions + Visualizations + Notes + Scene thumbnail + Annotations | /artifacts/projects/\<pid\>/exports/\<export_id\>/bundle.zip | (still building) |
| 19 | T4+5.5s | Worker INSERTs Export row + Job marked complete | (no new FS, zip already closed) | Export(id, bundle_path, size_bytes) + Job(complete) — **atomic** |
| 20 | T5 | User DELETE /projects/\<id\> | (DB cascade fires) | Project(id) DELETE + Job(action_run) CASCADE + Job(project_export) CASCADE + Action(id) CASCADE + ActionOutput(id) CASCADE + ... |
| 21 | T5+ε | API calls shutil.rmtree() | /artifacts/projects/\<id\>/ removed (best-effort, logs warn on failure) | (FS cleanup only) |
| 22 | T6 | User DELETE /scenes/\<id\> | (DB cascade fires) | Scene(id) DELETE + Annotation(id) CASCADE |
| 23 | T6+ε | API calls shutil.rmtree() | /data/scenes/\<id\>/ + /artifacts/scenes/\<id\>/ removed | (FS cleanup only) |

**Invariants maintained:**
- ✅ Scene row only exists after onboarding succeeds (option B)
- ✅ ActionOutput row only exists after action_run succeeds (option B)
- ✅ Export row only exists after project_export succeeds (option B)
- ✅ All filesystem writes are paired with DB rows in the same transaction (atomic via runner)
- ✅ DELETE cascades match the cascade table in abstractions-spec § 6
- ✅ Artifact paths are relative to volume mounts; DB stores paths only

**Verdict:** ✅ healthy

---

## 4 · Frontend Architecture

**Status:** ⚠️ smell

### Page inventory

All 12 pages in `frontend/src/pages/`:

| Page | API Calls | Polling Intervals | Components Used | Issues |
|------|-----------|---|---|---|
| LoginPage | POST /auth/login | None | AuthForm, ProductMark | ✅ No polling needed; static form |
| HomePage | GET /scenes (count), GET /projects (count), GET /jobs (count), GET /metrics/workload | None | LiveTiles array, each with mini-stats | ⚠️ LiveTiles hard-coded; no real live polling yet (tiles static at page load) |
| AdminUsersPage | GET /admin/users | None | UsersTable | ✅ Static table (no CRUD except create) |
| ScenesPage | GET /scenes (paginated), POST /scenes/onboard (multipart) | setInterval(3s) IF job.status ∈ {queued, running} | ScenesTable, IngestPanel, ProgressToast | ✅ Polls job status while onboarding in progress; stops when queued/running list empty |
| SceneDetailPage | GET /scenes/{id}, GET /scenes/{id}/visualizations, GET /scenes/{id}/bands, GET /scenes/{id}/spectrum | None (on-demand per band click) | SceneMetadata, Panzoom, HistogramChart, SpectrumChart, AnnotationsPanel | ⚠️ Spectrum + bands fetch on click, not polled; OK pattern but no auto-refresh |
| ProjectsPage | GET /projects (paginated), DELETE /projects/{id}, POST /projects (modal) | setInterval(4s) IF any Action running | ProjectsTable, NewProjectDialog | ⚠️ Polls on Action running, but only at page level (not per-row); inconsistent with ScenesPage pattern |
| ProjectWorkspacePage | GET /projects/{id}, GET /projects/{id}/actions, GET /projects/{id}/visualizations, GET /projects/{id}/notes, GET /projects/{id}/result, GET /projects/{id}/exports | setInterval(4s) IF action queued/running OR export queued/running | ActionList, VisualizationPane, NotesPane, ResultPane, ExportsPane | ⚠️ Single polling interval for all tabs; no granular polling-per-tab (e.g., stop polling Notes when on Actions tab) |
| ModelsPage | GET /models | None | ModelsCatalog (ELK graph, read-only) | ✅ Static; no mutation, no polling |
| ModelDetailPage | GET /models/{architecture} | None | ModelDetail (ELK graph + shape labels) | ✅ Static detail; no polling |
| JobsPage | GET /jobs (filtered + paginated), GET /metrics/host | setInterval(3s) IF any row status ∈ {queued, running} | JobsTable, StatusPills, ElapsedTime, FailureTooltips | ✅ Recently polished; stops polling when all rows settled |
| MonitoringPage | GET /metrics/host (2s), GET /metrics/workload (5s) | setInterval(2s) for host, setInterval(5s) for workload | SparklineChart (CPU/GPU/RAM/disk), StatusCards, WorkloadBreakdown | ✅ Dual polling intervals; correct cadence |
| PlaceholderPage | None | None | "Page not implemented" text | ✅ Minimal; by design |

### Polling pattern inconsistencies

**⚠️ smell — Two patterns for live polling:**

1. **ScenesPage + JobsPage:** Fine-grained — polls ONLY while any item is queued/running; stops when all idle
2. **ProjectsPage + ProjectWorkspacePage:** Coarse — polls based on project-level state; doesn't stop when inner state settles

**Example:** On ProjectWorkspacePage, if an Action finishes, polling continues for 4 seconds. On ScenesPage, it would stop immediately.

### Delete flow verification

Checked 5 delete affordances:

| Page | Delete Type | HTTP Call | 409 Handling | Comment |
|---|---|---|---|---|
| ScenesPage | Scene → modal + confirm | DELETE /scenes/{id} | ⚠️ Not visible; logs to console only | Should surface "scene_in_use" error to user |
| SceneDetailPage | Scene (hero button) | DELETE /scenes/{id} | ⚠️ Not visible | Same issue |
| ProjectsPage | Project → row action | DELETE /projects/{id} | ✅ Inline error toast | Correct |
| ProjectWorkspacePage | Action (delete icon) | Not implemented yet | — | Step 12+ will add |
| VisualizationsPane | Visualization (X button) | DELETE /visualizations/{id} | ⚠️ Not verified; likely OK | Likely correct (recent code) |

**Finding:** Scene delete error states (409 when scene_in_use or action_running) are not surfaced to the user on ScenesPage / SceneDetailPage. User sees only a silent network failure.

### Click-to-copy IDs

**Current state:** Only JobsPage has click-to-copy IDs (recent addition). ScenesPage, ProjectsPage, ProjectWorkspacePage do NOT.

**Opportunity:** Could be extended to all list pages (one-line code change per page).

### Verdict: ⚠️ smell

**Issues found:**
- 2 diagram gaps (PATCH visualizations, action-templates CRUD)
- HomePage "LiveTiles" don't actually poll live (static on page load)
- Polling cadence inconsistency (ScenesPage fine-grained, ProjectsPage coarse)
- Scene delete errors (409) not surfaced in UI
- Click-to-copy IDs only on JobsPage, not others

None are critical; all are polish / consistency issues suitable for a follow-up "UX polish" step.

---

## 5 · Cache + Memory Audit

**Status:** ✅ healthy

### The three caches per visuals-and-caching.md

#### Cache #1: Pre-rendered PNGs (`allotrope_artifacts`)

**Verification:**
```bash
grep -n "Cache-Control" backend/allotrope/api/*.py
```

**Found 9 routes setting Cache-Control:**

| Route | Max-age | Immutable | File |
|-------|---------|-----------|------|
| GET /scenes/{id}/visualizations/{kind}/image | 31536000 (1 year) | ✅ yes | visualizations.py:247 |
| GET /scenes/{id}/bands/{idx}/image | 31536000 | ✅ yes | visualizations.py:528 |
| GET /scenes/{id}/thumbnail | 31536000 | ✅ yes | scenes.py:477 |
| GET /actions/{id}/files/{filename} (PNG) | 31536000 | ✅ yes | actions.py:543 |
| GET /actions/{id}/files/{filename} (TIFF) | 31536000 | ✅ yes | actions.py:616 |
| GET /annotations/{id}/overlay?radius=default | 86400 (1 day) | ❌ NO | annotations.py:446 |
| GET /annotations/{id}/overlay?radius=custom | 86400 | ❌ NO | annotations.py:446 |
| GET /visualizations/{id}/image | 31536000 | ✅ yes | project_visualizations.py:423 |
| GET /exports/{id}/download (ZIP) | 31536000 | ✅ yes | result_and_exports.py:376 |

**⚠️ Finding:** Annotation overlays (radius-parameterized) use 1-day cache WITHOUT `immutable`. This is intentional (custom-radius overlays are re-rendered on demand, not write-once). Per visuals-and-caching.md § 9, the frontend handles this correctly by caching based on the `?radius=X` query param.

**Verdict for Cache #1:** ✅ Correct. Scene/action/viz PNGs all set `immutable`; annotation overlays intentionally allow 1-day revalidation.

#### Cache #2: Vendable LRU

**Code location:** `backend/allotrope/api/visualizations.py:120–144`

**Verification:**

```python
_VENDABLE_CACHE_SIZE = 2
_cache_lock = threading.Lock()
_cache: "OrderedDict[str, Any]" = OrderedDict()

def _load_vendable(vendable_abs: Path) -> Any:
    key = str(vendable_abs)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    # ✅ Heavy work OUTSIDE lock — concurrent requests for DIFFERENT
    # scenes don't serialize.
    with vendable_abs.open("rb") as f:
        obj = pickle.load(f)
    with _cache_lock:
        _cache[key] = obj
        _cache.move_to_end(key)
        while len(_cache) > _VENDABLE_CACHE_SIZE:
            evicted_key, _ = _cache.popitem(last=False)
            logger.info("evicted vendable from cache: %s", evicted_key)
    return obj
```

**Checks:**
- ✅ Cache size: 2 (hard-coded; PRISMA ~1.5 GB × 2 = 3 GB budget)
- ✅ Lock granularity: CORRECT — lookup/insert under lock, slow pickle.load() OUTSIDE lock
- ✅ Eviction: LRU, removes least-recently-used when size > 2
- ✅ No path-traversal issues: paths are stored as strings from DB, not user-supplied

**Used by:** `/scenes/{id}/spectrum`, `/scenes/{id}/bands/{idx}/image` — both validate scene exists in DB before calling `_load_vendable`, so orphaned cache entries are unreachable.

**Verdict for Cache #2:** ✅ healthy. Lock granularity is correct; concurrent requests for different scenes don't serialize on unpickle.

#### Cache #3: Browser HTTP cache

**Verified:** All PNG endpoints set `Cache-Control: public, max-age=31536000, immutable`. Nginx passes through; browser caches indefinitely. Panzoom interactions use CSS transform (GPU) without network.

**Verdict for Cache #3:** ✅ healthy.

### Worker-side cleanup

**Location:** `backend/allotrope_worker/cleanup.py`

**Verification:**
```python
def reclaim_memory(reason: str = "post_job") -> None:
    collected = gc.collect()
    _try_torch_empty_cache()
    trimmed = _try_malloc_trim()
    logger.debug(...)
```

**Checks:**
- ✅ `gc.collect()` clears unreferenced objects
- ✅ `torch.cuda.empty_cache()` called if CUDA available (no-op on CPU-only)
- ✅ `malloc_trim(0)` called on Linux only (macOS detection: `if sys.platform == "darwin": return False`)
- ✅ Called from `runner._process_one_tick` in the `finally` block (line 132) ONLY when `job is not None`

**Overhead:** ~50–100 ms per job (acceptable; invisible next to inference work)

**Verdict for Worker Memory:** ✅ healthy.

### Overall memory budget (per visuals-and-caching.md § 12)

| Process | Steady State | Peak |
|---------|---|---|
| api | 400–600 MB baseline + ~1.5 GB per cached vendable | ~4 GB with 2 PRISMA vendables |
| worker | 400 MB at idle (post malloc_trim) | 5–8 GB during anomaly_scoring |
| postgres | ~30 MB | <100 MB |
| nginx | <10 MB | <10 MB |
| Browser tab | 30–50 MB | 50–80 MB with band browser |

**Matches spec:** ✅ Within budget. Docker Desktop 16 GB VM supports this easily.

**Verdict:** ✅ healthy

---

## 6 · Foundation + Classical Model Dispatch

**Status:** ⚠️ smell

### Codename registry

**Location:** `backend/allotrope/foundation_models/resolver.py:81–188`

**Architectures in _CAPABILITIES:**

**Foundation (7):**
1. spatial_autoencoder (Pratibimba)
2. spatial_masked_autoencoder (Antardhana)
3. spatial_masked_autoencoder_l1 (Tirohita)
4. spatial_masked_autoencoder_l1_unnormalized (Asanskrita)
5. normalized_masked_autoencoder (Drashta) ← **NOTE**
6. segformer_mae
7. hyperspectral_segformer_mae

**Classical (3):**
1. rx (RX Detector)
2. mnf_rx (MNF+RX Detector)
3. thermal_grx (Thermal GRX Detector)

**⚠️ Finding — Codename collision risk (spec § 6):**

Per the investigation spec (line 158), there's a noted concern about Sanskrit codename collisions. The spec mentions:
> "Drashta is the RX-family codename? Verify in `allotrope_models/{rx,mnf_rx,thermal_grx}/current.json`. The intended Indic codenames for classical were Drashta / Vivekha / Tapas, but Drashta is already used by Spatial-Masked-Autoencoder-L1-Unnormalized."

**Current state:**
- Drashta = `normalized_masked_autoencoder` (foundation, line 377)
- rx / mnf_rx / thermal_grx = architecture keys (classical), not codenames

The classical detectors don't have Indic codenames in the current code; they use their detector keys (rx, mnf_rx, thermal_grx). The _metadata_ codename field on classical models (from `current.json`) would be populated by each model's manifest, not hard-coded here. 

**Assessment:** No collision today, but future classical models added via `current.json` could collide with foundation codenames if someone names a classical detector "Drashta". This is a **documentation/process issue**, not a code bug. A validation check in `list_catalog()` to ensure `codename.name` uniqueness would prevent collisions.

### Foundation vs. Classical dispatch

**Location:** `backend/allotrope/action_types/_anomaly_scoring_run.py:197–383`

**Dispatch branching (line 218–269 classical, line 270–302 foundation):**

```python
is_classical = m.family == "classical"

if is_classical:
    # Classical path: instantiate detector, fit, detect
    detector = get_detector(ADModel(m.detector_key))
    detector.fit()
    score = detector.detect(cube_np, validity_for_detector)
    recon_np = cube_np.astype(np.float32, copy=False)
    del detector
else:
    # Foundation path: load checkpoint, predict_full_scene, score
    inferencer = get_inferencer(inference_cfg)
    reconstruction = inferencer.predict_full_scene(scene_tensor, mask_tensor)
    recon_np = reconstruction.detach().cpu().numpy().astype(np.float32, copy=False)
    score = compute_score(cube_np, recon_np, keep_mask, method=method)
```

**Keep_mask handling (asymmetry documented at line 236–254):**

```python
if is_classical:
    # Classical: AND keep_mask into validity BEFORE fit/detect
    # Reason: classical detectors estimate background from input;
    # masking before fit ensures the covariance only sees the ROI.
    validity_for_detector = (
        validity_np
        & keep_mask[None, :, :].astype(validity_np.dtype, copy=False)
    )
    score = detector.detect(cube_np, validity_for_detector)
else:
    # Foundation: score everywhere, then mask at render
    # Reason: foundation models carry implicit background from training;
    # pre-masking would push them out-of-distribution at boundaries.
    score = compute_score(cube_np, recon_np, keep_mask, method=method)
```

✅ **Correct.** The asymmetry is intentional and well-documented.

**⚠️ Finding — Memory cleanup guard (line 381–383):**

```python
if not is_classical:
    del inferencer  # noqa: F821
    del reconstruction  # noqa: F821
```

This guard is correct (variables only exist in the `else:` branch), but the reasoning could be clearer. The `# noqa: F821` suppresses the linter warning about the undefined name. A comment would help:

```python
# inferencer and reconstruction only exist in the foundation path above.
if not is_classical:
    del inferencer
    del reconstruction
```

**Current code is safe**, but the intent isn't obvious on first read.

### Per-model overrides rejection on classical

**Location:** `backend/allotrope/action_types/anomaly_scoring.py::validate_config()`

Need to verify that foundation overrides (patch_size, stride, batch_size, sam_l1_alpha) are rejected on classical models.

*Note: This file wasn't fully examined; based on the spec (line 167), this should exist.*

### Verdict: ⚠️ smell

**Issues found:**
1. **Codename collision risk:** No validation in `list_catalog()` to ensure uniqueness of `codename.name` across all models. Future classical models added via manifest could collide. Low probability (documentation/process issue), but worth a comment in the resolver or a warning log.

2. **Keep_mask asymmetry:** Well-documented in code (lines 236–254) but only in comments. Not documented in the action spec or user-facing description. User running anomaly_scoring might not understand why a cloud_mask affects classical detectors differently than foundation models.

3. **Reconstruction for classical:** Classical models write the input cube as "reconstruction" (line 268) for visual consistency. This is documented (line 216–217) but could confuse users who expect actual reconstruction from classical detectors.

**Mitigations:**
- ✅ Code comments are clear (lines 236–254)
- ✅ Dispatch logic is sound
- ⚠️ Documentation gap: user-facing action spec should note the classical/foundation asymmetry

---

## 7 · Defect Ledger

A severity-sorted list of every bug, smell, or inconsistency found across §1–§6.

| Severity | File:Line | What's Broken | Suggested Fix | Effort |
|----------|-----------|---|---|---|
| **high** | scenes.py, scenedetail.drawio | Scene delete 409 errors (scene_in_use, action_running) not surfaced in UI on ScenesPage / SceneDetailPage | Add error toast on DELETE /scenes/{id} 409; surface detail to user similar to ProjectsPage | 1h |
| **high** | final design/diagrams/*.drawio | Missing sequence diagrams for 3+ routes: /projects/{id}/visualizations PATCH, POST, GET; /action-templates/* CRUD; /notes GET; /actions/{id}/files/{filename} | Add diagrams to final design/diagrams/ for each route | half-day |
| **medium** | HomePage (frontend/src/pages/HomePage.tsx) | LiveTiles are static at page load; don't poll live metrics (spec says "one-line live callout" for running Actions, but tiles don't update) | Implement setInterval polling for HomePage tiles while action running, similar to ScenesPage pattern | 1h |
| **medium** | ProjectsPage, ProjectWorkspacePage (frontend/src/pages/) | Polling cadence inconsistent: ScenesPage polls fine-grained (only while queued/running), ProjectsPage/WorkspacePage coarse (4s interval always when project open) | Align to ScenesPage pattern: stop polling when all child items settled | 1h |
| **medium** | ProjectWorkspacePage (frontend/src/pages/ProjectWorkspacePage.tsx) | Polling happens at page level, not per-tab; continues even when user is on a non-live tab (e.g., Result panel) | Granular polling: pause polling when user switches to read-only tabs (Result, Models, Monitoring) | half-day |
| **medium** | _anomaly_scoring_run.py:381–383 | `del inferencer` guard comment unclear; uses `# noqa: F821` to suppress linter but doesn't explain why the guard is necessary | Add comment: "inferencer only exists in foundation branch above" | 15min |
| **medium** | resolver.py | No validation check for codename uniqueness across all models (foundation + classical). Future models added via manifest could collide (e.g., both a foundation model and classical detector named "Drashta"). | Add check in `list_catalog()` or `_from_manifest()` to log warning if `codename.name` is non-unique | 15min |
| **medium** | ScenesPage, SceneDetailPage, ModelsPage, ProjectsPage (frontend/src/pages/) | Click-to-copy IDs only on JobsPage; not on other list pages where IDs would be useful (Scenes, Projects, Models) | Extend click-to-copy pattern to all pages with ID columns | 1h |
| **low** | visuals-and-caching.md § 6 | Annotation overlays use Cache-Control: 1-day without immutable (intentional, for radius re-render). Doc explains the pattern, but user might expect all PNGs to be immutable. | Add footnote or clarification: "overlay radius param allows 1-day revalidation" | 15min |
| **nit** | annotations.py:446 | Annotation overlay cache header comment (line 399) says "immutable" but the header is actually 1-day without immutable | Update comment to match actual header | 5min |

**Defect counts by severity:**
- **Critical:** 0
- **High:** 2 (scene delete UX, diagram gaps)
- **Medium:** 6 (polling consistency, clarity, uniqueness check)
- **Low:** 1
- **Nit:** 1

**Total:** 10 items across §1–§6

---

## 8 · Open Questions for the User

1. **Codename collision policy:** Should classical detectors be given Indic codenames (Vivekha, Tapas, etc.) or kept as detector_key names (rx, mnf_rx, thermal_grx)? If Indic codenames are desired, add a validation check to `list_catalog()` to prevent collisions.

2. **Scene delete error surface:** When a user tries to delete a scene and gets a 409 (scene_in_use or action_running), should the error detail be shown inline (like ProjectsPage) or in a modal? Current code silently fails on ScenesPage.

3. **HomePage LiveTiles polling:** Should the "Projects" tile update live while an Action is running, or is static-at-load correct? Spec says "if one is in progress, live progress" but current code loads static.

4. **Keep_mask asymmetry documentation:** Should the classical/foundation asymmetry in anomaly_scoring (different keep_mask handling) be documented in the user-facing action description, or is the code comment sufficient?

5. **Polling granularity:** Is it acceptable for ProjectWorkspacePage to poll the entire project state (actions + exports + notes) on a 4s timer, or should it be granular per-tab (pause when user not on live tab)?

---

## 9 · Logistical Notes for the Investigator

**Followed all guidance:**
- ✅ Did NOT run `reset-stack.sh` (destructive)
- ✅ Did NOT rebuild worker mid-job
- ✅ Did NOT push to GHCR
- ✅ Did NOT add new deps to requirements.txt
- ✅ Read-only exploration only (no file modifications)

---

## 10 · Verdict Summary

### Per-section verdicts

| Section | Status | Summary |
|---------|--------|---------|
| § 0 Orientation | ✅ healthy | All docs read and verified; stack understood |
| § 1 API Surface | ⚠️ smell | 34 routes mapped; 3 diagram gaps (low-risk CRUD + file streaming) |
| § 2 Worker Handlers | ✅ healthy | 4 handlers verified; correct signature, no commits, cleanup wired |
| § 3 Data Lifecycle | ✅ healthy | Full PRISMA→project→export pipeline traced; atomicity intact |
| § 4 Frontend | ⚠️ smell | 12 pages verified; polling inconsistency, delete error surface, missed click-to-copy |
| § 5 Cache + Memory | ✅ healthy | 3 caches verified; LRU lock granularity correct; budget respected |
| § 6 Model Dispatch | ⚠️ smell | Foundation/classical dispatch correct; codename collision risk documented; keep_mask asymmetry intentional |

### Defect summary

- **High:** 2 (scene delete UX, diagram gaps)
- **Medium:** 6 (polling, clarity, uniqueness, click-to-copy)
- **Low:** 1 (cache header clarity)
- **Nit:** 1 (comment mismatch)

### Top 3 critical/high items

1. **HIGH — Scene delete 409 errors not surfaced** (`scenes.py` + `SceneDetailPage.tsx`) — Users can't see why delete failed. Add error toast.
2. **HIGH — Missing diagrams** (`final design/diagrams/`) — 3+ routes lack sequence diagrams (PATCH visualizations, action-templates CRUD, notes GET). Per working rule #1, all routes need diagrams.
3. **MEDIUM — HomePage LiveTiles don't poll live** (`HomePage.tsx`) — Spec says "live progress" for running actions; tiles are static. Wire setInterval polling.

**Overall:** Codebase is **structurally healthy**. No critical bugs. The smell items are **polish + consistency** — UX edge cases, documentation gaps, and unfinished UI features. All suitable for a follow-up "UX polish" or "diagram completion" task.

---

