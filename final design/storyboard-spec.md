---
title: Storyboard Spec — Allotrope (build-ready distillation)
status: locked at structural altitude (UX details to be coloured in during implementation)
distilled_from: storyboard.md (sessions 1–15, 2026-05-08)
companion: final design.md
---

# Storyboard Spec

This is the build-ready distillation of the storyboard sessions. It contains only **locked structural decisions**. No questions, no chronology, no strawmen. Read this end-to-end and you have everything needed to start scaffolding the frontend and shaping the API surface.

For chronological context, decision history, and rationale, see [storyboard.md](storyboard.md).
For containerization, queue, and bundle decisions, see [final design.md](final design.md).

---

## 1. Product framing

| | |
|---|---|
| **Audience** | Operational + commercial (mixed). Needs both substance and polish. |
| **Demo length** | 30 minutes. Pacing matters. |
| **North-star line** | *"I'd fund more compute for you to build sophisticated models."* Every design decision serves this. |
| **No single wow moment** | The product wins by visual density, real data on every panel, and attention to detail. |
| **Offline demo constraint** | Whole stack ships as a tarball; runs from `docker compose up` with no internet. |

## 2. Aesthetic anchor

Bloomberg terminal · flight-ops console · scientific instrument. **Not** SaaS landing page.

- **Yes:** dense, instrument-like panels; monospaced numbers where they fit; real data on every surface; small live sparklines and counters; restrained domain visuals (e.g. faint hyperspectral spectrum strips).
- **No:** gradient heroes, oversized cards, dashboard templates, emoji-decorated tiles, lorem-ipsum sidebars, "modern startup" polish.
- **Test for any panel:** would this fit on the boot screen of a ground station? If it would also fit on a SaaS landing page, redesign.

## 3. Authentication

- **Real auth.** Postgres `users` table. Passwords hashed (argon2 or bcrypt). Real session token.
- Even if only one account is seeded for the demo, the implementation is real.
- **Login screen aesthetic: branded utility.** Minimal form (username, password, button) + product mark + version number + one restrained domain element. Nothing else.
- Login sets the visual tone for the rest of the app — every later screen must feel consistent with it.

## 4. App shell

### Sidebar (6 destinations, this order)

```
1. Home
2. Scenes        (Library; hosts the Ingest button)
3. Projects      (workspaces; encompasses Actions, Outputs, Visualizations, Notes, Result)
4. Models        (read-only catalog)
5. Jobs          (queued + running + recent jobs from the queue)
6. Monitoring    (system + workload metrics)
```

Settings is on the **top bar**, not the sidebar.

### Top bar (persistent on every screen)

- **Active Project** name (or empty/dimmed when none).
- **GPU status** indicator.
- **CPU / GPU / RAM sparklines** — small, live, host-level metrics.
- **User icon** + **Settings**.

The top bar is the visual signature of the product. It runs on every screen and gives the audience continuous proof the system is real and alive.

## 5. Destinations

### 5.1 Home (landing)

- Reached on successful login and via the Home sidebar item.
- Contains a **clear, informational** description of the platform's capabilities (not promotional — the audience is already inside the app).
- Below the description: **a row of five live tiles**, each clickable, navigating to the corresponding destination.

#### Live tiles (5 total)

| Tile | Navigates to | Content |
|---|---|---|
| **Library** | Scenes | Total scene count + per-sensor breakdown + recent thumbnails |
| **Ingest activity** | Scenes (with intent to ingest) | Most recent onboarding event; if one is in progress, live progress; small "today / this week" counter |
| **Projects** | Projects | Active project count; most recent project; if any Action is currently running, a one-line live callout (`▶ <type> · <scene> · <elapsed>`) |
| **Model Catalog** | Models | Model count + top names by recency-of-use |
| **Workload** | Jobs | Workload metrics (inference throughput, queue depth, average Action duration) — distinct from top-bar host metrics |

Tiles are instrument-console panels, not feature cards. Every tile shows real data tied to real work.

### 5.2 Scenes (the Library)

Two screens: **Scenes landing** and **Scene Detail**.

#### Scenes landing
- Structurally a **table**. Not a bookshelf, not a card grid, not a map. The book metaphor is conceptual only.
- Each row = one onboarded Scene.
- Columns: name, sensor/product, metadata (specific column selection deferred — driven by the Scene metadata object produced at onboarding), small thumbnail.
- **Ingest button** lives on this page (not as a separate sidebar item). Clicking it opens the onboarding flow (flow shape deferred).
- **Deep advanced filtering** is a first-class feature. Filter surface should be powerful (multi-facet across sensor type, date range, geographic bounds, cloud cover, valid-pixel %, band count, "has annotations," etc.). UI shape deferred.

#### Scene Detail (open on row click)
Must contain:
1. **Visuals of the Scene** — Template-rendered, with optional Annotation overlay.
2. **List of Projects** associated with this Scene.
3. **Create Project** button — opens the canonical New Project dialog (see § 5.3).

Layout deferred.

### 5.3 Projects

Two screens: **Projects landing** and **Project workspace**.

#### Projects landing
- Structurally a **table**.
- Columns at minimum: project name, associated Scene.
- **New Project** button — opens a dialog that lets the user choose a Scene; on confirm, the user lands in the new Project's workspace. *(Same dialog is reused from the Scene Detail page's Create Project button.)*

#### Project workspace
The heart of the product. Must surface, at minimum, these abstractions (specific layout regions deferred):

- **Actions list** (sequential, ordered by creation/completion) + **+ New Action** affordance.
- **Action detail pane** — when an Action is selected, its configuration and Output viewer appear **in place** (no new window, no navigation).
- **Visualizations collection** — curated visuals (see § 7).
- **Notes** — project-owned, supports inline references to Actions and the Scene.
- **Result panel** — persistent, auto-live (see § 8).
- **Export button** — bundles the current Result state into a downloadable package.

#### Project lifecycle
- Just exist + deletable. No active/archived/finalized states.

#### Not in scope
- **No comparison surface.** Switching between Actions in the list is sufficient.
- **No DAG / pipeline-composer UI.** The Action list is sequential / notebook-style. Actions can reference prior Action Outputs as inputs, but the user creates and runs Actions one at a time.

### 5.4 Models (read-only catalog) — light spec

- Read-only **Model Catalog**.
- Each entry: name, version, architecture diagram, plain-language description, training data summary, performance metrics, intended uses & limitations (the standard ML "model card" pattern).
- **No selection / swap from this page.** Selecting a model for an Action happens inside the Action's configuration dialog.
- Detailed layout deferred.

### 5.5 Jobs — light spec

- A list of **queue jobs** ordered by recency / priority. Filterable by status (`queued`, `running`, `complete`, `failed`, `cancelled`) and type (`scene_onboard`, `action_run`, …).
- Per-row: id, type, status, started_at, elapsed, owning entity (scene_id for onboarding, action_id + project_id for an action run), failure_reason if failed.
- Click a row → job detail (configuration payload, log/error trace, link to the resulting entity if `complete`).
- Default v1 view: **queued + running**. History accessed via filter.
- Complements Monitoring (aggregate metrics) by listing **the actual work items**.
- Detailed layout deferred.

### 5.6 Monitoring — light spec

- Standard system + workload metrics.
- Distinct from the top bar:
  - Top bar = **host** metrics (CPU/GPU/RAM "is the machine alive").
  - Monitoring page = **workload** metrics (Action throughput, queue depth, average Action duration, per-process / per-model breakdowns) and deeper host metrics if useful.
- Specifics deferred to implementation.

## 6. Vocabulary (canonical entities)

These terms have one definition each. Use them consistently.

| Term | Definition |
|---|---|
| **Library** | The global, persistent collection of all onboarded Scenes. The "Scenes" sidebar is the view into it. |
| **Scene** | A single thermal or hyperspectral file (or folder, e.g. EnMAP). The atomic unit of input. Three v1 sensor types: PRISMA (HE5), Landsat 9 (TIF), EnMAP (GeoTIFF folder + XML). |
| **Onboarding** | The act of bringing a Scene into the Library. Triggered from the Scenes page's Ingest button. |
| **Annotation** | Optional artifact associated with a Scene. v1 starting type: raster mask / ground-truth file. A Scene can have **zero, one, or many** Annotations. Used to produce overlay visuals on the Scene Detail page. |
| **Project** | A workspace bound to **exactly one** Scene. A Scene can have many Projects. The "big tent" containing Actions, Visualizations, Notes, and a 1:1 Result. |
| **Action** | A verb taken on a Scene within a Project. Sequential (not a DAG). Catalog locked at 3 types in v1 (see § 7). Lifecycle: `queued → running → complete \| failed \| cancelled`. |
| **Action Output** | The artifact every completed Action produces. Persistent, owned by the Action, addressable. **Invariant:** every Action has exactly one Output. |
| **Note** | Free-form project-owned text. Can inline-reference Actions and the Scene. Reference syntax deferred. |
| **Visualization** | A first-class **curated** project-level item. Source = Scene OR Action Output, paired with a Visualization Template. Persisted via a synchronous "save" path (no queue, no worker). May or may not be tied to a specific Action. |
| **Visualization Template** | A reusable presentation specification (band assignments, colormap, threshold, overlay style, plot style). Applied at view time (ephemeral) or stored as part of a Visualization (persistent). |
| **Result** | The auto-live, project-level rollup. **1:1 with Project.** Composes from Actions + Visualizations + Notes. No "finalize" step. |
| **Export** | A snapshot of the current Result state, bundled into a downloadable package. Triggered by a button. |

## 7. Action catalog (v1)

Three types only:

1. **Anomaly Detection**
   - Input: Scene (annotations may be passed as auxiliary in future).
   - Configuration: list of algorithms to run (one Action can fan out across multiple algorithms internally).
   - Output: bundled artifact containing per-algorithm results + a comparison view.

2. **Spectral Detection**
   - Input: Scene + an Action Output (typically an anomaly mask) and/or a region of interest.
   - Configuration: detection parameters.
   - Output: spectral signature data + flagged regions.

3. **Cloud Masking**
   - Input: Scene.
   - Configuration: thresholds; defaults from existing B10 adaptive cloud masker.
   - Output: cloud / clear binary mask aligned to the Scene.

**Rendering is not an Action.** Rendering is always lightweight; persistence happens via the Visualization "save" path (see § 8). If gigapixel/tiled rendering becomes a real need later, a `render` Action type can be added then.

### Action input model

```
Action
  - type:           anomaly_detection | spectral_detection | cloud_mask
  - input_data:     scene_id | action_output_id | both, depending on type
  - configuration:  type-specific JSONB
```

### Multi-algorithm rule

"Run multiple algorithms" is a single Action with a list-valued algorithm field, not multiple Actions. The Output is a bundled comparison.

## 8. Visualizations & Templates

### Origin paths (uniform)

A Visualization is created by selecting a **source** + a **Template** in a viewer and saving:
- Source = **Scene** (Scene Detail page viewer), OR
- Source = **Action Output** (Action detail pane in the Project workspace).

In both cases, save is **synchronous** via the **api** (no queue, no worker). Saves return in milliseconds.

### Template definition (provisional)

```
VisualizationTemplate
  - id, name, description
  - input_type:    scene | action_output
  - applicable_to: list of sensor types (for scene templates) or Action types (for output templates)
  - configuration: JSONB (band assignments, colormap, threshold, overlay opts, etc.)
```

### Action Output invariance + Visualization curation

- **Action Outputs are invariant:** every completed Action has its Output, period. Outputs are accessible via the Action's detail pane.
- **Visualizations are a curated layer on top.** A user views an Action Output under a Template, decides it's worth keeping, saves it as a Visualization. The Output stays unchanged whether or not the user saves a Visualization from it.

## 9. Result and Export

- **Result** is auto-live and 1:1 with the Project. Always reflects the current state.
- Composes from: all Actions (and their Outputs by reference), all curated Visualizations, all Notes.
- **No "finalize" / "publish" step.**
- **Export** is a separate button that snapshots the current Result state into a downloadable bundle on demand.

## 10. Backend & runtime architecture

(Distilled from [final design.md](final design.md). Detailed rationale lives there.)

### Image set (4 + bootstrap)

```
postgres   ·   api   ·   worker (GPU)   ·   frontend
                          ↑
                     bootstrap (one-shot, seeds volumes on first run)
```

- **api** — FastAPI (or equivalent), lightweight, no GPU. Handles auth, CRUD, Visualization saves (synchronous), Action submission (enqueue), status polling.
- **worker** — heavy ML deps + CUDA + `--gpus all`. Pulls Action jobs from the Postgres-backed queue. Concurrency=1 in v1.
- **postgres** — application data + jobs queue.
- **frontend** — Vite + React + TypeScript, served behind nginx in prod compose.
- **bootstrap** — one-shot service that seeds the named volumes from baked-in tarballs on first run; no-op afterwards.

### Queue

- **Postgres-backed.** `SELECT … FOR UPDATE SKIP LOCKED`. No Redis, no Celery, no RabbitMQ.
- Action lifecycle stored on the Action row itself (`queued → running → complete | failed | cancelled`).
- Worker heartbeats; reaper marks stale jobs as `failed`.
- Cancel-while-queued: supported (mark the row). Cancel-while-running: best-effort (worker checks at boundaries).

### Status delivery

- **Polling**, not WebSockets, in v1.
- Cadence ~1s while any visible Action is queued/running; ~5s otherwise. Defer fine-tuning.

### Volumes (named, partitioned)

- `allotrope_db` — Postgres data dir.
- `allotrope_data` — scene-rooted source data.
  ```
  scenes/<scene_id>/
    raw/...                                  raw scene file(s) (HE5 / TIF / EnMAP folder)
    vendable/...                             VendableDataset built during onboarding
    annotations/<annotation_id>/...          attached annotation files
  ```
- `allotrope_models` — model checkpoints.
- `allotrope_artifacts` — derived artifacts; project-rooted for project-owned items, scene-rooted for scene-level derived files.
  ```
  scenes/<scene_id>/
    thumbnail.png                            generated during onboarding
  projects/<project_id>/
    actions/<action_id>/output/...           per-Action artifacts
    visualizations/<visualization_id>/...    curated visuals (Project-scoped)
    exports/<result_id>/<bundle>             Result snapshots on export
  ```
  This layout makes Project deletion `rm -rf allotrope_artifacts/projects/<project_id>/` — one operation, no orphans. Same property at the Action / Visualization sub-level.

**Path convention:** all `file_path` / `*_path` columns in DB hold paths **relative to the volume mount**. The volume name is config, not data — the api/worker resolves them at read time.

### Architecture pinning

- All images built with `--platform linux/amd64` (demo machine: Linux x86_64 + NVIDIA).
- CUDA base on the worker image (version pinned to demo server's CUDA version).

## 11. Cross-cutting rules

- **Working altitude:** structure first, details during implementation. Do not pre-decide button placement, polling cadence specifics, modal-vs-wizard, tile variants, etc.
- **No DAG / pipeline composer UX.** Sequential Action list, period.
- **No comparison surface inside Projects.** Switching between Actions is sufficient.
- **No login bypass / dev shortcuts in shipped builds.** Demo is single-machine but auth is real.

## 12. Explicitly deferred to implementation

These are known questions; they have no impact on schema or API design and will be resolved when we build:

- Specific Scene listing columns (driven by metadata captured at onboarding).
- Filter UI shape on the Scenes landing (left rail / drawer / chip bar).
- Onboarding flow shape (modal / wizard / page).
- Ingest button placement on the Scenes landing.
- Whether the Home Ingest tile is renamed/dropped/kept-as-is.
- Whether annotations also drive evaluation metrics (precision/recall/IoU panels) or only visualization overlay.
- Note reference syntax for inline Action / Scene mentions.
- Per-Action-type Output viewer details.
- Project workspace exact pane layout.
- Visualization Template scope (global vs. project-scoped vs. both).
- Visualization Template management surface (Settings vs. in-context vs. elsewhere).
- Empty states across the app.
- Polling cadence specifics.
- Models page detailed layout.
- Monitoring page detailed layout.
- Failed-login UX details.
- Multi-user / role model (single seeded account is fine for v1).

---

## Index of locked decisions (quick reference)

| # | Decision | Source session |
|---|---|---|
| 1 | Operational + commercial audience, 30 min, no single wow moment | 2 |
| 2 | Walk-away line: fund more compute for sophisticated models | 2 |
| 3 | Anti-AI-template aesthetic, branded utility, instrument-console anchor | 2, 3 |
| 4 | Real auth, Postgres-backed users, hashed passwords | 3 |
| 5 | Sidebar: Home, Scenes, Projects, Models, Monitoring (5 destinations) | 4, 10, 11 |
| 6 | Top bar: active project, GPU status, CPU/GPU/RAM sparklines, user, settings | 4 |
| 7 | Home: 5 live tiles | 4, 8, 11 |
| 8 | Scenes landing = table; Scene Detail page exists with 3 required parts | 9 |
| 9 | Ingest is a button on Scenes, not a separate destination | 10 |
| 10 | Annotations are optional, multiple per Scene, raster-mask v1, combined with Scene for visuals | 9 |
| 11 | Project = workspace bound to one Scene; Scene can have many Projects | 5, 6 |
| 12 | Action is the unit of work; sequential list (no DAG composer) | 6, 7 |
| 13 | Action catalog v1 = 3 types: anomaly_detection, spectral_detection, cloud_mask | 7, 15 |
| 14 | Multi-algorithm = one Action | 7 |
| 15 | Postgres-backed queue, polling for status | 8, final design.md §4 |
| 16 | Result = auto-live, 1:1 with Project, no finalize step | 8 |
| 17 | Notes = project-owned, can reference Actions and Scene | 8 |
| 18 | Models = read-only catalog (model-card pattern) | 8 |
| 19 | Project lifecycle: just exist + delete | 12 |
| 20 | Project workspace surfaces: Actions list + detail pane + Visualizations + Notes + Result panel | 12, 14 |
| 21 | No comparison surface | 12 |
| 22 | Action Outputs invariant; Visualizations curated | 14 |
| 23 | Visualizations = curated, source = Scene OR Action Output, sync save via api | 14, 15 |
| 24 | Visualization Templates: view-time + saved roles, decouple data from presentation | 12, 13, 15 |
| 25 | Rendering is not an Action; rendering is lightweight and synchronous | 15 |
| 26 | Working altitude: structure first, details during implementation | 10 |
