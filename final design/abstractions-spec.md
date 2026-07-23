---
title: Abstractions Spec — Allotrope (build-ready entity model)
status: locked at structural altitude (column types, indexes, exact API paths to be coloured in during implementation)
distilled_from: abstractions.md (cross-cutting + 13 entity walks, 2026-05-09)
companion: storyboard-spec.md, final design.md
---

# Abstractions Spec

This is the build-ready distillation of the entity model. It contains only **locked structural decisions** — no chronology, no rationale, no open questions. Read this end-to-end and you have everything needed to derive the Postgres DDL and the API surface.

For chronological context, decision history, and rationale, see [abstractions.md](abstractions.md).
For UX, sidebar destinations, and what the user sees, see [storyboard-spec.md](storyboard-spec.md).
For containerization, queue, bundle, and auth flow, see [final design.md](final%20design.md).

---

## 1. Cross-cutting decisions (apply to every entity)

| ID | Decision |
|---|---|
| CC-1 | **Identifiers:** UUID PKs (Postgres `uuid` type). API serializes as **prefixed strings** at the wire boundary (`<entity>_<uuid>`). Recommendation: UUID v7 generation for index locality. |
| CC-2a | **Timestamps:** `created_at` on every row. `updated_at` only on entities that legitimately mutate post-creation (see § 3 deviation list). |
| CC-2b | **Deletion model:** hard-delete with explicit cascade rules per relationship. No soft-delete (`deleted_at`). |
| CC-3 | **Ownership:** `user_id` on Project (the ownership root). Scenes are library-shared with `created_by_user_id` for audit only. Visualizations / Notes / Actions inherit ownership through Project. Models and Templates are system-shared. |
| CC-4 | **Audit:** `created_by_user_id` only on Projects & Scenes. No dedicated audit log table. |
| CC-5 | **File-vs-row split:** metadata + small JSONB in Postgres; binary artifacts in volumes via relative paths. Volume name is config; paths are stored relative to the mount. |
| CC-6 | **Naming:** Postgres-standard. `snake_case` plural tables, `id` PK column, `<entity>_id` FK columns, `_at` suffix for timestamps, `is_*` for booleans, JSONB columns named by purpose (`configuration`, `metadata`, `payload`, `summary`). |

**Wire-format prefix registry** (CC-1):

| Entity | Prefix |
|---|---|
| User | `user_` |
| Scene | `scene_` |
| Annotation | `annotation_` |
| Project | `project_` |
| Action | `action_` |
| ActionOutput | `output_` |
| ActionTemplate | `action_template_` |
| Visualization | `visualization_` |
| VisualizationTemplate | `visualization_template_` |
| Note | `note_` |
| NoteReference | `note_ref_` |
| Export | `export_` |
| Job | `job_` |

Separator is `_` (URL-safe; not `#`).

## 2. Authentication (JWT, stateless)

| | |
|---|---|
| **Algorithm** | HS256 (HMAC-SHA256, single server secret) |
| **Browser storage** | `HttpOnly; Secure; SameSite=Strict` cookie |
| **Lifetime** | 24 hours |
| **Server secret** | Generated once at bundle bootstrap, persisted in api env / Docker secret |
| **Claims** | `sub` (User uuid), `iat`, `exp`, `username`, optionally `display_name` |
| **Sessions table** | None (stateless) |
| **Logout** | Clear cookie via `Max-Age=0`. Token remains technically valid until `exp`; browser no longer holds it. |
| **CSRF** | Mitigated by `SameSite=Strict` + same-origin api/frontend |

The api never reads the User row on authenticated requests; the claims are sufficient. See [final design.md](final%20design.md) "Auth flow end-to-end" for the full sequence.

## 3. Mutability deviation list (CC-2a)

Entities with `updated_at` (legitimately mutable post-creation):
- **User** — `password_hash`, `display_name`, `email` change.
- **Project** — `name`, `description` change.
- **Note** — `content` change.
- **ActionTemplate** — `name`, `description`, `configuration` editable for user-created templates.
- **VisualizationTemplate** — same as ActionTemplate.
- **Visualization** — `name`, `description` change (rename).

All other entities are write-once (state transitions on a few of them — Action, Job — use named timestamp columns like `started_at` / `completed_at` instead of a generic `updated_at`).

## 4. Storage layout (volumes)

Locked in [storyboard-spec.md § 10](storyboard-spec.md):

```
allotrope_db                          Postgres data dir

allotrope_data                        scene-rooted source data
  scenes/<scene_id>/
    raw/...                           raw scene file(s) (HE5 / TIF / EnMAP folder)
    vendable/...                      VendableDataset built during onboarding
    annotations/<annotation_id>/...   attached annotation files

allotrope_models                      model checkpoints

allotrope_artifacts                   derived artifacts
  scenes/<scene_id>/
    thumbnail.png                     generated during onboarding
  projects/<project_id>/
    actions/<action_id>/output/...    per-Action artifacts
    visualizations/<viz_id>/...       curated visuals (Project-scoped)
    exports/<export_id>/<bundle>      Result snapshots on export
```

**Path convention:** all `*_path` columns hold paths **relative to the volume mount**. The volume name is config, not data. Project-rooted layout means `rm -rf allotrope_artifacts/projects/<id>/` removes everything for that project — same property at Action / Visualization sub-levels.

## 5. Entities

13 tables. Each section gives: purpose, identity, ownership, lifecycle, key invariants, relationships, locked field set, special notes.

### 5.1 User

The authenticated identity. Ownership root for Projects.

- **Identity:** `id : uuid` PK. Wire `user_<uuid>`.
- **Natural key:** `username` (unique, case-insensitive).
- **Ownership:** root.
- **Lifecycle:** no states. Created via bootstrap seed (v1) or admin path (v2). Hard-delete **RESTRICT-ed** if any Projects reference the user.
- **Mutability:** `username` immutable; `password_hash`, `email`, `display_name`, `last_login_at` mutable. Has `updated_at`.

```
User
  id              uuid             (PK; wire: user_<uuid>)
  username        text             (unique, case-insensitive, immutable)
  email           text             (unique, case-insensitive)
  password_hash   text             (argon2id encoded)
  display_name    text?
  last_login_at   timestamptz?
  is_admin        boolean          (default false; v1 admin/create-user gate)
  created_at      timestamptz
  updated_at      timestamptz
```

`is_admin` is a v1 binary flag — not a full role system (deferred per § 10).
The seeded user has `is_admin=true`; only admins can call `POST /admin/users`.

### 5.2 Scene

An onboarded thermal or hyperspectral file. The atomic unit of input. Library-shared.

- **Identity:** `id : uuid` PK. Wire `scene_<uuid>`. Natural uniqueness: `(sensor_type, sensor_scene_id)`.
- **Ownership:** none (library-shared). `created_by_user_id` is audit-only.
- **Lifecycle:** **option B** — Scene row exists only after onboarding succeeds. In-flight and failed onboardings live in `jobs`. No `status` column.
- **Mutability:** fully immutable. No `updated_at`.
- **Storage:** `raw/`, `vendable/`, optional `annotations/` under `allotrope_data/scenes/<id>/`. Thumbnail at `allotrope_artifacts/scenes/<id>/thumbnail.png`.

```
Scene
  id                      uuid             (PK; wire: scene_<uuid>)
  sensor_type             text             (enum: prisma | landsat9 | enmap)
  sensor_scene_id         text             (parsed from filename)
  name                    text             (display; default = sensor_scene_id)
  acquisition_at          timestamptz?
  processing_level        text?            (L2D / L2SP / L2A …)
  product_type            text?
  bbox_min_lon            numeric          (EPSG:4326)
  bbox_min_lat            numeric          (EPSG:4326)
  bbox_max_lon            numeric          (EPSG:4326)
  bbox_max_lat            numeric          (EPSG:4326)
  native_projection       text?
  band_count              int
  cloud_cover_pct         numeric?
  valid_pixel_pct         numeric?
  has_annotations         boolean          (denormalized)
  metadata                jsonb            (sensor-specific extras)
  raw_path                text             (relative to allotrope_data)
  vendable_path           text             (relative to allotrope_data)
  thumbnail_path          text?            (relative to allotrope_artifacts)
  created_at              timestamptz
  created_by_user_id      uuid?            (FK → users.id; SET NULL on user delete)
```

UNIQUE: `(sensor_type, sensor_scene_id)`.

**Conventions:**
- `bbox_*` always in EPSG:4326. `native_projection` carries the source CRS for downstream tooling.
- Timestamps are `timestamptz` in UTC; STAC's tz-naive datetimes get normalized at insert.
- `has_annotations` maintained at app level (in the same transaction as the Annotation insert/delete).
- No PostGIS in v1. Add later if polygon containment / distance queries become real.

### 5.3 Annotation

Optional artifact attached to a Scene. v1: raster mask only.

- **Identity:** `id : uuid`. Wire `annotation_<uuid>`.
- **Ownership:** Scene (library-shared).
- **Lifecycle:** created via `scene_onboard` (bundled) or `annotation_attach` (post-hoc) jobs. **Option B** — row exists only on success. Synchronous **delete** via api.
- **Mutability:** fully immutable. No `updated_at`.
- **Storage:** `allotrope_data/scenes/<scene_id>/annotations/<id>/<filename>`.

```
Annotation
  id                      uuid             (PK; wire: annotation_<uuid>)
  scene_id                uuid             (FK → scenes.id; CASCADE on scene delete)
  type                    text             (enum: raster_mask in v1)
  name                    text
  description             text?
  file_path               text             (relative to allotrope_data)
  metadata                jsonb            (class label map for multi-class masks, source/provenance, etc.)
  created_at              timestamptz
  created_by_user_id      uuid?            (FK → users.id; SET NULL on user delete)
```

When inserted/deleted, the api updates `scenes.has_annotations` in the same transaction.

**Type extensibility (CC-13a):** the `type` column is plain `text`, not a Postgres ENUM. New types are added via the **annotation type registry** — a per-type spec module on the worker side that owns:

| Field | Purpose |
|---|---|
| `KIND` | string written into `annotations.type` |
| `LABEL` | human-readable name for the UI |
| `ACCEPTED_EXTENSIONS` | tuple of file extensions for upload validation |
| `validate_upload(filename)` | api-side cheap reject (extension / format check) |
| `materialise(staging, final_dir)` | worker move/transform; returns final relative path |
| `render_overlay(file, dest_png)` | worker emits an RGBA PNG layer the frontend stacks on the viewport. Returns False when the type has no spatial overlay. |
| `extract_metadata(file)` | type-specific stats (band count, polygon count, …) → `Annotation.metadata` JSONB |

The registry lives at `backend/allotrope_worker/annotation_types/`; the api imports it for upload validation and the worker imports it for dispatch. Adding a new type is a single new module + one registry entry on each side.

**Planned types beyond v1:**

| `KIND` | Use case | Storage shape |
|---|---|---|
| `raster_mask` | binary or multi-class pixel labels (v1) | TIF, single band |
| `multi_class_raster_mask` | per-class labels with categorical palette | TIF, single band; `metadata.class_palette` |
| `polygon_vector` | anomaly regions as vector polygons | GeoJSON FeatureCollection |
| `point_features` | individual pixel hotspots | GeoJSON points or CSV |
| `bounding_boxes` | rectangular ROIs (detection benchmarks) | JSON / GeoJSON |

All types share the same `Annotation` row shape; `metadata` JSONB carries the type-specific extras (class palette, feature count, source provenance, …). The overlay renderer rasterises vector / point / box types to RGBA PNGs in v1 so the frontend's overlay layer stays uniform; SVG / canvas overlays for vector types are a future option if zoom-level fidelity becomes important.

### 5.4 Project

A workspace bound to exactly one Scene. The structural pivot of the application.

- **Identity:** `id : uuid`. Wire `project_<uuid>`. No natural uniqueness (no name constraint).
- **Ownership:** root for Actions / Visualizations / Notes / Exports / project-scoped Jobs.
- **Lifecycle:** no states. Just exists + deletable. CASCADE down to all children.
- **Mutability:** `name`, `description` mutable. Has `updated_at`. `user_id` and `scene_id` immutable.

```
Project
  id              uuid             (PK; wire: project_<uuid>)
  user_id         uuid             (FK → users.id; RESTRICT on user delete; immutable)
  scene_id        uuid             (FK → scenes.id; RESTRICT on scene delete; immutable)
  name            text             (NOT NULL; mutable)
  description     text?            (mutable)
  created_at      timestamptz
  updated_at      timestamptz
```

### 5.5 Action

A verb taken on a Scene within a Project. The unit of investigation work.

- **Identity:** `id : uuid`. Wire `action_<uuid>`.
- **Ownership:** Project (no `user_id` — inherits).
- **Lifecycle:** `queued → running → complete | failed | cancelled`. Created via api with `status='queued'` AND a paired `action_run` Job enqueued. Worker keeps `actions.status` in sync with `jobs.status` inside its transaction. **Not individually deletable in v1** — cleanup via Project delete only. Cancel-while-queued: direct row update. Cancel-while-running: `cancellation_requested=true`; worker checks at boundaries.
- **Mutability:** lifecycle/timestamp columns mutable; everything else immutable. No `updated_at` (lifecycle uses `started_at` / `completed_at`).

```
Action
  id                      uuid             (PK; wire: action_<uuid>)
  project_id              uuid             (FK → projects.id; CASCADE)
  action_template_id      uuid?            (FK → action_templates.id; SET NULL — NOT NULL at insert,
                                            nullable in storage to support template delete)
  type                    text             (enum: anomaly_scoring | spectral_detection | cloud_mask)
  configuration           jsonb            (params + input refs; copied from template at submit time)
  status                  text             (enum: queued | running | complete | failed | cancelled)
  failure_reason          text?
  started_at              timestamptz?
  completed_at            timestamptz?
  cancellation_requested  boolean          (default false)
  created_at              timestamptz
```

**Inputs** live inside `configuration` JSONB:

```json
{
  "input_scene_id": "scene_<uuid>",
  "input_action_output_ids": ["output_<uuid>", ...],
  "input_annotation_ids": ["annotation_<uuid>", ...],
  "params": { ... }
}
```

API validates references at submit time; **no FK enforcement** (referential safety comes from the cascade structure — Actions only delete via Project). Multi-algorithm runs are a single Action with `params.algorithms = [...]`.

**Invariant:** `status='complete'` ⇔ exactly one ActionOutput row exists for this Action.

### 5.6 ActionOutput

The artifact every completed Action produces. 1:1 with its Action when `status='complete'`.

- **Identity:** `id : uuid`. Wire `output_<uuid>`. `action_id` is UNIQUE.
- **Lifecycle:** created in the same worker transaction that marks the Action complete. Immutable.
- **Storage:** `allotrope_artifacts/projects/<project_id>/actions/<action_id>/output/...` (per Project-rooted layout).

```
ActionOutput
  id                      uuid             (PK; wire: output_<uuid>)
  action_id               uuid             (FK → actions.id, UNIQUE; CASCADE)
  artifact_path           text             (relative to allotrope_artifacts; the Output's directory)
  summary                 jsonb            (metrics, comparison tables, small structured data)
  created_at              timestamptz
```

**Single unified table.** Per-Action-type schema variation handled by api/worker layer (Pydantic models per type), not by separate tables. For multi-algorithm runs, the directory contains per-algorithm subdirs + a comparison artifact; `summary` holds the per-algorithm metrics.

### 5.7 ActionTemplate

A reusable recipe for Action runs. System-seeded defaults + user save-as.

- **Identity:** `id : uuid`. Wire `action_template_<uuid>`.
- **Ownership:** system-shared (no `user_id`, no `project_id`).
- **Lifecycle:** seeded at bootstrap (one default per Action type). User templates created via "save as Template" at successful run completion. Editable for user templates; read-only for `is_system=true`. Deletion fires `SET NULL` on `actions.action_template_id`.
- **Mutability:** `name`, `description`, `configuration` mutable for user templates. Has `updated_at`. `type`, `is_system` immutable.

```
ActionTemplate
  id              uuid             (PK; wire: action_template_<uuid>)
  type            text             (enum: anomaly_scoring | spectral_detection | cloud_mask)
  name            text
  description     text?
  configuration   jsonb            (params only — algorithms, thresholds, model refs;
                                    NOT input refs)
  is_system       boolean
  created_at      timestamptz
  updated_at      timestamptz
```

`configuration` is **copied** into Action.configuration at submit time, not referenced live (history-preservation).

### 5.8 VisualizationTemplate

A reusable presentation specification. Same shape as ActionTemplate.

- **Identity:** `id : uuid`. Wire `visualization_template_<uuid>`.
- **Ownership:** system-shared.
- **Lifecycle:** bootstrap seeds one default per Scene sensor type (for scene-targeted) and per Action type (for output-targeted). User save-as supported. Deletion fires `SET NULL` on `visualizations.template_id`.
- **Mutability:** same pattern as ActionTemplate.

```
VisualizationTemplate
  id              uuid             (PK; wire: visualization_template_<uuid>)
  input_kind      text             (enum: scene | action_output)
  applicable_to   jsonb            (list of compatible sensor types or Action types)
  name            text
  description     text?
  configuration   jsonb            (band assignments, colormap, threshold, overlay opts, etc.)
  is_system       boolean
  created_at      timestamptz
  updated_at      timestamptz
```

### 5.9 Visualization

A first-class curated, project-level persisted visual. Saved synchronously by the api when the user clicks "save as Visualization" in any viewer.

- **Identity:** `id : uuid`. Wire `visualization_<uuid>`.
- **Ownership:** Project (no separate `user_id`).
- **Lifecycle:** synchronous create; immutable in source/template/artifact; **mutable in `name` / `description`**; individually deletable.
- **Storage:** `allotrope_artifacts/projects/<project_id>/visualizations/<id>/<filename>`.
- **Polymorphic source:** exactly one of `source_scene_id` or `source_action_output_id` is non-NULL; `source_kind` discriminates.

```
Visualization
  id                          uuid             (PK; wire: visualization_<uuid>)
  project_id                  uuid             (FK → projects.id; CASCADE)
  source_kind                 text             (enum: scene | action_output)
  source_scene_id             uuid?            (FK → scenes.id; CASCADE)
  source_action_output_id     uuid?            (FK → action_outputs.id; CASCADE)
  template_id                 uuid?            (FK → visualization_templates.id; SET NULL)
  name                        text             (NOT NULL; mutable)
  description                 text?            (mutable)
  artifact_path               text             (relative to allotrope_artifacts)
  created_at                  timestamptz
  updated_at                  timestamptz
```

CHECK constraint:

```sql
CHECK (
  (source_kind = 'scene'         AND source_scene_id IS NOT NULL AND source_action_output_id IS NULL) OR
  (source_kind = 'action_output' AND source_action_output_id IS NOT NULL AND source_scene_id IS NULL)
)
```

### 5.10 Note

Project-scoped free-form text (markdown).

- **Identity:** `id : uuid`. Wire `note_<uuid>`.
- **Ownership:** Project.
- **Lifecycle:** synchronous create/edit/delete. Mutable; has `updated_at`.

```
Note
  id              uuid             (PK; wire: note_<uuid>)
  project_id      uuid             (FK → projects.id; CASCADE)
  content         text             (markdown; NOT NULL; mutable)
  created_at      timestamptz
  updated_at      timestamptz
```

### 5.11 NoteReference

Typed pointer from a Note to a specific entity within the same Project.

- **Identity:** `id : uuid`. Wire `note_ref_<uuid>`.
- **Ownership:** Note.
- **Lifecycle:** created/deleted by the api in sync with the Note's content (when the user inserts or removes a reference in the editor).
- **Polymorphic targets:** exactly one of `ref_project_id`, `ref_action_id`, `ref_output_id`, `ref_viz_id`, `ref_scene_id` is non-NULL.
- **All FKs CASCADE.** Deleting any referenced entity removes the NoteReference rows but leaves the Note itself intact.

```
NoteReference
  id                  uuid             (PK; wire: note_ref_<uuid>)
  note_id             uuid             (FK → notes.id; CASCADE)
  ref_project_id      uuid?            (FK → projects.id; CASCADE)
  ref_action_id       uuid?            (FK → actions.id; CASCADE)
  ref_output_id       uuid?            (FK → action_outputs.id; CASCADE)
  ref_viz_id          uuid?            (FK → visualizations.id; CASCADE)
  ref_scene_id        uuid?            (FK → scenes.id; CASCADE)
  created_at          timestamptz
```

CHECK constraint:

```sql
CHECK (
  (CASE WHEN ref_project_id IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN ref_action_id  IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN ref_output_id  IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN ref_viz_id     IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN ref_scene_id   IS NOT NULL THEN 1 ELSE 0 END
  ) = 1
)
```

API validates the referenced entity is in the same Project as the Note.

### 5.12 Export

A persisted snapshot of a Project's Result state, packaged into a downloadable bundle.

- **Identity:** `id : uuid`. Wire `export_<uuid>`.
- **Ownership:** Project.
- **Lifecycle:** **async via the queue** (`project_export` job). **Option B** — Export row exists only after the job succeeds. Fully immutable thereafter.
- **Storage:** `allotrope_artifacts/projects/<project_id>/exports/<id>/<filename>`.

```
Export
  id              uuid             (PK; wire: export_<uuid>)
  project_id      uuid             (FK → projects.id; CASCADE)
  bundle_path     text             (relative to allotrope_artifacts)
  snapshot_at     timestamptz      (moment the bundle was assembled)
  size_bytes      bigint
  format          text             (e.g. "tar.gz" / "zip")
  created_at      timestamptz
```

### 5.13 Job

The Postgres-backed queue. The single source of truth for asynchronous work.

- **Identity:** `id : uuid`. Wire `job_<uuid>`.
- **Ownership:** heterogeneous. `project_id` is nullable; populated only for project-scoped types (`action_run`, `project_export`). Library-scoped types (`scene_onboard`, `annotation_attach`) have `project_id IS NULL`.
- **Lifecycle:** `queued → running → complete | failed | cancelled`. Worker pulls via `SELECT … WHERE status='queued' FOR UPDATE SKIP LOCKED LIMIT 1`. Heartbeats while running; reaper marks stale jobs `failed`. **Persists after completion as audit history** (visible in the Jobs sidebar). No auto-delete in v1.
- **Mutability:** lifecycle / timestamp / target columns mutable. No generic `updated_at`.

```
Job
  id                       uuid             (PK; wire: job_<uuid>)
  type                     text             (enum: scene_onboard | annotation_attach | action_run | project_export)
  status                   text             (enum: queued | running | complete | failed | cancelled)
  project_id               uuid?            (FK → projects.id; CASCADE; non-NULL for action_run, project_export)
  payload                  jsonb            (type-specific input spec)
  target_kind              text?            (e.g. 'scene', 'annotation', 'action_output', 'export')
  target_id                uuid?            (id of the produced/updated entity; SOFT REF, no FK)
  failure_reason           text?
  cancellation_requested   boolean          (default false)
  started_at               timestamptz?
  last_heartbeat_at        timestamptz?
  completed_at             timestamptz?
  created_at               timestamptz
```

**`target_id` is a soft reference** — no FK enforcement. The polymorphism is inherent (different job types produce different entity kinds). If the target is deleted, `target_id` becomes a dangling informational ref (acceptable for audit history).

For `action_run` Jobs: the Action's `status` is denormalized from this Job's status; worker keeps them in sync inside its transactions.

## 6. Cascade summary

| Source | Target | Cascade |
|---|---|---|
| User | Project | RESTRICT |
| User | Scene | SET NULL on `created_by_user_id` |
| User | Annotation | SET NULL on `created_by_user_id` |
| Scene | Project | RESTRICT |
| Scene | Annotation | CASCADE |
| Scene | Visualization | CASCADE (via `source_scene_id`; in practice fires via Project) |
| Scene | NoteReference | CASCADE (via `ref_scene_id`) |
| Project | Action | CASCADE |
| Project | Visualization | CASCADE |
| Project | Note | CASCADE |
| Project | NoteReference | CASCADE (via `ref_project_id`) |
| Project | Export | CASCADE |
| Project | Job | CASCADE (only when `project_id` set: `action_run`, `project_export`) |
| Action | ActionOutput | CASCADE |
| Action | NoteReference | CASCADE (via `ref_action_id`) |
| ActionOutput | Visualization | CASCADE (via `source_action_output_id`) |
| ActionOutput | NoteReference | CASCADE (via `ref_output_id`) |
| ActionTemplate | Action | SET NULL on `action_template_id` |
| VisualizationTemplate | Visualization | SET NULL on `template_id` |
| Visualization | NoteReference | CASCADE (via `ref_viz_id`) |
| Note | NoteReference | CASCADE |

## 7. Polymorphic patterns

Three places in the schema use multi-nullable-FK polymorphism with a CHECK constraint:

| Entity | Discriminator | FK columns (exactly one non-NULL) |
|---|---|---|
| Visualization | `source_kind` | `source_scene_id`, `source_action_output_id` |
| NoteReference | (implicit; no discriminator column) | `ref_project_id`, `ref_action_id`, `ref_output_id`, `ref_viz_id`, `ref_scene_id` |
| Job (target) | `target_kind` | (no FK — soft ref via `target_id`) |

For Visualization and NoteReference, all FKs CASCADE — this gives DB-level integrity without a polymorphic-type system in Postgres. For Job's target, the polymorphism is too broad (4 different produced entity types) and FK enforcement isn't worth the schema complexity for an audit-history table.

## 8. Job types reference

| Type | Project-scoped? | Target on success | Notes |
|---|---|---|---|
| `scene_onboard` | No | `(scene, <scene_id>)` | Library-scoped. Creates Scene + bundled Annotations atomically. Fails ⇒ no Scene row. |
| `annotation_attach` | No | `(annotation, <annotation_id>)` | Library-scoped. Adds Annotation to existing Scene. |
| `action_run` | Yes | `(action_output, <output_id>)` | Project-scoped. Action row pre-exists; Job runs and creates ActionOutput on success. Action.status mirrors Job.status. |
| `project_export` | Yes | `(export, <export_id>)` | Project-scoped. Bundles Project artifacts on disk; creates Export row on success. |

## 9. Index of locked decisions

| # | Decision | Source |
|---|---|---|
| 1 | UUID PKs with prefixed-string serialization at api boundary | CC-1 |
| 2 | `created_at` everywhere; `updated_at` only on legitimately-mutable entities | CC-2a |
| 3 | Hard-delete with explicit cascade rules; no soft-delete | CC-2b |
| 4 | Ownership root at Project; Library-shared scenes; system-shared models/templates | CC-3 |
| 5 | `created_by_user_id` audit only on Projects & Scenes; no audit log table | CC-4 |
| 6 | File-vs-row split; relative paths in DB; volume name is config | CC-5 |
| 7 | Postgres-standard `snake_case` plural naming | CC-6 |
| 8 | JWT (HS256, HttpOnly+SameSite=Strict cookie, 24h, no sessions table) | § 2 |
| 9 | Project-rooted artifact storage layout | § 4 |
| 10 | Scene lifecycle option B (row only on success; in-flight in jobs) | § 5.2 |
| 11 | No PostGIS in v1 (bbox numeric columns; native_projection text) | § 5.2 |
| 12 | `has_annotations` denormalized, app-level maintenance | § 5.2, § 5.3 |
| 13 | Annotation v1 type: `raster_mask` only | § 5.3 |
| 14 | Annotation attach via queue (`annotation_attach` job); delete synchronous | § 5.3 |
| 15 | Project just-exists + delete; no states; no transient UI state on the row | § 5.4 |
| 16 | Action not individually deletable in v1 | § 5.5 |
| 17 | Action inputs in configuration JSONB; no DB-level FK on input refs | § 5.5 |
| 18 | Single unified `action_outputs` table; type-shape interpretation in api/worker | § 5.6 |
| 19 | `actions.status` denormalized from `jobs.status`; worker keeps in sync | § 5.5 |
| 20 | ActionTemplate seeded defaults + user save-as; lives in Models destination | § 5.7 |
| 21 | Action.action_template_id NOT NULL at insert, nullable in storage; SET NULL on template delete | § 5.7 |
| 22 | Action.configuration copied from template at submit time (history-preservation) | § 5.7 |
| 23 | VisualizationTemplate parity with ActionTemplate; both live in Models | § 5.8 |
| 24 | Visualization individually deletable; rename mutable | § 5.9 |
| 25 | Visualization polymorphic source (Scene OR ActionOutput) with CHECK | § 5.9 |
| 26 | Visualization save synchronous via api (no queue, no worker) | § 5.9 |
| 27 | Note individually deletable; mutable content | § 5.10 |
| 28 | NoteReference table (separate from Note.content) for queryable typed pointers | § 5.11 |
| 29 | NoteReference polymorphic targets (Project / Action / Output / Viz / Scene) | § 5.11 |
| 30 | Result is **not** a DB entity; served as a computed view by the api | § (5 cast) |
| 31 | Export row exists only after `project_export` job succeeds | § 5.12 |
| 32 | Jobs persist after completion (audit history); no auto-delete in v1 | § 5.13 |
| 33 | Job `target_id` is a soft reference (no FK); `target_kind` discriminates | § 5.13 |
| 34 | Job `project_id` nullable; CASCADE only for project-scoped types | § 5.13 |

## 10. Explicitly deferred to implementation

- Exact Postgres column types (`text` vs `varchar(N)`, `numeric(p,s)` precision) and indexes.
- Note reference syntax in content (markdown link format, slash-command, etc.).
- Heartbeat cadence + stale-job threshold for Job (10s/60s is a reasonable starting point).
- Per-Action-type Output schema validation (Pydantic models per type).
- Action Template CRUD UX details (where in Models, edit dialog shape, etc.).
- Visualization save UX details ("save as Template" affordance specifics).
- Cleanup of ancient `scene_onboard` / `annotation_attach` jobs (v2 admin job).
- Refresh-token flow (v2).
- Multi-user / role model (single seeded user fine for v1).
- Per-user template ownership (v2).

These don't affect schema or API design; they're filled in during the build.
