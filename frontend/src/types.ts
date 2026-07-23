// Shared types between the api client and React components.
// The wire format for ids matches CC-1 in the abstractions-spec:
// `<entity>_<uuid>` (e.g. `user_3f29c4a8-…`).

export interface User {
  id: string;          // wire format: user_<uuid>
  username: string;
  email: string;
  display_name: string | null;
  is_admin: boolean;
}

// Sidebar destination ids (storyboard-spec § 4).
// Six standard + the admin-only "admin-users" surface.
export type Destination =
  | "home"
  | "scenes"
  | "projects"
  | "models"
  | "jobs"
  | "monitoring"
  | "admin-users";

// --- Scenes (abstractions-spec § 5.2) ---------------------------------

export type SensorType = "prisma" | "landsat9" | "enmap" | "aviris_ng" | "hotsat1";

export interface Scene {
  id: string;                          // wire format: scene_<uuid>
  sensor_type: string;
  sensor_scene_id: string;
  name: string;
  acquisition_at: string | null;       // ISO 8601 timestamp
  processing_level: string | null;
  product_type: string | null;
  bbox_min_lon: number;
  bbox_min_lat: number;
  bbox_max_lon: number;
  bbox_max_lat: number;
  native_projection: string | null;
  band_count: number;
  cloud_cover_pct: number | null;
  valid_pixel_pct: number | null;
  has_annotations: boolean;
  metadata: Record<string, unknown>;
  raw_path: string;
  vendable_path: string;
  thumbnail_path: string | null;
  created_at: string;
  created_by_user_id: string | null;   // wire format: user_<uuid>
}

export interface ScenesPage {
  items: Scene[];
  total: number;
  limit: number;
  offset: number;
}

// --- Jobs (abstractions-spec § 5.13) ----------------------------------

export type JobStatus =
  | "queued"
  | "running"
  | "complete"
  | "failed"
  | "cancelled";

export type JobType =
  | "scene_onboard"
  | "annotation_attach"
  | "action_run"
  | "project_export";

export interface Job {
  id: string;                            // wire format: job_<uuid>
  type: JobType | string;
  status: JobStatus;
  project_id: string | null;             // wire format: project_<uuid>
  payload: Record<string, unknown>;
  target_kind: string | null;            // e.g. "scene" once scene_onboard succeeds
  target_id: string | null;              // wire format: <kind>_<uuid>
  failure_reason: string | null;
  cancellation_requested: boolean;
  started_at: string | null;             // ISO 8601
  last_heartbeat_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface OnboardAccepted {
  job_id: string;
  staged_count: number;
  staged_bytes: number;
}

// --- Scene visualizations (Step 8.5) ---------------------------------

export type VisualizationKind =
  | "color"
  | "nir"
  | "swir"
  | "ndvi"
  | "band_mosaic";

export interface VisualizationItem {
  kind: VisualizationKind | string;
  label: string;
  image_url: string;       // api-relative; prepend /api when fetching
}

export interface VisualizationList {
  items: VisualizationItem[];
  histogram_url: string | null;
}

export interface HistogramStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  p2: number;
  p50: number;
  p98: number;
  valid_pct: number;
}

export interface HistogramJson {
  sensor_type: string;
  kind: string;            // "thermal" | "hyperspectral_summary"
  bins: number[];          // length N+1 edges
  counts: number[];        // length N
  stats: HistogramStats | Record<string, never>;
}

export interface SpectrumPoint {
  wavelength_nm: number;
  reflectance: number;
  spectral_family: string | null;
  is_valid: boolean;
}

export interface SpectrumResponse {
  row: number;
  col: number;
  height: number;
  width: number;
  points: SpectrumPoint[];
}

// --- Annotations (Step 9) ---------------------------------------------

export interface Annotation {
  id: string;                          // wire format: annotation_<uuid>
  scene_id: string;                    // wire format: scene_<uuid>
  type: string;                        // v1: "raster_mask"
  name: string;
  description: string | null;
  file_path: string;                   // relative to allotrope_data
  metadata: Record<string, unknown>;
  has_overlay: boolean;                // true if a worker-rendered overlay PNG exists
  created_at: string;                  // ISO 8601
  created_by_user_id: string | null;   // wire format: user_<uuid>
}

export interface AnnotationsList {
  items: Annotation[];
}

export interface AnnotationAttachAccepted {
  job_id: string;
  annotation_name: string;
}

export interface AnnotationTypeCatalogItem {
  kind: string;
  label: string;
  accepted_extensions: string[];
}

export interface AnnotationTypeCatalog {
  items: AnnotationTypeCatalogItem[];
}

// --- Projects (abstractions-spec § 5.4) -------------------------------

export interface Project {
  id: string;                       // wire format: project_<uuid>
  user_id: string;                  // wire format: user_<uuid>
  scene_id: string;                 // wire format: scene_<uuid>
  name: string;
  description: string | null;
  created_at: string;               // ISO 8601
  updated_at: string;
  scene_name: string;               // denormalised on the response
  scene_sensor_type: string;
}

/** Workspace-detail response from GET /projects/{id}. Adds child counts
 * to the base Project shape (Step 11 — counts are 0 until Step 12+
 * source tables land). */
export interface ProjectDetail extends Project {
  action_count: number;
  visualization_count: number;
  note_count: number;
}

export interface ProjectsPage {
  items: Project[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateProjectPayload {
  scene_id: string;                 // wire format: scene_<uuid>
  name: string;
  description?: string;
}

// --- Foundation models (read-only catalog) ----------------------------

export interface ModelCodename {
  name: string;     // e.g. "Indradhanu"
  script: string;   // Devanagari, e.g. "इंद्रधनु"
  meaning: string;
  why: string;
}

export interface ModelSummary {
  architecture: string;
  label: string;
  codename: ModelCodename;
  sensor: string;          // "thermal" | "hyperspectral"
  family: string;          // "reconstruction"
  params: number;
  val_loss: number;
  version: string;
  epoch: number;
  normalization_mode: string;  // "baked_in" | "none"
  doc: string;             // path under repo root
  // Anomaly-scoring capabilities — drive the per-model knob block
  // (scoring / patch / stride / batch / sam_l1_alpha) in NewActionDialog.
  scoring_methods: string[];
  default_scoring_method: string | null;
  default_patch_size: number | null;
  default_stride: number | null;
  default_batch_size: number | null;
  valid_patch_sizes: number[];
}

export interface ModelCurrentCheckpoint {
  file: string;
  version: string;
  epoch: number;
  params: number;
  val_loss: number;
  encoder_dims?: string | null;
  decoder_dim?: number | null;
  spectral_dim_D?: number | null;
}

export interface ModelNormalization {
  mode: string;
  applied_in_forward: boolean;
  stats_shape: number[] | null;
  mean: number | string | null;
  std: number | string | null;
  source: string;
}

export interface ModelDetail {
  architecture: string;
  label: string;
  codename: ModelCodename;
  sensor: string;
  family: string;
  current: ModelCurrentCheckpoint;
  alternatives: Array<Record<string, unknown>>;
  normalization: ModelNormalization;
  doc: string;
  inferencer: string;
  inferencer_module: string;
  notes: string;
  updated_at: string | null;
}

// --- Actions (Step 12) ------------------------------------------------

export interface ActionInputSpec {
  key: string;
  label: string;
  description: string;
  ref_kind: "scene" | "action_output" | "annotation";
  producing_action_types: string[];
  required: boolean;
  cardinality_min: number;
  cardinality_max: number;
}

export interface ActionOutputSpec {
  key: string;
  label: string;
  description: string;
  artifact_type:
    | "raster"
    | "vector"
    | "pickle"
    | "json"
    | "image"
    | "directory";
  filename: string;
  always_present: boolean;
}

export interface ActionTypeMeta {
  type: string;
  label: string;
  short_description: string;
  description: string;
  when_to_use: string;
  inputs: ActionInputSpec[];
  outputs: ActionOutputSpec[];
  accepted_sensor_types: string[];
  default_config_per_sensor: Record<string, Record<string, unknown>>;
}

export interface Action {
  id: string;                                    // action_<uuid>
  project_id: string;                            // project_<uuid>
  action_template_id: string | null;             // action_template_<uuid>
  type: string;
  configuration: Record<string, unknown>;
  // "needs_threshold" is the terminal status for an
  // anomaly_detection_prep action: the worker has written the
  // composite score map but the user hasn't yet committed a
  // threshold. Until the commit pathway exists (ROADMAP step 14.6)
  // this is effectively a terminal state.
  status:
    | "queued"
    | "running"
    | "complete"
    | "failed"
    | "cancelled"
    | "needs_threshold";
  failure_reason: string | null;
  started_at: string | null;
  completed_at: string | null;
  cancellation_requested: boolean;
  created_at: string;
}

export interface ActionOutput {
  id: string;                                    // output_<uuid>
  action_id: string;                             // action_<uuid>
  artifact_path: string;
  summary: Record<string, unknown>;
  created_at: string;
}

export interface ActionDetail extends Action {
  output: ActionOutput | null;
}

export interface ActionsPage {
  items: Action[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateActionPayload {
  type: string;
  configuration: Record<string, unknown>;
  action_template_id?: string;
}
