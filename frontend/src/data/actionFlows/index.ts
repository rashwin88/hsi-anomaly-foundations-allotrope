// Per-action-type flow definitions. Each ActionFlow is a small graph
// of stage nodes (5–10 each) describing the recipe end-to-end. Rendered
// via @xyflow/react inside the workspace Action card so the user sees
// what runs *before* the first action even fires.
//
// Reuses the same `modelStage` node component as the Models page —
// consistent visual language across the app. The colour palette is
// driven by `kind`.

import type { Edge, Node } from "@xyflow/react";

import type { ModelFlowNodeData } from "./../modelFlows/types";

export type ActionFlowNode = Node<ModelFlowNodeData>;
export type ActionFlowEdge = Edge;

export interface ActionFlow {
  type: string;
  nodes: ActionFlowNode[];
  edges: ActionFlowEdge[];
}

// --- band_filter_apply ------------------------------------------------

const Y = 88; // step
const C = 0;  // central column

export const bandFilterApplyFlow: ActionFlow = {
  type: "band_filter_apply",
  nodes: [
    { id: "in",    type: "modelStage", position: { x: C, y: 0 * Y },  data: { kind: "input",       label: "Raw scene file",         shape: "PRISMA .he5 / EnMAP folder" } },
    { id: "dn",    type: "modelStage", position: { x: C, y: 1 * Y },  data: { kind: "special",     label: "DN → reflectance",       shape: "sensor-specific transformer", detail: "PRISMA: divide-by-65535 · EnMAP: gain × DN" } },
    { id: "qm",    type: "modelStage", position: { x: C - 280, y: 2 * Y }, data: { kind: "mask",   label: "Quality masks",          shape: "EnMAP only", detail: "cloud · cloud_shadow · haze" } },
    { id: "bf",    type: "modelStage", position: { x: C, y: 2 * Y },  data: { kind: "encoder",     label: "Spectral band filter",   detail: "drop atmospheric windows · trim edges · prune low-coverage bands" } },
    { id: "vox",   type: "modelStage", position: { x: C, y: 3 * Y },  data: { kind: "encoder",     label: "Voxel-fraction spatial mask", detail: "invalidate pixels with too many missing bands" } },
    { id: "fill",  type: "modelStage", position: { x: C, y: 4 * Y },  data: { kind: "decoder",     label: "PCHIP spectral fill",    detail: "shape-preserving interpolation along λ" } },
    { id: "rs",    type: "modelStage", position: { x: C, y: 5 * Y },  data: { kind: "decoder",     label: "Resample → 10 nm grid",  shape: "common wavelength grid", detail: "cross-sensor consistency" } },
    { id: "nv",    type: "modelStage", position: { x: C + 280, y: 5 * Y }, data: { kind: "mask",   label: "Nearest-valid fill",     shape: "optional", detail: "kill boundary artefacts in transformer inference" } },
    { id: "out",   type: "modelStage", position: { x: C, y: 6 * Y },  data: { kind: "output",      label: "filtered_vendable.pkl",  detail: "drop-in input for downstream HSI Actions" } },
  ],
  edges: [
    { id: "e1", source: "in", target: "dn", animated: true },
    { id: "e2", source: "dn", target: "bf", animated: true },
    { id: "eqm", source: "qm", target: "bf", animated: true, label: "EnMAP" },
    { id: "e3", source: "bf", target: "vox", animated: true },
    { id: "e4", source: "vox", target: "fill", animated: true },
    { id: "e5", source: "fill", target: "rs", animated: true },
    { id: "envf", source: "nv", target: "rs", animated: false, type: "smoothstep", style: { strokeDasharray: "4 4" }, label: "applied after" },
    { id: "e6", source: "rs", target: "out", animated: true },
  ],
};

// --- scene_segmentation -----------------------------------------------

export const sceneSegmentationFlow: ActionFlow = {
  type: "scene_segmentation",
  nodes: [
    { id: "in",    type: "modelStage", position: { x: C, y: 0 * Y }, data: { kind: "input",       label: "Filtered vendable",      shape: "from band_filter_apply" } },
    { id: "pick",  type: "modelStage", position: { x: C, y: 1 * Y }, data: { kind: "special",     label: "Pick bands by λ",        shape: "Red 660 · Green 560 · NIR 860 · VNIR end 910" } },
    { id: "ndvi",  type: "modelStage", position: { x: C - 280, y: 2 * Y }, data: { kind: "encoder", label: "NDVI",                shape: "(NIR − Red) / (NIR + Red)" } },
    { id: "ndwi",  type: "modelStage", position: { x: C, y: 2 * Y }, data: { kind: "encoder",     label: "NDWI",                   shape: "(Green − NIR) / (Green + NIR)" } },
    { id: "bri",   type: "modelStage", position: { x: C + 280, y: 2 * Y }, data: { kind: "encoder", label: "Brightness",          shape: "mean VNIR reflectance" } },
    { id: "th",    type: "modelStage", position: { x: C, y: 3 * Y }, data: { kind: "decoder",     label: "Threshold per index",    shape: "→ 4 binary class masks", detail: "water · cloud · shadow · vegetation" } },
    { id: "keep",  type: "modelStage", position: { x: C, y: 4 * Y }, data: { kind: "decoder",     label: "Build keep_mask",        shape: "spatial_valid ∧ ¬(union of class masks)" } },
    { id: "out",   type: "modelStage", position: { x: C, y: 5 * Y }, data: { kind: "output",      label: "8 GeoTIFFs + diagnostics", detail: "ndvi · ndwi · brightness · 4 class masks · keep_mask" } },
  ],
  edges: [
    { id: "e1", source: "in", target: "pick", animated: true },
    { id: "e2a", source: "pick", target: "ndvi", animated: true },
    { id: "e2b", source: "pick", target: "ndwi", animated: true },
    { id: "e2c", source: "pick", target: "bri", animated: true },
    { id: "e3a", source: "ndvi", target: "th", animated: true },
    { id: "e3b", source: "ndwi", target: "th", animated: true },
    { id: "e3c", source: "bri", target: "th", animated: true },
    { id: "e4", source: "th", target: "keep", animated: true },
    { id: "e5", source: "keep", target: "out", animated: true },
  ],
};

// --- cloud_mask -------------------------------------------------------

export const cloudMaskFlow: ActionFlow = {
  type: "cloud_mask",
  nodes: [
    { id: "in",    type: "modelStage", position: { x: C, y: 0 * Y }, data: { kind: "input",       label: "Onboarding vendable",     shape: "thermal · B10 in °C" } },
    { id: "valid", type: "modelStage", position: { x: C, y: 1 * Y }, data: { kind: "mask",        label: "Validity-mask the cube",  shape: "scenes/<id>/vendable.validity_cube" } },
    { id: "probe", type: "modelStage", position: { x: C - 280, y: 2 * Y }, data: { kind: "special", label: "Probe percentiles",   shape: "P2 · P8 · P50 · P92 · P98", detail: "physical anchors for the GMM" } },
    { id: "fit",   type: "modelStage", position: { x: C, y: 2 * Y }, data: { kind: "encoder",     label: "Fit GMM",                 shape: "5 components · means_init = anchors", detail: "trained on sampling_ratio × valid pixels" } },
    { id: "ver",   type: "modelStage", position: { x: C + 280, y: 2 * Y }, data: { kind: "decoder", label: "Physical verification", shape: "cluster_mean < scene_median − 12 °C", detail: "= cloud cluster" } },
    { id: "pred",  type: "modelStage", position: { x: C, y: 3 * Y }, data: { kind: "decoder",     label: "Predict cloud labels",    shape: "→ binary cloud_mask (H, W)" } },
    { id: "keep",  type: "modelStage", position: { x: C, y: 4 * Y }, data: { kind: "decoder",     label: "Build keep_mask",         shape: "spatial_valid ∧ ¬cloud" } },
    { id: "out",   type: "modelStage", position: { x: C, y: 5 * Y }, data: { kind: "output",      label: "cloud_mask + keep_mask + preview", detail: "consumed by anomaly_scoring on thermal" } },
  ],
  edges: [
    { id: "e1", source: "in",    target: "valid", animated: true },
    { id: "e2", source: "valid", target: "probe", animated: true },
    { id: "e3", source: "valid", target: "fit",   animated: true },
    { id: "ep", source: "probe", target: "fit",   animated: true, label: "anchors" },
    { id: "e4", source: "fit",   target: "ver",   animated: true },
    { id: "e5", source: "fit",   target: "pred",  animated: true },
    { id: "ev", source: "ver",   target: "pred",  animated: true, label: "cloud cluster ids" },
    { id: "e6", source: "pred",  target: "keep",  animated: true },
    { id: "e7", source: "keep",  target: "out",   animated: true },
  ],
};

// --- anomaly_scoring ------------------------------------------------

export const anomalyScoringFlow: ActionFlow = {
  type: "anomaly_scoring",
  nodes: [
    { id: "bf",     type: "modelStage", position: { x: C - 280, y: 0 * Y }, data: { kind: "input",       label: "Filtered vendable",        shape: "from band_filter_apply" } },
    { id: "km",     type: "modelStage", position: { x: C + 280, y: 0 * Y }, data: { kind: "mask",        label: "Keep mask",                shape: "from scene_segmentation · optional" } },
    { id: "gt",     type: "modelStage", position: { x: C + 560, y: 0 * Y }, data: { kind: "mask",        label: "GT annotation",            shape: "raster mask · optional · enables ROC" } },
    { id: "fan",    type: "modelStage", position: { x: C, y: 1 * Y }, data: { kind: "special",     label: "Fan out across model_codenames", detail: "1..N picked from the codename catalog" } },
    { id: "m1",     type: "modelStage", position: { x: C - 320, y: 2 * Y }, data: { kind: "encoder",     label: "Model A · load checkpoint", shape: "Indradhanu / Chakshu / Pratibimba / …" } },
    { id: "m2",     type: "modelStage", position: { x: C, y: 2 * Y }, data: { kind: "encoder",     label: "Model B · load checkpoint" } },
    { id: "mn",     type: "modelStage", position: { x: C + 320, y: 2 * Y }, data: { kind: "encoder",     label: "Model N · load checkpoint" } },
    { id: "score1", type: "modelStage", position: { x: C - 320, y: 3 * Y }, data: { kind: "decoder",     label: "Predict + score",          shape: "L1 · SAM · combined · MSE (per-model default; overridable)" } },
    { id: "score2", type: "modelStage", position: { x: C, y: 3 * Y }, data: { kind: "decoder",     label: "Predict + score" } },
    { id: "scoren", type: "modelStage", position: { x: C + 320, y: 3 * Y }, data: { kind: "decoder",     label: "Predict + score" } },
    { id: "roc",    type: "modelStage", position: { x: C + 320, y: 4 * Y }, data: { kind: "loss",        label: "Compute ROC + AUC",        shape: "per model · only when GT attached" } },
    { id: "out",    type: "modelStage", position: { x: C, y: 5 * Y }, data: { kind: "output",      label: "Score raster + reconstruction + RGB previews + diagnostics" } },
  ],
  edges: [
    { id: "e1", source: "bf", target: "fan", animated: true, label: "cube" },
    { id: "e2", source: "km", target: "fan", animated: true, label: "scoring domain" },
    { id: "e3", source: "gt", target: "roc", animated: true, label: "ground truth" },
    { id: "fa", source: "fan", target: "m1", animated: true },
    { id: "fb", source: "fan", target: "m2", animated: true },
    { id: "fc", source: "fan", target: "mn", animated: true },
    { id: "s1", source: "m1", target: "score1", animated: true },
    { id: "s2", source: "m2", target: "score2", animated: true },
    { id: "sn", source: "mn", target: "scoren", animated: true },
    { id: "tr1", source: "score1", target: "roc", animated: false, type: "smoothstep", style: { strokeDasharray: "4 4" } },
    { id: "tr2", source: "score2", target: "roc", animated: false, type: "smoothstep", style: { strokeDasharray: "4 4" } },
    { id: "trn", source: "scoren", target: "roc", animated: false, type: "smoothstep", style: { strokeDasharray: "4 4" } },
    { id: "o1",  source: "score1", target: "out", animated: true },
    { id: "o2",  source: "score2", target: "out", animated: true },
    { id: "on",  source: "scoren", target: "out", animated: true },
    { id: "or",  source: "roc",    target: "out", animated: false, type: "smoothstep", style: { strokeDasharray: "4 4" } },
  ],
};

export const ACTION_FLOWS: Record<string, ActionFlow> = {
  [bandFilterApplyFlow.type]: bandFilterApplyFlow,
  [sceneSegmentationFlow.type]: sceneSegmentationFlow,
  [cloudMaskFlow.type]: cloudMaskFlow,
  [anomalyScoringFlow.type]: anomalyScoringFlow,
};
