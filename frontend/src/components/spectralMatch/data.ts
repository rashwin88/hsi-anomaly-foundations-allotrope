// Data layer of the spectral match viewer: the file shapes a spectral_library_match action
// writes, their loaders, and the per-object summaries the right column shows.
//
// Every loader returns null for a file the action did not write (older actions predate the
// namer, family matching or the spectra snapshot), so the viewer degrades instead of failing.
// Pixels are keyed by key(row, col) everywhere: the namer, SAM and family tables each cover a
// different set of pixels, so rows are never matched by position.
//
// File contracts: docs/lld/spectral-namer-in-product.md (namer_*),
// docs/lld/spectral-family-match.md (family_*), _spectral_library_match_run.py (the rest).

export const KEY_W = 65536; // wider than any scene, so row * KEY_W + col is unique
export const key = (row: number, col: number) => row * KEY_W + col;

// The screen's colours, mirroring the shared Tufte tokens (--accent-manmade, --accent-natural,
// --accent-select, --ink, --ink-3 in index.css / tufte.css): a canvas cannot read CSS variables
// cheaply, so the values are repeated here. The worker's own PNGs keep their orange / teal.
export const NAMER_MANMADE = "#c0392b";
export const NAMER_NATURAL = "#2e7d6b";
export const POINT_COLOUR = "#d4a017";        // the selection (outline / halo)
// Points on the dimmed RGB: man-made in --accent-manmade, natural in a clearer leaf green than
// --accent-natural, which reads too teal at marker size (user, 2026-11-01). --accent-natural-point.
export const NATURAL_POINT = "#2e7d32";
export const INK = "#111111";
export const INK_3 = "#999999";

// --- summary.json ------------------------------------------------------

export interface SpectralLibraryMatchSummary {
  upstream_anomaly_detection_output_id?: string;
  sensor_type?: string;
  n_library_entries?: number;
  mode?: string;
  top_k?: number;
  min_coverage?: number;
  min_band_count?: number;
  sg_window_length?: number;
  sg_polyorder?: number;
  chapters?: string[];
  n_pixels?: number;
  n_pixels_matched?: number;
  n_pixels_no_match?: number;
  pixels_matched?: number;            // sent to SAM + family (man-made objects when the namer ran)
  pixels_natural_skipped?: number;    // the namer's natural objects, not matched
  timing_seconds?: { namer?: number | null; match?: number; total?: number };
  namer?: NamerSummary;
  family?: FamilySummary;
}

// Absent on actions that predate the namer; those get the SAM / family view.
export interface NamerSummary {
  status: "ok" | "failed" | "skipped" | "disabled";
  reason?: string;
  error?: string;
  codename?: string;
  checkpoint?: string;
  confident_p?: number;
  n_objects?: number;
  n_objects_manmade?: number;
  n_objects_natural?: number;
  n_objects_confident?: number;
  n_pixels_manmade?: number;
  n_pixels_natural?: number;
  class_counts?: Record<string, number>;
  class_counts_confident?: Record<string, number>;
  timing_seconds?: { infer?: number; total?: number };
}

export interface FamilySummary {
  status: "ok" | "skipped" | "failed";
  reason?: string;
  error?: string;
  alpha?: number;
  p_star?: number;
  n_manmade_families?: number;
  families?: string[];
  n_scored?: number;
  n_not_scored?: number;
  n_named_manmade?: number;
  n_unconfirmed_manmade?: number;
  n_natural?: number;
  counts?: Record<string, number>;
  timing_seconds?: number;
}

// --- namer -------------------------------------------------------------

export interface NamerObject {
  obj_id: number;
  n_px: number;
  row: number;            // centroid
  col: number;
  row_min: number;
  row_max: number;
  col_min: number;
  col_max: number;
  p_manmade: number;
  call: "manmade" | "natural";
  cls: string;            // closest known class, "" for natural objects
  p_cls: number;          // its mean P(class | man-made), NaN for natural objects
  confident: boolean;
  fill: number;
  // Mean P(class | man-made) per class, in NamerData.classOrder; null on runs before the
  // prob_<class> columns (2026-11-01), whose natural objects then have no known class.
  probs: number[] | null;
}

export interface NamerPixel {
  row: number;
  col: number;
  p_manmade: number;
  cls: string;
  p_cls: number;
  fill: number;
  obj_id: number;
}

export interface NamerLegendItem {
  label: string;
  colour: string;
  alpha?: number;
  cls?: string;
  tentative?: boolean;
}

export interface NamerData {
  objects: NamerObject[];                    // sorted by p_manmade, highest first
  byId: Map<number, NamerObject>;
  pixels: NamerPixel[];
  pixelByKey: Map<number, NamerPixel>;
  pixelsByObj: Map<number, NamerPixel[]>;
  classes: Map<string, NamerLegendItem>;     // class name -> its confident legend entry
  classOrder: string[];                      // the run's classes, in legend code order
  confidentP: number;
}

// --- SAM, family, spectra ------------------------------------------------

export interface MatchRow {
  row: number;
  col: number;
  rank: number;
  library_ix: number;
  material_id: string;
  name: string;
  chapter: string;
  asd_subtype: string | null;
  angle_deg: number;
  n_bands_used: number;
}

export interface FamilyLegendItem {
  family: string;
  group: "manmade" | "natural" | "none";
  label: string;
  colour: string;
}

export interface FamilyRow {
  row: number;
  col: number;
  scored: boolean;
  family: string;
  family_group: string;
  family_code: number;
  confidence: number;
  answered: boolean;
  best_entry: string;
}

export interface FamilyData {
  byKey: Map<number, FamilyRow>;
  legend: Record<string, FamilyLegendItem>;  // family_code (as string) -> item
  pStar: number;
  libIx: Map<string, number>;                // best_entry (material_id) -> library row
  libLabels: string[];
  libCls: (string | null)[];                 // namer class per library row; null on older actions
  lib: Spectra2D;
}

// Row-major (N, B) spectra with a per-band validity mask.
export interface Spectra2D {
  refl: Float32Array;
  valid: Uint8Array;
  bands: number;
}

export interface PixelSpectra extends Spectra2D {
  wavelengths: number[];
  rowOf: Map<number, number>;                // key -> row of refl / valid
}

export interface HistogramRow {
  library_ix: number;
  name: string;
  chapter: string;
  count: number;
}

// --- loaders -------------------------------------------------------------

export const fileUrl = (actionId: string, name: string) =>
  `/api/actions/${encodeURIComponent(actionId)}/files/${name}`;

async function fetchBuf(url: string): Promise<ArrayBuffer | null> {
  const r = await fetch(url);
  return r.ok ? r.arrayBuffer() : null;
}

async function fetchJson<T>(url: string): Promise<T | null> {
  const r = await fetch(url);
  return r.ok ? ((await r.json()) as T) : null;
}

export async function readParquet<T>(url: string): Promise<T[] | null> {
  const buf = await fetchBuf(url);
  if (!buf) return null;
  const parquetMod = await import("hyparquet");
  const file = {
    byteLength: buf.byteLength,
    async slice(start: number, end?: number) {
      return buf.slice(start, end ?? buf.byteLength);
    },
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (await (parquetMod as any).parquetReadObjects({ file })) as T[];
}

async function readNpzUrl(url: string) {
  const buf = await fetchBuf(url);
  if (!buf) return null;
  const { readNpz } = await import("../../lib/npy");
  return readNpz(buf);
}

// parquet int64 can come back as BigInt; the viewer wants plain numbers.
function numbers<T>(rows: T[], keys: (keyof T)[]): T[] {
  for (const r of rows) {
    for (const k of keys) (r as Record<keyof T, unknown>)[k] = Number(r[k]);
  }
  return rows;
}

export async function loadNamer(actionId: string): Promise<NamerData | null> {
  const [objects, pixels, legend] = await Promise.all([
    readParquet<NamerObject>(fileUrl(actionId, "namer_objects.parquet")),
    readParquet<NamerPixel>(fileUrl(actionId, "namer_pixels.parquet")),
    fetchJson<{ classes: Record<string, NamerLegendItem>; confident_p: number }>(
      fileUrl(actionId, "namer_legend.json")),
  ]);
  if (!objects || !pixels || !legend) return null;
  numbers(objects, ["obj_id", "n_px", "row", "col", "row_min", "row_max", "col_min", "col_max",
                    "p_manmade", "p_cls", "fill"]);
  for (const o of objects) o.confident = Boolean(o.confident);
  objects.sort((a, b) => b.p_manmade - a.p_manmade);
  numbers(pixels, ["row", "col", "p_manmade", "p_cls", "fill", "obj_id"]);
  const pixelByKey = new Map<number, NamerPixel>();
  const pixelsByObj = new Map<number, NamerPixel[]>();
  for (const p of pixels) {
    pixelByKey.set(key(p.row, p.col), p);
    let bucket = pixelsByObj.get(p.obj_id);
    if (!bucket) pixelsByObj.set(p.obj_id, (bucket = []));
    bucket.push(p);
  }
  const classes = new Map<string, NamerLegendItem>();
  const coded = Object.entries(legend.classes).sort((a, b) => Number(a[0]) - Number(b[0]));
  for (const [, item] of coded) {
    if (item.cls && !item.tentative) classes.set(item.cls, item);
  }
  const classOrder = [...classes.keys()];
  const probKeys = classOrder.map((c) => `prob_${c}`);
  const hasProbs = objects.length > 0 && probKeys.every((k) => k in objects[0]);
  for (const o of objects) {
    const r = o as unknown as Record<string, unknown>;
    o.probs = hasProbs ? probKeys.map((k) => Number(r[k])) : null;
  }
  return {
    objects, byId: new Map(objects.map((o) => [o.obj_id, o])), pixels, pixelByKey, pixelsByObj,
    classes, classOrder, confidentP: Number(legend.confident_p),
  };
}

export async function loadMatches(actionId: string): Promise<Map<number, MatchRow[]> | null> {
  const rows = await readParquet<MatchRow>(fileUrl(actionId, "matches.parquet"));
  if (!rows) return null;
  numbers(rows, ["row", "col", "rank", "library_ix", "angle_deg", "n_bands_used"]);
  const byKey = new Map<number, MatchRow[]>();
  for (const m of rows) {
    const k = key(m.row, m.col);
    let bucket = byKey.get(k);
    if (!bucket) byKey.set(k, (bucket = []));
    bucket.push(m);
  }
  for (const bucket of byKey.values()) bucket.sort((a, b) => a.rank - b.rank);
  return byKey;
}

export async function loadFamily(actionId: string): Promise<FamilyData | null> {
  const [rows, legend, lib] = await Promise.all([
    readParquet<FamilyRow>(fileUrl(actionId, "family_matches.parquet")),
    fetchJson<{ codes: Record<string, FamilyLegendItem>; p_star: number;
                library: { id: string; label: string; cls?: string | null }[] }>(
      fileUrl(actionId, "family_legend.json")),
    readNpzUrl(fileUrl(actionId, "family_library_refl.npz")),
  ]);
  if (!rows || !legend || !lib) return null;
  numbers(rows, ["row", "col", "family_code", "confidence"]);
  // ids / labels are in the legend, in npz row order (readNpz is numeric-only).
  const entries = legend.library ?? [];
  return {
    byKey: new Map(rows.map((f) => [key(f.row, f.col), f])),
    legend: legend.codes,
    pStar: Number(legend.p_star),
    libIx: new Map(entries.map((e, i) => [e.id, i])),
    libLabels: entries.map((e) => e.label),
    libCls: entries.map((e) => e.cls ?? null),
    lib: { refl: lib.refl.data as Float32Array, valid: lib.valid.data as Uint8Array,
           bands: lib.refl.shape[1] },
  };
}

// anomaly_pixel_spectra.npz: every anomaly pixel's smoothed spectrum. Tens of MB on a big
// scene, so it loads after the map and only feeds the spectrum plot.
export async function loadPixelSpectra(actionId: string): Promise<PixelSpectra | null> {
  const npz = await readNpzUrl(fileUrl(actionId, "anomaly_pixel_spectra.npz"));
  if (!npz) return null;
  const rows = npz.rows.data as Int32Array;
  const cols = npz.cols.data as Int32Array;
  const rowOf = new Map<number, number>();
  for (let i = 0; i < rows.length; i++) rowOf.set(key(rows[i], cols[i]), i);
  return {
    wavelengths: Array.from(npz.wavelengths.data as Float64Array),
    refl: npz.spectra.data as Float32Array,
    valid: npz.valid.data as Uint8Array,
    bands: npz.spectra.shape[1],
    rowOf,
  };
}

export async function loadSamLibrary(actionId: string): Promise<Spectra2D | null> {
  const npz = await readNpzUrl(fileUrl(actionId, "library_refl.npz"));
  if (!npz) return null;
  return { refl: npz.refl.data as Float32Array, valid: npz.valid.data as Uint8Array,
           bands: npz.refl.shape[1] };
}

export async function loadHistogram(actionId: string): Promise<HistogramRow[] | null> {
  const j = await fetchJson<{ top1_counts?: HistogramRow[] }>(fileUrl(actionId, "histogram.json"));
  return j ? (j.top1_counts ?? []) : null;
}

// Centroid lat / lon per object, from the api (the browser cannot project). Null when the
// scene has no georeferencing; the viewer then shows row / col.
export async function loadObjectCoords(actionId: string): Promise<Map<number, [number, number]> | null> {
  const j = await fetchJson<{ coords: Record<string, [number, number]> }>(
    `/api/actions/${encodeURIComponent(actionId)}/spectral_library_match/object_coords`);
  if (!j) return null;
  return new Map(Object.entries(j.coords).map(([k, v]) => [Number(k), v]));
}

// --- per-object summaries --------------------------------------------------

export const prettyFamily = (f: string) => f.replace(/_/g, " / ");

export function className(data: NamerData, cls: string) {
  if (!cls) return "class unknown";
  return data.classes.get(cls)?.label ?? cls.replace(/_/g, " / ");
}

// SAM's top-1 over the object's pixels: the most frequent library spectra first.
export function objectSamTop(px: NamerPixel[], matches: Map<number, MatchRow[]> | null) {
  const tally = new Map<number, { name: string; libraryIx: number; n: number; angle: number }>();
  for (const p of px) {
    const top = matches?.get(key(p.row, p.col))?.[0];
    if (!top) continue;
    const t = tally.get(top.library_ix) ?? { name: top.name, libraryIx: top.library_ix, n: 0, angle: 0 };
    t.n += 1;
    t.angle += top.angle_deg;
    tally.set(top.library_ix, t);
  }
  return [...tally.values()]
    .map((t) => ({ ...t, angle: t.angle / t.n }))
    .sort((a, b) => b.n - a.n);
}

// The family verdicts of the object's pixels, in the family LLD's words, most frequent first,
// plus the most frequent best library entry among scored pixels (the plot's comparison).
export function objectFamily(px: NamerPixel[], fam: FamilyData | null) {
  const verdicts = new Map<string, number>();
  const entries = new Map<string, number>();
  for (const p of px) {
    const f = fam?.byKey.get(key(p.row, p.col));
    if (!f) continue;
    const v = !f.scored ? "not scored"
      : f.family_group === "manmade" && !f.answered ? "man-made, type unconfirmed"
        : prettyFamily(f.family);
    verdicts.set(v, (verdicts.get(v) ?? 0) + 1);
    if (f.scored && f.best_entry) entries.set(f.best_entry, (entries.get(f.best_entry) ?? 0) + 1);
  }
  const bestEntry = [...entries.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  return { verdicts: [...verdicts.entries()].sort((a, b) => b[1] - a[1]), bestEntry };
}

// Ascending, positive wavelengths: sensors flag dropped bands with 0.
export function bandOrder(wavelengths: number[]): number[] {
  return wavelengths.map((w, i) => [w, i] as const).filter(([w]) => w > 0)
    .sort((a, b) => a[0] - b[0]).map(([, i]) => i);
}

// Mean spectrum of the given pixels, band by band over valid values; NaN where none is valid.
export function meanSpectrum(spectra: PixelSpectra, cells: { row: number; col: number }[]): number[] {
  const B = spectra.bands;
  const sum = new Float64Array(B);
  const n = new Uint32Array(B);
  for (const c of cells) {
    const i = spectra.rowOf.get(key(c.row, c.col));
    if (i === undefined) continue;
    for (let b = 0; b < B; b++) {
      const v = spectra.refl[i * B + b];
      if (spectra.valid[i * B + b] && Number.isFinite(v)) { sum[b] += v; n[b] += 1; }
    }
  }
  return Array.from(sum, (s, b) => (n[b] ? s / n[b] : NaN));
}

export function libRow(lib: Spectra2D, ix: number): number[] {
  const B = lib.bands;
  return Array.from({ length: B }, (_, b) =>
    lib.valid[ix * B + b] ? lib.refl[ix * B + b] : NaN);
}

// L2-normalise both on the bands where both are finite, as SAM compares them: shapes become
// comparable even when the brightness differs. `order` picks and sorts the bands.
export function normalisedPair(a: number[], b: number[] | null, order: number[]) {
  const ok = order.map((i) => Number.isFinite(a[i]) && (!b || Number.isFinite(b[i])));
  const norm = (v: number[]) =>
    Math.sqrt(order.reduce((s, i, j) => (ok[j] ? s + v[i] * v[i] : s), 0)) || 1;
  const na = norm(a);
  const nb = b ? norm(b) : 1;
  return {
    a: order.map((i, j) => (ok[j] || !b ? a[i] / na : NaN)),
    b: b ? order.map((i, j) => (ok[j] ? b[i] / nb : NaN)) : null,
  };
}

// The library row of class `cls` whose shape is closest to `mean` (smallest spectral angle on
// the bands both have), or null when no row carries that class: older actions' legends have
// no classes, and a class may have no entry covering this sensor.
export function nearestInClass(mean: number[], fam: FamilyData, cls: string): number | null {
  let best: number | null = null;
  let bestCos = -Infinity;
  fam.libCls.forEach((c, ix) => {
    if (c !== cls) return;
    const ref = libRow(fam.lib, ix);
    let dot = 0, na = 0, nb = 0;
    for (let b = 0; b < ref.length; b++) {
      if (!Number.isFinite(mean[b]) || !Number.isFinite(ref[b])) continue;
      dot += mean[b] * ref[b]; na += mean[b] * mean[b]; nb += ref[b] * ref[b];
    }
    const cos = na && nb ? dot / Math.sqrt(na * nb) : -Infinity;
    if (cos > bestCos) { bestCos = cos; best = ix; }
  });
  return bestCos > -Infinity ? best : null;
}
