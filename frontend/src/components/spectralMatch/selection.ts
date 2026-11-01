// What the analyst picked on the spectral match screen, and what each object is called at the
// chosen sensitivity (docs/proposals/class-driven-flow.md, Gap 2 and addition A).
//
// Sensitivity: the worker calls an object man-made when its mean P(man-made) is above 0.5
// (app/spectral_match/namer/objects.py MANMADE_CALL_P). The screen can lower that cut-off and
// recomputes every call here, live, from namer_objects.parquet. An object that crosses it takes
// its top mean P(class | man-made) (the prob_<class> columns) as its class, tentative below
// namer_confident_p. Runs from before those columns use the export's lower bound from
// namer_pixels.parquet (per class, the sum of p_cls over the object's pixels whose own top class
// it is, over n_px), exactly as app/spectral_match/class_selection.py does, so the screen and the
// pack agree; with no pixels either the class is "class unknown".
//
// Selection: a set of leaves - each material class, "unknown" and "natural" - always in one of
// three coarse states (user, 2026-11-01: the All | Man-made | Natural switch is the primary
// filter, materials only refine man-made): All = every leaf; Man-made = all or some materials;
// Natural = natural alone. Picking a material implies Man-made; All or Natural clears the picks.
// The export's `classes=` tokens are "manmade" when every man-made leaf is on, else the material
// keys, plus "natural". One predicate (`picked`) drives the classified map, the points panel, the
// list and the export label, so the screen shows what the export will hold. "Confident only"
// narrows the man-made objects; natural objects have no class to be confident about and stay.

import { type NamerData, type NamerObject, className } from "./data";

export const CUTS = [
  { p: 0.5, label: "Balanced" },
  { p: 0.35, label: "Sensitive" },
  { p: 0.25, label: "Very sensitive" },
];
export const DEFAULT_CUT = 0.5;
export const UNKNOWN = "unknown";
export const NATURAL = "natural";

export interface Call {
  call: "manmade" | "natural";
  cls: string;            // "" natural, or man-made with no stored class probabilities
  pCls: number;           // NaN when cls is ""
  confident: boolean;
  moved: boolean;         // the worker called it natural; this cut-off makes it man-made
}

export function callAt(o: NamerObject, cut: number, data: NamerData): Call {
  if (o.p_manmade <= cut) return { call: "natural", cls: "", pCls: NaN, confident: false, moved: false };
  if (o.call === "manmade") {
    return { call: "manmade", cls: o.cls, pCls: o.p_cls, confident: o.confident, moved: false };
  }
  const [cls, p] = o.probs ? topOfProbs(o.probs, data) : topOfPixels(o, data);
  return { call: "manmade", cls, pCls: p, confident: Boolean(cls) && p >= data.confidentP, moved: true };
}

function topOfProbs(probs: number[], data: NamerData): [string, number] {
  let best = 0;
  probs.forEach((p, i) => { if (p > probs[best]) best = i; });
  return [data.classOrder[best], probs[best]];
}

function topOfPixels(o: NamerObject, data: NamerData): [string, number] {
  const sums = new Map<string, number>();
  for (const px of data.pixelsByObj.get(o.obj_id) ?? []) sums.set(px.cls, (sums.get(px.cls) ?? 0) + px.p_cls);
  let best: [string, number] = ["", NaN];
  for (const [c, v] of sums) {
    const p = v / Math.max(o.n_px, 1);
    if (!(p <= best[1])) best = [c, p];
  }
  return best;
}

export function callsAt(data: NamerData, cut: number): Map<number, Call> {
  return new Map(data.objects.map((o) => [o.obj_id, callAt(o, cut, data)]));
}

export const leafOf = (c: Call) => (c.call === "natural" ? NATURAL : c.cls || UNKNOWN);
export const manmadeLeaves = (data: NamerData) => [...data.classOrder, UNKNOWN];

export type Coarse = "all" | "manmade" | "natural";

export function coarseOf(sel: Set<string>, data: NamerData): Coarse {
  if (!sel.has(NATURAL)) return "manmade";
  return manmadeLeaves(data).some((m) => sel.has(m)) ? "all" : "natural";
}

export function coarseSel(c: Coarse, data: NamerData): Set<string> {
  if (c === "natural") return new Set([NATURAL]);
  return new Set(c === "all" ? [...manmadeLeaves(data), NATURAL] : manmadeLeaves(data));
}

// A material click: from All, Natural or every material on, it narrows to that material alone;
// once narrowed it toggles, and clearing the last pick returns to every man-made object.
export function pickMaterial(sel: Set<string>, m: string, data: NamerData): Set<string> {
  const mats = manmadeLeaves(data);
  const narrowed = !sel.has(NATURAL) && !mats.every((x) => sel.has(x));
  if (!narrowed) return new Set([m]);
  const next = new Set(sel);
  if (next.has(m)) next.delete(m); else next.add(m);
  return next.size ? next : new Set(mats);
}

export function picked(c: Call, sel: Set<string>, confidentOnly: boolean) {
  return sel.has(leafOf(c)) && (!confidentOnly || c.call === "natural" || c.confident);
}

export function tokens(sel: Set<string>, data: NamerData): string[] {
  const all = manmadeLeaves(data).every((m) => sel.has(m));
  const out = all ? ["manmade"] : data.classOrder.filter((m) => sel.has(m));
  if (sel.has(NATURAL)) out.push(NATURAL);
  return out;
}

export function selectionName(sel: Set<string>, data: NamerData): string {
  const t = tokens(sel, data);
  if (!t.length) return "nothing";
  return t.map((k) => (k === "manmade" ? "man-made" : k === NATURAL ? "natural" : className(data, k)))
    .join(" + ");
}

export function countPicked(data: NamerData, calls: Map<number, Call>, sel: Set<string>,
                            confidentOnly: boolean) {
  let objects = 0;
  let pixels = 0;
  for (const o of data.objects) {
    if (picked(calls.get(o.obj_id)!, sel, confidentOnly)) { objects += 1; pixels += o.n_px; }
  }
  return { objects, pixels };
}

export function exportQuery(sel: Set<string>, confidentOnly: boolean, cut: number, data: NamerData) {
  return `classes=${tokens(sel, data).join(",")}&confident_only=${confidentOnly}&manmade_p=${cut}`;
}

// The pre-class-driven export only knows subset=manmade|natural at the worker's own 0.5 call.
export function legacySubset(sel: Set<string>, confidentOnly: boolean, cut: number,
                             data: NamerData): "manmade" | "natural" | null {
  if (cut !== DEFAULT_CUT) return null;
  const t = tokens(sel, data).join(",");
  if (t === "manmade" && !confidentOnly) return "manmade";
  if (t === NATURAL) return "natural";
  return null;
}

export interface ServerCounts { objects: number; pixels: number; per_class?: Record<string, number> }

// The api's count of a selection. "unsupported" when the endpoint does not exist yet (FastAPI's
// plain 404 "Not Found"); null on any other failure - the screen then shows its own count.
export async function fetchSelectionCounts(actionId: string, query: string, signal: AbortSignal):
    Promise<ServerCounts | "unsupported" | null> {
  try {
    const r = await fetch(`/api/actions/${encodeURIComponent(actionId)}/spectral_library_match/`
      + `selection_counts?${query}`, { signal });
    if (r.status === 404) {
      const detail = await r.json().then((j) => j.detail).catch(() => null);
      return detail === "Not Found" ? "unsupported" : null;
    }
    if (!r.ok) return null;
    const j = await r.json();
    return { objects: Number(j.objects), pixels: Number(j.pixels), per_class: j.per_class };
  } catch {
    return null;
  }
}
