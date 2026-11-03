// Full-screen viewer for a spectral_library_match action: two maps side by side (the classified
// objects and only the anomaly points), a class selection that drives both maps, the object list
// and the export, a man-made sensitivity, and the selected object's card.
//
// Opened from SpectralLibraryMatchOutputViewer (ActionDetailPane.tsx). With summary.namer ok
// the viewer works on namer OBJECTS (docs/lld/spectral-namer-in-product.md); both overlays are
// painted here so the selection and the sensitivity redraw them without a re-run. Older actions
// (no namer) get one map with the worker's family / SAM overlay PNG and a pixel card, and export
// the full bundle.
//
// Design: docs/proposals/class-driven-flow.md (Gap 2 UI, additions A and D), drawn in the Tufte
// style agreed 2026-11-01 - the maps and the numbers dominate, controls are quiet text toggles,
// labels sit next to the data. The header's All | Man-made | Natural is the primary filter, of the
// anomaly points map too (v1.0.1); the class rows only refine man-made (user feedback the same day). The two maps are small multiples: same size, same scale, one
// pan / zoom (MatchMap PanLink). On a narrow window they become a switch.
//
// Export: `classes=&confident_only=&manmade_p=` (selection.ts), counted by the api's
// selection_counts. Until the api has that endpoint (a plain 404) the label uses the screen's own
// count and the export falls back to the older `subset=manmade|natural` where the selection is
// exactly that; other selections then say why they cannot be exported yet.

import { useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";

import { downloadStoryGif } from "../../api/story";
import { type Box, MatchMap, PanLink, type Px } from "./MatchMap";
import {
  type FamilyData, type HistogramRow, type MatchRow, type NamerData, type NamerObject,
  type PixelSpectra, type Spectra2D, type SpectralLibraryMatchSummary, INK_3, NAMER_MANMADE,
  NAMER_NATURAL, NO_GEOREF_NOTE, className, fileUrl, key, loadFamily, loadHistogram, loadMatches, loadNamer,
  loadObjectCoords, loadPixelSpectra, loadSamLibrary,
} from "./data";
import {
  type ColourBy, ClassTree, MaterialsList, ObjectCard, ObjectList, PixelCard, Toggle, ToggleSet,
  callConfidence,
} from "./panels";
import { MARK_PX, paintPoints, pointMarkers } from "./points";
import {
  type Call, type Coarse, CUT_MAX, CUT_MIN, CUTS, DEFAULT_CUT, type ServerCounts, callsAt, coarseOf, coarseSel,
  countPicked, exportQuery, fetchSelectionCounts, legacySubset, manmadeLeaves, picked, selectionName,
  tokens,
} from "./selection";

// Alpha (of 255) of objects the selection leaves out: still visible as context.
const DIMMED_ALPHA = 45;
const TENTATIVE_ALPHA = 120;
// How far (image px) a hover or click looks for the nearest anomaly pixel.
const SNAP_PX = 4;
const NARROW = "(max-width: 900px)";
const fmt = (n: number) => n.toLocaleString("en-US");

// A file the action may not have: null until loaded, null if absent, never throws.
function useLoad<T>(load: () => Promise<T | null>, enabled: boolean, deps: unknown[]) {
  const [state, setState] = useState<{ data: T | null; done: boolean; error: string | null }>(
    { data: null, done: !enabled, error: null });
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    load()
      .then((data) => { if (!cancelled) setState({ data, done: true, error: null }); })
      .catch((err) => {
        console.error("[spectral-match] load failed", err);
        if (!cancelled) setState({ data: null, done: true, error: String(err) });
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

function useNarrow() {
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW).matches);
  useEffect(() => {
    const mq = window.matchMedia(NARROW);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return narrow;
}

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

// The classified overlay at image size. Objects outside the selection are dimmed, tentative
// classes faded in the material colouring, and drawn pixels get a darker one-pixel halo so a
// single-pixel object still reads at full-scene zoom (as the worker's PNGs do).
function paintNamer(w: number, h: number, data: NamerData, calls: Map<number, Call>,
                    colourBy: ColourBy, show: (o: NamerObject) => boolean): ImageData {
  const im = new ImageData(w, h);
  const d = im.data;
  const rgb = new Map<string, number[]>();
  const colourOf = (c: Call) => {
    const col = colourBy === "level1"
      ? (c.call === "manmade" ? NAMER_MANMADE : NAMER_NATURAL)
      : (c.call === "manmade" ? data.classes.get(c.cls)?.colour ?? NAMER_MANMADE : INK_3);
    if (!rgb.has(col)) rgb.set(col, hex(col));
    return rgb.get(col)!;
  };
  const lit: number[] = [];
  for (const p of data.pixels) {
    if (p.row >= h || p.col >= w) continue;
    const o = data.byId.get(p.obj_id);
    const c = o && calls.get(o.obj_id);
    if (!o || !c) continue;
    const on = show(o);
    const [r, g, b] = colourOf(c);
    const a = !on ? DIMMED_ALPHA
      : colourBy === "material" && c.call === "manmade" && !c.confident ? TENTATIVE_ALPHA : 255;
    const i = (p.row * w + p.col) * 4;
    d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = a;
    if (on) lit.push(p.row * w + p.col);
  }
  for (const at of lit) {
    const row = Math.floor(at / w);
    const col = at - row * w;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const rr = row + dr;
        const cc = col + dc;
        if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
        const j = (rr * w + cc) * 4;
        if (d[j + 3] !== 0) continue;
        const i = at * 4;
        d[j] = (d[i] * 3) >> 2; d[j + 1] = (d[i + 1] * 3) >> 2; d[j + 2] = (d[i + 2] * 3) >> 2;
        d[j + 3] = (d[i + 3] * 7) >> 3;
      }
    }
  }
  return im;
}

// The nearest key in `has` within SNAP_PX of px.
function snap(px: Px, has: (k: number) => boolean): Px | null {
  if (has(key(px.row, px.col))) return px;
  let best: Px | null = null;
  let bestD = Infinity;
  for (let dr = -SNAP_PX; dr <= SNAP_PX; dr++) {
    for (let dc = -SNAP_PX; dc <= SNAP_PX; dc++) {
      const d = dr * dr + dc * dc;
      if (d >= bestD || d > SNAP_PX * SNAP_PX) continue;
      if (has(key(px.row + dr, px.col + dc))) { best = { row: px.row + dr, col: px.col + dc }; bestD = d; }
    }
  }
  return best;
}

const boxOf = (o: NamerObject): Box => ({ r0: o.row_min, r1: o.row_max, c0: o.col_min, c1: o.col_max });

// Saves the zip; true when the api wrote it in pixel coordinates (the scene has no georeference).
async function download(r: Response, fallback: string): Promise<boolean> {
  const blob = await r.blob();
  const name = r.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1] ?? fallback;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
  return r.headers.get("X-Allotrope-Georef") === "none";
}

async function failure(r: Response): Promise<string> {
  let detail = `HTTP ${r.status}`;
  try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
  if (typeof detail !== "string") detail = JSON.stringify(detail);
  return `Export failed: ${detail}`;
}

export function SpectralMatchModal({ actionId, summary, onClose }: {
  actionId: string;
  summary: Record<string, unknown>;
  onClose: () => void;
}) {
  const s = summary as SpectralLibraryMatchSummary;
  const hasNamer = s.namer?.status === "ok";
  const hasFamily = s.family?.status === "ok";

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const namer = useLoad<NamerData>(() => loadNamer(actionId), hasNamer, [actionId]);
  const coords = useLoad<Map<number, [number, number]>>(() => loadObjectCoords(actionId), hasNamer, [actionId]);
  const matches = useLoad<Map<number, MatchRow[]>>(() => loadMatches(actionId), true, [actionId]);
  const fam = useLoad<FamilyData>(() => loadFamily(actionId), hasFamily, [actionId]);
  const spectra = useLoad<PixelSpectra>(() => loadPixelSpectra(actionId), true, [actionId]);
  const samLib = useLoad<Spectra2D>(() => loadSamLibrary(actionId), true, [actionId]);
  const histogram = useLoad<HistogramRow[]>(() => loadHistogram(actionId), !hasNamer && !hasFamily, [actionId]);

  const nd = namer.data;
  const [cut, setCut] = useState(DEFAULT_CUT);
  const [selState, setSel] = useState<Set<string> | null>(null);
  const [confidentOnly, setConfidentOnly] = useState(false);
  const [colourBy, setColourBy] = useState<ColourBy>("level1");
  const [showMm, setShowMm] = useState(true);
  const [showNat, setShowNat] = useState(true);
  const [selectedOnly, setSelectedOnly] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [focus, setFocus] = useState<{ box: Box; seq: number } | null>(null);
  const [hover, setHover] = useState<Px | null>(null);
  const [pinned, setPinned] = useState<Px | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const narrow = useNarrow();
  const [pane, setPane] = useState<"classified" | "points">("classified");
  const link = useMemo(() => new PanLink(), []);

  // Default selection: every man-made object, as the worker's own man-made subset.
  const sel = useMemo(() => selState ?? new Set(nd ? manmadeLeaves(nd) : []), [selState, nd]);
  // The slider repaints both maps on every step; deferring keeps the thumb responsive.
  const paintCut = useDeferredValue(cut);
  const calls = useMemo(() => (nd ? callsAt(nd, paintCut) : new Map<number, Call>()), [nd, paintCut]);
  const isPicked = useCallback((o: NamerObject) => picked(calls.get(o.obj_id)!, sel, confidentOnly),
                               [calls, sel, confidentOnly]);
  const onPoints = useCallback((o: NamerObject) => {
    const mm = calls.get(o.obj_id)?.call === "manmade";
    return (mm ? showMm : showNat) && (!selectedOnly || isPicked(o));
  }, [calls, showMm, showNat, selectedOnly, isPicked]);

  const classImage = useMemo(
    () => (nd && dims ? paintNamer(dims.w, dims.h, nd, calls, colourBy, isPicked) : null),
    [nd, dims, calls, colourBy, isPicked]);
  const pointsImage = useMemo(
    () => (nd && dims ? paintPoints(dims.w, dims.h, nd, calls, onPoints) : null), [nd, dims, calls, onPoints]);
  const coarse: Coarse = nd ? coarseOf(sel, nd) : "manmade";
  // The header switch also decides which kinds of point the right map shows; its caption
  // toggles refine that until the switch moves again.
  useEffect(() => {
    setShowMm(coarse !== "natural");
    setShowNat(coarse !== "manmade");
  }, [coarse]);
  // The ochre halo marks the selection only when it is a part of the objects.
  const haloed = useCallback((o: NamerObject) => coarse !== "all" && isPicked(o), [coarse, isPicked]);
  const pointObjs = useMemo(() => (nd ? nd.objects.filter(onPoints) : []), [nd, onPoints]);
  const pointIds = useMemo(() => new Set(pointObjs.map((o) => o.obj_id)), [pointObjs]);
  const pointCounts = useMemo(() => {
    let mm = 0;
    let nat = 0;
    for (const o of nd?.objects ?? []) {
      if (selectedOnly && !isPicked(o)) continue;
      if (calls.get(o.obj_id)?.call === "manmade") mm += 1; else nat += 1;
    }
    return { mm, nat };
  }, [nd, calls, selectedOnly, isPicked]);

  // The list follows the selection, the surest calls first.
  const listed = useMemo(() => {
    if (!nd) return [];
    return nd.objects.filter(isPicked)
      .sort((a, b) => callConfidence(b, calls.get(b.obj_id)!) - callConfidence(a, calls.get(a.obj_id)!));
  }, [nd, isPicked, calls]);

  // Counts for the export label: the api's when it answers, the screen's own meanwhile.
  const query = nd ? exportQuery(sel, confidentOnly, cut, nd) : "";
  const local = useMemo(() => (nd ? countPicked(nd, calls, sel, confidentOnly) : { objects: 0, pixels: 0 }),
                        [nd, calls, sel, confidentOnly]);
  const [server, setServer] = useState<{ q: string; c: ServerCounts | "unsupported" | null } | null>(null);
  useEffect(() => {
    if (!hasNamer || !nd || server?.c === "unsupported") return;
    const ac = new AbortController();
    const t = window.setTimeout(() => {
      void fetchSelectionCounts(actionId, query, ac.signal).then((c) => setServer({ q: query, c }));
    }, 250);
    return () => { window.clearTimeout(t); ac.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [actionId, query, hasNamer, nd]);
  const legacyApi = server?.c === "unsupported";
  const counted = server && server.q === query && server.c && server.c !== "unsupported" ? server.c : local;

  const selectedObj = nd && selected != null ? nd.byId.get(selected) ?? null : null;
  const hoverObj = nd && hoverId != null ? nd.byId.get(hoverId) ?? null : null;
  const select = (o: NamerObject, zoom: boolean) => {
    setSelected(o.obj_id);
    if (zoom) setFocus((f) => ({ box: boxOf(o), seq: (f?.seq ?? 0) + 1 }));
  };
  // The object under the pointer: an anomaly pixel within SNAP_PX, else the nearest marker
  // centroid within a marker's reach (single pixels are far below a screen pixel when zoomed out).
  const objectAt = (px: Px | null, zoom: number, pool: NamerObject[] | null): NamerObject | null => {
    if (!px || !nd) return null;
    const inPool = pool ? pointIds : null;
    const hit = snap(px, (k) => {
      const p = nd.pixelByKey.get(k);
      return Boolean(p && (!inPool || inPool.has(p.obj_id)));
    });
    const p = hit ? nd.pixelByKey.get(key(hit.row, hit.col)) : undefined;
    if (p) return nd.byId.get(p.obj_id) ?? null;
    const reach = (MARK_PX / 2 + 2) / zoom;
    let best: NamerObject | null = null;
    let bestD = reach * reach;
    for (const o of pool ?? nd.objects) {
      const d = (o.row + 0.5 - px.row) ** 2 + (o.col + 0.5 - px.col) ** 2;
      if (d <= bestD) { best = o; bestD = d; }
    }
    return best;
  };
  // Older actions: pixels with a result.
  const pixelHas = (k: number) =>
    Boolean(matches.data?.has(k) || fam.data?.byKey.has(k) || spectra.data?.rowOf.has(k));

  const [exportBusy, setExportBusy] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportNote, setExportNote] = useState<string | null>(null);
  const handleExport = async () => {
    setExportBusy(true);
    setExportError(null);
    setExportNote(null);
    const base = `/api/actions/${encodeURIComponent(actionId)}/export`;
    const short = actionId.replace(/^action_/, "").slice(0, 8);
    try {
      if (!hasNamer || !nd) {
        const r = await fetch(base, { method: "POST" });
        if (!r.ok) throw new Error(await failure(r));
        if (await download(r, `hyper_${short}.zip`)) setExportNote(NO_GEOREF_NOTE);
        return;
      }
      const legacy = legacySubset(sel, confidentOnly, cut, nd);
      const post = (q: string) => fetch(`${base}?${q}`, { method: "POST" });
      if (legacyApi && !legacy) {
        throw new Error("This server's export knows only man-made or natural at Balanced; pick one of those.");
      }
      let r = await post(legacyApi ? `subset=${legacy}` : query);
      // An api that validates query params but predates `classes`: the older subset, when it fits.
      if (!r.ok && r.status === 422 && !legacyApi && legacy) {
        const again = await post(`subset=${legacy}`);
        if (again.ok) r = again;
      }
      if (!r.ok) throw new Error(await failure(r));
      if (await download(r, `hyper_${short}_${selectionName(sel, nd).replace(/[^a-z]+/gi, "_")}.zip`)) {
        setExportNote(NO_GEOREF_NOTE);
      }
    } catch (err) {
      setExportError(String((err as Error).message ?? err));
    } finally {
      setExportBusy(false);
    }
  };

  // The story GIF (docs/lld/story-gif.md) of the same selection: for presentations, not the pack.
  const [storyBusy, setStoryBusy] = useState(false);
  const handleStory = async () => {
    setStoryBusy(true);
    setExportError(null);
    try {
      await downloadStoryGif(actionId, hasNamer && nd ? { classes: tokens(sel, nd), confidentOnly, manmadeP: cut } : {});
    } catch (err) {
      setExportError(String((err as Error).message ?? err));
    } finally {
      setStoryBusy(false);
    }
  };

  const nm = s.namer;
  const loadError = namer.error ?? (hasNamer && namer.done && !nd ? "namer results missing" : null);
  const status = loadError ? <span className="smv-error">Could not load: {loadError}</span>
    : hasNamer && !namer.done ? "loading…" : null;
  const nMm = nd ? [...calls.values()].filter((c) => c.call === "manmade").length : nm?.n_objects_manmade ?? 0;
  const nAll = nd ? nd.objects.length : nm?.n_objects ?? 0;

  const describe = (o: NamerObject) => {
    const c = calls.get(o.obj_id)!;
    return (
      <>
        Object {o.obj_id}: <strong>{c.call === "manmade" ? "man-made" : "natural"}</strong>
        {c.call === "manmade" && nd && <>, closest material {className(nd, c.cls)}{c.confident ? "" : " (tentative)"}</>}
        , {fmt(o.n_px)} px{isPicked(o) ? "" : " · not in the selection"}
      </>
    );
  };
  const readout = hasNamer
    ? (hoverObj ? describe(hoverObj) : "Hover or click an object; the two maps move together.")
    : hover && pixelHas(key(hover.row, hover.col))
      ? <>row {hover.row}, col {hover.col}{matches.data?.get(key(hover.row, hover.col))?.[0]
          ? ` · ${matches.data.get(key(hover.row, hover.col))![0].name}` : ""}</>
      : "Click a highlighted pixel to see its closest material.";

  const mapHover = (pool: NamerObject[] | null) => (px: Px | null, zoom: number) => {
    if (hasNamer) setHoverId(objectAt(px, zoom, pool)?.obj_id ?? null);
    else setHover(px);
  };
  const mapPick = (pool: NamerObject[] | null) => (px: Px, zoom: number) => {
    if (hasNamer) {
      const o = objectAt(px, zoom, pool);
      if (o) select(o, false);
    } else {
      const hit = snap(px, pixelHas);
      if (hit) setPinned(hit);
    }
  };
  const rings = {
    selected: selectedObj ? boxOf(selectedObj) : null,
    hovered: hoverObj && hoverObj.obj_id !== selected ? boxOf(hoverObj) : null,
  };

  const classified = (
    <MatchMap
      id="classified"
      link={link}
      rgbUrl={fileUrl(actionId, "rgb.png")}
      overlayUrl={hasNamer ? undefined
        : fileUrl(actionId, hasFamily ? "family_map_overlay.png" : "match_map_overlay.png")}
      overlayImage={classImage}
      {...rings}
      marked={hasNamer ? null : pinned ?? hover}
      focus={focus}
      status={status}
      caption={!hasNamer ? <>Closest material per anomaly pixel.</>
        : colourBy === "level1"
          ? <>Classified objects: <span className="smv-ink-mm">man-made</span> and{" "}
              <span className="smv-ink-nat">natural</span>; the selection in full colour, the rest faded.</>
          : <>Classified by closest material, coloured as in the class list; tentative classes
              faded, natural in grey.</>}
      onDims={(w, h) => setDims({ w, h })}
      onHover={mapHover(null)}
      onPick={mapPick(null)}
    />
  );
  const points = hasNamer && (
    <MatchMap
      id="points"
      link={link}
      rgbUrl={fileUrl(actionId, "rgb.png")}
      dimRgb
      overlayImage={pointsImage}
      markers={(v) => (nd ? pointMarkers(v, pointObjs, calls, haloed) : null)}
      {...rings}
      focus={focus}
      caption={<>
        Only the committed anomaly points:{" "}
        <Toggle on={showMm} onClick={() => setShowMm(!showMm)} title="Show or hide man-made points">
          <span className="smv-ink-mm">■</span> man-made {fmt(pointCounts.mm)}</Toggle>
        <span className="smv-sep"> · </span>
        <Toggle on={showNat} onClick={() => setShowNat(!showNat)} title="Show or hide natural points">
          <span className="smv-ink-natpt">●</span> natural {fmt(pointCounts.nat)}</Toggle>
        <span className="smv-sep"> · </span>
        <Toggle on={selectedOnly} onClick={() => setSelectedOnly(!selectedOnly)}
                title="Only what the export will contain">selected only</Toggle>
      </>}
      onDims={(w, h) => setDims({ w, h })}
      onHover={mapHover(pointObjs)}
      onPick={mapPick(pointObjs)}
    />
  );

  return (
    <div className="anomaly-viewer__modal-backdrop" onClick={onClose}>
      <div className="anomaly-viewer__modal spectral-match-modal" role="dialog" aria-modal="true"
           aria-label="Spectral match" onClick={(e) => e.stopPropagation()}>
        <header className="smv-header">
          <h3 className="smv-title">Spectral match</h3>
          {hasNamer && (
            <p className="smv-lede">
              {fmt(nAll)} anomaly objects: {fmt(nMm)} man-made, {fmt(nAll - nMm)} natural.
            </p>
          )}
          <button type="button" className="smv-toggle smv-close" onClick={onClose} aria-label="Close">close</button>
        </header>
        <div className="smv-controls">
          {hasNamer && nd && (
            <ToggleSet label="Show" value={coarse} onChange={(c) => setSel(coarseSel(c, nd))} options={[
              { value: "all", label: <>All <span className="smv-num">{fmt(nAll)}</span></> },
              { value: "manmade", label: <>Man-made <span className="smv-num">{fmt(nMm)}</span></> },
              { value: "natural", label: <>Natural <span className="smv-num">{fmt(nAll - nMm)}</span></> },
            ]} />
          )}
          {hasNamer && (
            <>
              <span className="smv-cut">
                <ToggleSet label="Call man-made when P is above" value={cut} onChange={setCut}
                           options={CUTS.map((c) => ({ value: c.p, label: <>{c.label} <span className="smv-num">{c.p}</span></> }))} />
                <input type="range" className="smv-cut__range" min={CUT_MIN} max={CUT_MAX} step={0.01}
                       value={cut} aria-label="Call man-made when P is above"
                       onChange={(e) => setCut(Math.round(Number(e.target.value) * 100) / 100)} />
                <span className="smv-num smv-cut__val">{cut.toFixed(2)}</span>
              </span>
              <ToggleSet label="Colour" value={colourBy} onChange={setColourBy} options={[
                { value: "level1", label: "man-made / natural" },
                { value: "material", label: "material" },
              ]} />
            </>
          )}
          <span className="smv-export">
            <button type="button" className="smv-export__btn" onClick={handleExport}
                    disabled={exportBusy || (hasNamer && (!nd || counted.objects === 0))}
                    title="Download a GeoTIFF + shapefile pack of the selection">
              {exportBusy ? "Exporting…" : !hasNamer ? "Export"
                : `Export ${nd ? selectionName(sel, nd) : ""} — ${fmt(counted.objects)} objects`}
            </button>
            <button type="button" className="smv-toggle smv-story" onClick={handleStory}
                    disabled={storyBusy || (hasNamer && (!nd || counted.objects === 0))}
                    title="Animated GIF of how the scene was narrowed down to this selection. For presentations; not part of the export pack">
              {storyBusy ? "Rendering…" : "Story GIF"}
            </button>
            {exportError && <span className="smv-error" title={exportError}>{exportError}</span>}
            {exportNote && <span className="smv-quiet">{exportNote}</span>}
          </span>
        </div>

        <div className={`smv-body${hasNamer ? "" : " smv-body--single"}`}>
          <div className="smv-figs">
            {hasNamer && narrow && (
              <ToggleSet label="Map" value={pane} onChange={setPane} options={[
                { value: "classified", label: "classified" },
                { value: "points", label: "anomaly points" },
              ]} />
            )}
            <div className={`smv-figs__maps${hasNamer && !narrow ? " smv-figs__maps--two" : ""}`}>
              {(!narrow || pane === "classified" || !hasNamer) && classified}
              {(!narrow || pane === "points") && points}
            </div>
            <p className="smv-readout">{readout}</p>
          </div>

          <aside className="smv-side">
            {hasNamer && nd && (
              <>
                <section className="smv-panel">
                  <h4 className="smv-label">Show and export</h4>
                  <ClassTree data={nd} calls={calls} sel={sel} confidentOnly={confidentOnly}
                             colourBy={colourBy} onChange={setSel} />
                  <p className="smv-panel__foot">
                    <Toggle on={confidentOnly} onClick={() => setConfidentOnly(!confidentOnly)}
                            title={`Man-made objects whose closest class is at least ${nd.confidentP} likely`}>
                      confident only</Toggle>
                    <span className="smv-quiet"> (man-made classes at P ≥ {nd.confidentP})</span>
                  </p>
                </section>
                <section className="smv-panel smv-panel--grow">
                  <h4 className="smv-label">
                    Selected objects <span className="smv-num">{fmt(listed.length)}</span>
                  </h4>
                  <ObjectList objects={listed} data={nd} calls={calls} coords={coords.data}
                              selected={selected} hovered={hoverId}
                              onSelect={(o) => select(o, true)}
                              onHover={(o) => setHoverId(o?.obj_id ?? null)} />
                </section>
                {selectedObj && (
                  <ObjectCard obj={selectedObj} call={calls.get(selectedObj.obj_id)!} cut={cut} data={nd}
                              coords={coords.data} matches={matches.data} fam={fam.data} spectra={spectra.data} />
                )}
              </>
            )}
            {!hasNamer && (
              <>
                {nm && nm.status !== "disabled" && (
                  <p className="smv-quiet">Man-made / natural call unavailable ({nm.status}).</p>
                )}
                <section className="smv-panel">
                  <h4 className="smv-label">{hasFamily ? "Material families" : "Top materials"}</h4>
                  <MaterialsList fam={fam.data} famCounts={s.family?.counts} histogram={histogram.data} />
                </section>
                {pinned ? (
                  <PixelCard px={pinned} matches={matches.data} fam={fam.data} spectra={spectra.data}
                             samLib={samLib.data} />
                ) : (
                  <p className="smv-quiet">Click a highlighted pixel on the map.</p>
                )}
              </>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
