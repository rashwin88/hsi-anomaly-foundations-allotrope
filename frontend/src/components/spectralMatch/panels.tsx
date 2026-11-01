// Right-column pieces of the spectral match viewer: the class selection, the object list and the
// cards for a selected object (namer actions) or a picked pixel (older actions).
//
// Tufte-style (agreed 2026-11-01): no boxes, badges or filled buttons. Controls are text
// toggles - ink and underlined when on, light grey when off. Class rows are labelled directly,
// "oil / fuel ▏▏▏▏ 62", a thin bar and the count in tabular figures, so the rows are also the
// legend. Words on screen stay plain ("Closest material", "tentative"); probabilities, SAM and
// family internals sit behind "Details". The namer never says an object "is" a material: an
// unseen material still gets the nearest known class (run N002), so the label is "closest".
//
// Every call shown comes from selection.ts `callAt` at the chosen sensitivity, not from the
// stored call, so an object the cut-off moved reads the same here as on the maps. The class
// rows refine the header's All | Man-made | Natural switch (selection.ts `pickMaterial`).

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import {
  type FamilyData, type FamilyRow, type HistogramRow, type MatchRow, type NamerData,
  type NamerObject, type PixelSpectra, type Spectra2D, NAMER_MANMADE, NAMER_NATURAL,
  bandOrder, className, key, libRow, meanSpectrum, nearestInClass, normalisedPair, objectFamily,
  objectSamTop, prettyFamily,
} from "./data";
import { type Call, NATURAL, UNKNOWN, coarseSel, leafOf, manmadeLeaves, pickMaterial } from "./selection";
import { type Line, SpectrumPlot } from "./spectrum";

export type ColourBy = "level1" | "material";

const fmt = (n: number) => n.toLocaleString("en-US");

// How sure the namer is of the call: P(man-made) for man-made, 1 - that for natural.
export const callConfidence = (o: NamerObject, c: Call) =>
  c.call === "manmade" ? o.p_manmade : 1 - o.p_manmade;

export function Toggle({ on, onClick, children, title }: {
  on: boolean;
  onClick: () => void;
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <button type="button" className="smv-toggle" aria-pressed={on} onClick={onClick} title={title}>
      {children}
    </button>
  );
}

// "Balanced 0.5 · Sensitive 0.35 · ..." - exactly one on.
export function ToggleSet<T extends string | number>({ label, value, options, onChange }: {
  label: string;
  value: T;
  options: { value: T; label: React.ReactNode }[];
  onChange: (v: T) => void;
}) {
  return (
    <span className="smv-toggles" role="group" aria-label={label}>
      <span className="smv-label">{label}</span>
      {options.map((o, i) => (
        <span key={String(o.value)}>
          {i > 0 && <span className="smv-sep"> · </span>}
          <Toggle on={value === o.value} onClick={() => onChange(o.value)}>{o.label}</Toggle>
        </span>
      ))}
    </span>
  );
}

// --- (a) the class selection --------------------------------------------------

interface Tally { n: number; conf: number }

export function ClassTree({ data, calls, sel, confidentOnly, colourBy, onChange }: {
  data: NamerData;
  calls: Map<number, Call>;
  sel: Set<string>;
  confidentOnly: boolean;
  colourBy: ColourBy;
  onChange: (sel: Set<string>) => void;
}) {
  const tally = new Map<string, Tally>();
  for (const c of calls.values()) {
    const t = tally.get(leafOf(c)) ?? { n: 0, conf: 0 };
    t.n += 1;
    if (c.confident) t.conf += 1;
    tally.set(leafOf(c), t);
  }
  const mats = manmadeLeaves(data);
  const mm = mats.reduce((a, m) => ({ n: a.n + (tally.get(m)?.n ?? 0), conf: a.conf + (tally.get(m)?.conf ?? 0) }),
                         { n: 0, conf: 0 });
  const nat = tally.get(NATURAL) ?? { n: 0, conf: 0 };
  const top = Math.max(1, mm.n, nat.n);
  const allMm = mats.every((m) => sel.has(m));
  const someMm = mats.some((m) => sel.has(m));
  const shown = mats.filter((m) => (tally.get(m)?.n ?? 0) > 0);

  const row = (k: string, label: string, t: Tally, scale: number, on: boolean, colour: string,
               onClick: () => void, nested = false, partial = false) => (
    <li key={k}>
      <button type="button" className={`smv-class${nested ? " smv-class--nested" : ""}${partial ? " is-partial" : ""}`}
              aria-pressed={on} onClick={onClick}
              title={k === NATURAL ? "Objects called natural" : `${fmt(t.conf)} of ${fmt(t.n)} confident`}>
        <span className="smv-class__label">{label}</span>
        <span className="smv-class__bar">
          <span className="smv-class__all" style={{ width: `${(t.n / scale) * 100}%`, background: colour }} />
          {k !== NATURAL && (
            <span className="smv-class__conf" style={{ width: `${(t.conf / scale) * 100}%`, background: colour }} />
          )}
        </span>
        <span className="smv-class__n">{fmt(confidentOnly && k !== NATURAL ? t.conf : t.n)}</span>
      </button>
    </li>
  );
  return (
    <ul className="smv-classes" aria-label="Classes to show and export">
      {row("manmade", "man-made", mm, top, allMm, NAMER_MANMADE, () => onChange(coarseSel("manmade", data)),
           false, someMm && !allMm)}
      {shown.map((m) => row(m, className(data, m === UNKNOWN ? "" : m), tally.get(m) ?? { n: 0, conf: 0 },
                            Math.max(1, mm.n), sel.has(m),
                            colourBy === "material" && m !== UNKNOWN
                              ? data.classes.get(m)?.colour ?? NAMER_MANMADE : NAMER_MANMADE,
                            () => onChange(pickMaterial(sel, m, data)), true))}
      {row(NATURAL, "natural", nat, top, sel.has(NATURAL), NAMER_NATURAL,
           () => onChange(coarseSel("natural", data)))}
    </ul>
  );
}

// --- (b) the object list, virtualised (PRISMA has thousands of objects) -------------------

const ROW_H = 42;
const OVERSCAN = 8;

export function ObjectList({ objects, data, calls, coords, selected, hovered, onSelect, onHover }: {
  objects: NamerObject[];
  data: NamerData;
  calls: Map<number, Call>;
  coords: Map<number, [number, number]> | null;
  selected: number | null;
  hovered: number | null;
  onSelect: (o: NamerObject) => void;
  onHover: (o: NamerObject | null) => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [top, setTop] = useState(0);
  const [height, setHeight] = useState(400);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // Bring an object picked or hovered on a map into view.
  const target = hovered ?? selected;
  useEffect(() => {
    const el = ref.current;
    const i = objects.findIndex((o) => o.obj_id === target);
    if (!el || i < 0) return;
    if (i * ROW_H < el.scrollTop || (i + 1) * ROW_H > el.scrollTop + el.clientHeight) {
      el.scrollTop = Math.max(0, i * ROW_H - el.clientHeight / 2);
    }
  }, [target, objects]);

  const first = Math.max(0, Math.floor(top / ROW_H) - OVERSCAN);
  const last = Math.min(objects.length, Math.ceil((top + height) / ROW_H) + OVERSCAN);
  return (
    <div ref={ref} className="smv-list" onScroll={(e) => setTop(e.currentTarget.scrollTop)}
         onMouseLeave={() => onHover(null)} role="list" aria-label="Objects">
      <div style={{ height: objects.length * ROW_H, position: "relative" }}>
        {objects.slice(first, last).map((o, j) => {
          const c = calls.get(o.obj_id)!;
          const mm = c.call === "manmade";
          const ll = coords?.get(o.obj_id);
          const conf = callConfidence(o, c);
          return (
            <button key={o.obj_id} type="button" role="listitem"
                    className={`smv-row${selected === o.obj_id ? " is-selected" : ""}${hovered === o.obj_id ? " is-hovered" : ""}`}
                    style={{ top: (first + j) * ROW_H, height: ROW_H }}
                    onClick={() => onSelect(o)} onMouseEnter={() => onHover(o)}
                    aria-pressed={selected === o.obj_id}>
              <span className="smv-row__name">
                <span className={`smv-shape smv-shape--${c.call}`} aria-hidden="true" />
                {mm ? className(data, c.cls) : "natural"}
                {mm && !c.confident && <span className="smv-quiet"> tentative</span>}
                {c.moved && <span className="smv-quiet"> · at this sensitivity</span>}
              </span>
              <span className="smv-row__size">{fmt(o.n_px)} px</span>
              <span className="smv-conf" title={`Sure of the call: ${Math.round(conf * 100)}%`}>
                <span className="smv-conf__fill"
                      style={{ width: `${Math.round(conf * 100)}%`, background: mm ? NAMER_MANMADE : NAMER_NATURAL }} />
              </span>
              <span className="smv-row__where">
                {ll ? `${ll[0].toFixed(4)}, ${ll[1].toFixed(4)}` : `row ${Math.round(o.row)}, col ${Math.round(o.col)}`}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// --- (c) the selected object -------------------------------------------------

export function ObjectCard({ obj, call, cut, data, coords, matches, fam, spectra }: {
  obj: NamerObject;
  call: Call;
  cut: number;
  data: NamerData;
  coords: Map<number, [number, number]> | null;
  matches: Map<number, MatchRow[]> | null;
  fam: FamilyData | null;
  spectra: PixelSpectra | null;
}) {
  const mm = call.call === "manmade";
  const px = data.pixelsByObj.get(obj.obj_id) ?? [];
  const sam = objectSamTop(px, matches);
  const family = objectFamily(px, fam);
  const ll = coords?.get(obj.obj_id);

  // Object mean against the nearest family-library spectrum WITHIN the object's named class,
  // so the plot never contradicts "Closest material" (the class comes from class-pooled namer
  // scores; the nearest entry over all classes can sit in another class). No entry of that
  // class in this action's library, or a natural object: the mean alone.
  let chart: { wavelengths: number[]; lines: Line[] } | null = null;
  if (spectra) {
    const mean = meanSpectrum(spectra, px);
    const famIx = mm && call.cls && fam && fam.lib.bands === spectra.bands
      ? nearestInClass(mean, fam, call.cls) : null;
    const ref = famIx !== null && fam ? libRow(fam.lib, famIx) : null;
    const order = bandOrder(spectra.wavelengths);
    const { a, b } = normalisedPair(mean, ref, order);
    if (a.some(Number.isFinite)) {
      chart = {
        wavelengths: order.map((i) => spectra.wavelengths[i]),
        lines: [
          ...(b && famIx !== null && fam ? [{ label: fam.libLabels[famIx], values: b, muted: true }] : []),
          { label: "this object", values: a },
        ],
      };
    }
  }

  return (
    <section className="smv-card" aria-label="Selected object">
      <h4 className="smv-card__head">
        Object {obj.obj_id}: <span className={`smv-call smv-call--${call.call}`}>{mm ? "man-made" : "natural"}</span>
      </h4>
      {mm && (
        <p className="smv-card__line">
          Closest material <strong>{className(data, call.cls)}</strong>
          {!call.confident && <span className="smv-quiet">, tentative</span>}
        </p>
      )}
      {call.moved && (
        <p className="smv-card__line smv-quiet">
          The namer called it natural; at this sensitivity (man-made above {cut}) it is man-made.
        </p>
      )}
      <p className="smv-card__line smv-quiet">
        {fmt(obj.n_px)} px ·{" "}
        {ll ? `${ll[0].toFixed(5)}, ${ll[1].toFixed(5)}` : `row ${Math.round(obj.row)}, col ${Math.round(obj.col)}`}
      </p>
      {chart && (
        <SpectrumPlot wavelengths={chart.wavelengths} lines={chart.lines}
                      caption="Spectral shape (brightness removed), the object's mean pixel against the closest library spectrum of its class." />
      )}
      <details className="smv-details">
        <summary>Details</summary>
        <dl className="smv-dl">
          <dt>P(man-made)</dt><dd>{obj.p_manmade.toFixed(2)} (man-made above {cut})</dd>
          {mm && call.cls && <><dt>P(class)</dt><dd>{call.pCls.toFixed(2)} (confident at {data.confidentP})</dd></>}
          <dt>Centroid</dt><dd>row {obj.row.toFixed(1)}, col {obj.col.toFixed(1)}</dd>
          <dt>Fill</dt><dd>{obj.fill.toFixed(2)}</dd>
        </dl>
        {mm && sam.length > 0 && (
          <>
            <p className="smv-details__h">USGS library, best match per pixel (SAM)</p>
            <ol className="smv-details__list">
              {sam.slice(0, 3).map((t) => (
                <li key={t.libraryIx}>{t.name} · {t.n} px · {t.angle.toFixed(1)}°</li>
              ))}
            </ol>
          </>
        )}
        {mm && family.verdicts.length > 0 && (
          <>
            <p className="smv-details__h">Material family of its pixels</p>
            <ul className="smv-details__list">
              {family.verdicts.map(([v, n]) => <li key={v}>{v} · {n} px</li>)}
            </ul>
          </>
        )}
        {obj.call === "natural" && (
          <p className="smv-quiet">The worker matched only objects it called man-made; this one was not matched against the libraries.</p>
        )}
      </details>
    </section>
  );
}

// --- older actions: one pixel's SAM / family result ----------------------------------------

export function PixelCard({ px, matches, fam, spectra, samLib }: {
  px: { row: number; col: number };
  matches: Map<number, MatchRow[]> | null;
  fam: FamilyData | null;
  spectra: PixelSpectra | null;
  samLib: Spectra2D | null;
}) {
  const k = key(px.row, px.col);
  const ranks = matches?.get(k) ?? [];
  const f: FamilyRow | undefined = fam?.byKey.get(k);
  const headline = f ? familyHeadline(f) : ranks[0] ? `Closest library spectrum: ${ranks[0].name}` : "No match here";

  let chart: { wavelengths: number[]; lines: Line[] } | null = null;
  if (spectra && spectra.rowOf.has(k)) {
    const mine = meanSpectrum(spectra, [px]);
    const famIx = f?.scored && fam ? fam.libIx.get(f.best_entry) : undefined;
    const ref = famIx !== undefined && fam && fam.lib.bands === spectra.bands ? libRow(fam.lib, famIx)
      : ranks[0] && samLib && samLib.bands === spectra.bands ? libRow(samLib, ranks[0].library_ix) : null;
    const refLabel = famIx !== undefined && fam ? fam.libLabels[famIx] : ranks[0]?.name ?? "";
    const order = bandOrder(spectra.wavelengths);
    const { a, b } = normalisedPair(mine, ref, order);
    chart = {
      wavelengths: order.map((i) => spectra.wavelengths[i]),
      lines: [...(b ? [{ label: refLabel, values: b, muted: true }] : []), { label: "this pixel", values: a }],
    };
  }
  return (
    <section className="smv-card" aria-label="Selected pixel">
      <h4 className="smv-card__head">{headline}</h4>
      <p className="smv-card__line smv-quiet">row {px.row}, col {px.col}</p>
      {chart && (
        <SpectrumPlot wavelengths={chart.wavelengths} lines={chart.lines}
                      caption="Spectral shape (brightness removed) against the closest library spectrum." />
      )}
      <details className="smv-details">
        <summary>Details</summary>
        {f && fam && (
          <p className="smv-quiet">
            family {prettyFamily(f.family)} · confidence {f.confidence.toFixed(4)} · bar p* {fam.pStar.toFixed(4)}
          </p>
        )}
        {ranks.length > 0 && (
          <ol className="smv-details__list">
            {ranks.map((m) => <li key={m.rank}>{m.name} · {m.angle_deg.toFixed(2)}° · {m.chapter}</li>)}
          </ol>
        )}
      </details>
    </section>
  );
}

// The family result in the family LLD's words: a name only when it cleared p*.
function familyHeadline(f: FamilyRow): string {
  if (!f.scored) return "Not scored (missing bands)";
  if (f.family_group === "manmade") {
    return f.answered ? `Likely material: ${prettyFamily(f.family)}` : "Man-made, type unconfirmed";
  }
  return `Natural: ${prettyFamily(f.family)}`;
}

// Older actions' summary: family counts when family matching ran, SAM's top materials otherwise.
export function MaterialsList({ fam, famCounts, histogram }: {
  fam: FamilyData | null;
  famCounts: Record<string, number> | undefined;
  histogram: HistogramRow[] | null;
}) {
  const rows: { label: string; n: number; colour?: string }[] = fam
    ? Object.values(fam.legend)
        .map((it) => ({ label: it.label, n: famCounts?.[it.family] ?? 0, colour: it.colour }))
        .filter((r) => r.n > 0)
    : (histogram ?? []).map((h) => ({ label: h.name, n: h.count }));
  rows.sort((a, b) => b.n - a.n);
  if (rows.length === 0) return <p className="smv-quiet">No matches.</p>;
  const max = Math.max(1, ...rows.map((r) => r.n));
  return (
    <ul className="smv-classes">
      {rows.slice(0, 10).map((r) => (
        <li key={r.label} className="smv-class smv-class--static">
          <span className="smv-class__label">{r.label}</span>
          <span className="smv-class__bar">
            <span className="smv-class__all smv-class__all--solid"
                  style={{ width: `${(r.n / max) * 100}%`, background: r.colour ?? "#555" }} />
          </span>
          <span className="smv-class__n">{fmt(r.n)}</span>
        </li>
      ))}
    </ul>
  );
}
