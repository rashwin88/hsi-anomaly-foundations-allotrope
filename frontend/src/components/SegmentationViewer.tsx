// Output viewer for scene_segmentation with method "indradhanu_v2" (Indradhanu V2's
// segmentation head): one map, scene RGB with the class colours on top, per-class toggles
// with swatch and coverage, and the classes left out of keep_mask marked.
//
// Data, all written by backend/allotrope/action_types/_scene_segmentation_run.py:
//   summary.legend      [{key, label, color, code, pixels, pct_of_valid, excluded}]
//   rgb.png             the scene, full resolution unless longer than 4096 px
//   class_codes.bin     uint8 per pixel, row-major, summary.scene_shape: the class code
//                       (summary.class_names), | haze_bit (0x10) for thin cloud, 255 outside valid
//
// The overlay is painted client-side from class_codes.bin, so the toggles need no extra
// files, and is drawn by MatchMap inside its zoomed wrapper - it moves with pan and zoom.
// The index method ("indices") keeps its own viewer in ActionDetailPane.

import { useEffect, useMemo, useState } from "react";

import { type Px, MatchMap } from "./spectralMatch/MatchMap";
import "./SegmentationViewer.css";

interface LegendEntry {
  key: string;
  label: string;
  color: string;
  code: number | null;      // null for thin cloud, which is the haze bit, not a code
  pixels: number;
  pct_of_valid: number;
  excluded: boolean;
}

const INVALID = 255;
const CODE_MASK = 0x0f;

function fileUrl(actionId: string, filename: string): string {
  return `/api/actions/${encodeURIComponent(actionId)}/files/${encodeURIComponent(filename)}`;
}

function hexRgb(hex: string): [number, number, number] {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number];
}

export function isV2Segmentation(summary: Record<string, unknown>): boolean {
  return summary.method === "indradhanu_v2";
}

export function SegmentationViewer({
  actionId,
  summary,
}: {
  actionId: string;
  summary: Record<string, unknown>;
}) {
  const legend = (summary.legend ?? []) as LegendEntry[];
  const shape = (summary.scene_shape ?? [0, 0]) as [number, number];
  const hazeBit = (summary.haze_bit as number | undefined) ?? 0x10;
  const [h, w] = shape;

  // Land is everywhere; it starts hidden so the other classes read.
  const [shown, setShown] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(legend.map((e) => [e.key, e.key !== "land"])),
  );
  const [opacity, setOpacity] = useState(0.6);
  const [codes, setCodes] = useState<Uint8Array | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [hover, setHover] = useState<Px | null>(null);

  useEffect(() => {
    let cancelled = false;
    setCodes(null);
    setErr(null);
    fetch(fileUrl(actionId, "class_codes.bin"), { credentials: "include" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = new Uint8Array(await res.arrayBuffer());
        if (buf.length !== h * w) throw new Error(`expected ${h * w} pixels, got ${buf.length}`);
        if (!cancelled) setCodes(buf);
      })
      .catch((e) => {
        if (!cancelled) setErr((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [actionId, h, w]);

  const byCode = useMemo(() => {
    const m = new Map<number, LegendEntry>();
    for (const e of legend) if (e.code !== null) m.set(e.code, e);
    return m;
  }, [legend]);
  const haze = legend.find((e) => e.key === "thin_cloud");

  const overlay = useMemo(() => {
    if (!codes || h * w === 0) return null;
    const a = Math.round(opacity * 255);
    // Colour per stored byte value, so the per-pixel loop is a lookup.
    const lut: (number[] | null)[] = new Array(256).fill(null);
    for (let v = 0; v < 255; v++) {
      const e = v & hazeBit && haze && shown[haze.key] ? haze : byCode.get(v & CODE_MASK);
      if (e && shown[e.key]) lut[v] = [...hexRgb(e.color), a];
    }
    const img = new ImageData(w, h);
    const d = img.data;
    for (let i = 0; i < codes.length; i++) {
      const c = lut[codes[i]];
      if (c) d.set(c, i * 4);
    }
    return img;
  }, [codes, shown, opacity, byCode, haze, hazeBit, h, w]);

  // rgb.png can be downscaled (longer than 4096 px); class_codes.bin is not.
  const hovered = useMemo(() => {
    if (!hover || !codes || !nat) return null;
    const row = Math.min(h - 1, Math.floor((hover.row * h) / nat.h));
    const col = Math.min(w - 1, Math.floor((hover.col * w) / nat.w));
    const v = codes[row * w + col];
    if (v === INVALID) return { row, col, text: "no data" };
    const e = byCode.get(v & CODE_MASK);
    const parts = [e?.label ?? "?"];
    if (v & hazeBit) parts.push("thin cloud");
    if (e?.excluded) parts.push("excluded");
    return { row, col, text: parts.join(" · ") };
  }, [hover, codes, nat, byCode, hazeBit, h, w]);

  const excludedPct = summary.excluded_pct_of_valid as number | undefined;
  const excludedNames = legend.filter((e) => e.excluded).map((e) => e.label.toLowerCase());
  const fmtPct = (x: number | undefined) => (typeof x === "number" ? `${x.toFixed(1)}%` : "—");
  const secs = summary.runtime_seconds as number | undefined;

  return (
    <div className="action-output-viewer segv">
      <h4>Scene segmentation · {String(summary.model_codename ?? "model")}</h4>
      <div className="segv__body">
        <div className="segv__map">
          <MatchMap
            rgbUrl={fileUrl(actionId, "rgb.png")}
            overlayImage={overlay}
            onDims={(nw, nh) => setNat({ w: nw, h: nh })}
            onHover={setHover}
            readout={hovered ? <><strong>{hovered.text}</strong> · row {hovered.row}, col {hovered.col}</> : null}
            status={err ? `Classes unavailable (${err})` : !codes ? "Loading classes…" : null}
          />
        </div>
        <div className="segv__side">
          <table className="segv__classes">
            <thead>
              <tr>
                <th>Show</th>
                <th>Class</th>
                <th className="num">Cover</th>
              </tr>
            </thead>
            <tbody>
              {legend.map((e) => (
                <tr key={e.key} className={e.excluded ? "segv__row--excluded" : undefined}>
                  <td>
                    <input
                      type="checkbox"
                      checked={!!shown[e.key]}
                      onChange={(ev) => setShown({ ...shown, [e.key]: ev.target.checked })}
                      aria-label={`Show ${e.label}`}
                    />
                  </td>
                  <td>
                    <span className="segv__swatch" style={{ background: e.color }} />
                    {e.label}
                    {e.excluded && <span className="segv__tag">excluded</span>}
                  </td>
                  <td className="num">{fmtPct(e.pct_of_valid)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <label className="segv__opacity">
            <span>Opacity</span>
            <input type="range" min={0.1} max={1} step={0.05} value={opacity}
                   onChange={(ev) => setOpacity(Number(ev.target.value))} />
          </label>
          <dl className="segv__facts">
            <dt>Excluded</dt>
            <dd>{fmtPct(excludedPct)}{excludedNames.length > 0 && ` (${excludedNames.join(", ")})`}</dd>
            <dt>Kept</dt>
            <dd>{fmtPct(summary.kept_pct_of_valid as number | undefined)}</dd>
            <dt>Thin cloud</dt>
            <dd>{summary.thin_cloud_is_cloud ? "counted as cloud" : "kept"}</dd>
            <dt>Runtime</dt>
            <dd>{typeof secs === "number" ? `${secs.toFixed(0)} s` : "—"}{summary.device ? ` on ${String(summary.device)}` : ""}</dd>
          </dl>
          <p className="segv__hint">
            Anomaly scoring uses <code>keep_mask.tif</code>: every valid pixel except the excluded classes.
            Percentages are of valid pixels.
          </p>
        </div>
      </div>
    </div>
  );
}
