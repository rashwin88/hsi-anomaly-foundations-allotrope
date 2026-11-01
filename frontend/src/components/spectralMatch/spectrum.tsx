// The object card's spectrum chart, drawn in plain svg: no box, no gridlines, one thin
// wavelength axis with a few ticks, and each line named at its right end instead of a legend.
//
// Both lines are L2-normalised shapes (data.ts normalisedPair), so the y scale carries no unit
// and gets no axis; the caption says what is plotted. Missing bands (NaN) break a line rather
// than being bridged, so atmospheric windows read as gaps.

const W = 360;
const H = 150;
const PAD = { l: 6, r: 112, t: 8, b: 22 };
const TICKS = [500, 1000, 1500, 2000, 2500];
const LABEL_GAP = 11;              // px between the two end labels when they would collide

export interface Line { label: string; values: number[]; muted?: boolean }

export function SpectrumPlot({ wavelengths, lines, caption }: {
  wavelengths: number[];
  lines: Line[];
  caption: string;
}) {
  const xs = wavelengths;
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const all = lines.flatMap((l) => l.values.filter(Number.isFinite));
  if (!all.length || !(x1 > x0)) return null;
  const y0 = Math.min(...all);
  const y1 = Math.max(...all);
  const sx = (w: number) => PAD.l + ((w - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => PAD.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - PAD.t - PAD.b);

  const paths = lines.map((l) => {
    let d = "";
    let pen = false;
    l.values.forEach((v, i) => {
      if (!Number.isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${sx(xs[i]).toFixed(1)} ${sy(v).toFixed(1)}`;
      pen = true;
    });
    const lastI = l.values.map((v, i) => (Number.isFinite(v) ? i : -1)).filter((i) => i >= 0).pop();
    return { d, end: lastI === undefined ? null : { x: sx(xs[lastI]), y: sy(l.values[lastI]) } };
  });
  // Push the end labels apart when the lines finish close together.
  const ys = paths.map((p) => p.end?.y ?? 0);
  if (ys.length === 2 && Math.abs(ys[0] - ys[1]) < LABEL_GAP) {
    const mid = (ys[0] + ys[1]) / 2;
    const up = ys[0] <= ys[1] ? 0 : 1;
    ys[up] = mid - LABEL_GAP / 2;
    ys[1 - up] = mid + LABEL_GAP / 2;
  }

  return (
    <figure className="smv-spec">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={caption}>
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b + 4} y2={H - PAD.b + 4} className="smv-spec__axis" />
        {TICKS.filter((t) => t >= x0 && t <= x1).map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={H - PAD.b + 4} y2={H - PAD.b + 7} className="smv-spec__axis" />
            <text x={sx(t)} y={H - 4} textAnchor="middle" className="smv-spec__tick">{t}</text>
          </g>
        ))}
        <text x={W - PAD.r + 4} y={H - 4} className="smv-spec__tick">nm</text>
        {paths.map((p, i) => (
          <g key={lines[i].label}>
            <path d={p.d} fill="none" className={lines[i].muted ? "smv-spec__line--muted" : "smv-spec__line"} />
            {p.end && (
              <text x={p.end.x + 5} y={ys[i] + 3.5}
                    className={lines[i].muted ? "smv-spec__label--muted" : "smv-spec__label"}>
                {lines[i].label.length > 26 ? `${lines[i].label.slice(0, 25)}…` : lines[i].label}
                <title>{lines[i].label}</title>
              </text>
            )}
          </g>
        ))}
      </svg>
      <figcaption className="smv-spec__caption">{caption}</figcaption>
    </figure>
  );
}
