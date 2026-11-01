// uPlot-based charts for action Output viewers.
//
// SpectrumChart   — multi-series line chart over wavelength (nm).
// HistogramChart  — bar chart of bin counts.
//
// Both lazy-import uPlot so the bundle stays light when no Action is
// selected. Pattern matches SceneVisualizations.tsx.

import { useEffect, useRef } from "react";

// === Tufte chart chrome ============================================
//
// Every chart here: no grid, thin --ink-2 tick labels in ET Book, a range frame (the axis
// line runs only over the data's extent) and, where it fits, labels written at the end of
// each line instead of a legend. Colours are read from the CSS tokens at build time so the
// charts follow the light / dark theme.

export interface Ink { ink: string; ink2: string; ink3: string; rule: string; select: string }

export function readInk(): Ink {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    ink: v("--ink", "#111"),
    ink2: v("--ink-2", "#555"),
    ink3: v("--ink-3", "#999"),
    rule: v("--rule", "#ddd"),
    select: v("--accent-select", "#d4a017"),
  };
}

const FONT_FAMILY = '"ET Book", Georgia, serif';

export function tufteAxis(c: Ink, label?: string, extra: object = {}): object {
  return {
    stroke: c.ink2,
    font: `12px ${FONT_FAMILY}`,
    labelFont: `italic 13px ${FONT_FAMILY}`,
    grid: { show: false },
    ticks: { show: true, stroke: c.ink3, width: 1, size: 3 },
    border: { show: false },
    space: 56,
    gap: 3,
    ...(label ? { label, labelSize: 18, labelGap: 0 } : {}),
    ...extra,
  };
}

// uPlot draws in device pixels; everything below scales by its pxRatio.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Plot = any;

/** Range frame: x and y axis lines that span only the plotted data. */
function drawRangeFrame(u: Plot, c: Ink, firstSeries: number, lastSeries: number) {
  const xs: number[] = u.data[0];
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let i = firstSeries; i <= lastSeries; i++) {
    if (!u.data[i] || (u.series[i] && u.series[i].show === false)) continue;
    for (const v of u.data[i] as (number | null)[]) {
      if (v === null || !Number.isFinite(v)) continue;
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (!xs.length || !Number.isFinite(yMin)) return;
  yMin = Math.max(yMin, u.scales.y.min ?? yMin);
  yMax = Math.min(yMax, u.scales.y.max ?? yMax);
  const ctx: CanvasRenderingContext2D = u.ctx;
  const { left, top, width, height } = u.bbox;
  const bottom = top + height;
  ctx.save();
  ctx.strokeStyle = c.ink3;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const x0 = Math.max(left, u.valToPos(xs[0], "x", true));
  const x1 = Math.min(left + width, u.valToPos(xs[xs.length - 1], "x", true));
  ctx.moveTo(x0, bottom + 0.5);
  ctx.lineTo(x1, bottom + 0.5);
  ctx.moveTo(left - 0.5, u.valToPos(yMin, "y", true));
  ctx.lineTo(left - 0.5, u.valToPos(yMax, "y", true));
  ctx.stroke();
  ctx.restore();
}

/** Spread label y-positions so no two are closer than `gap` (device px). */
function spread(ys: number[], gap: number, lo: number, hi: number): number[] {
  const order = ys.map((y, i) => [y, i] as [number, number]).sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(ys.length);
  let prev = -Infinity;
  for (const [y, i] of order) {
    const v = Math.max(y, prev + gap, lo);
    out[i] = v;
    prev = v;
  }
  // If the stack ran off the bottom, shift it back up.
  const over = prev - hi;
  if (over > 0) for (let i = 0; i < out.length; i++) out[i] -= over;
  return out;
}

function measureLabel(text: string): number {
  const ctx = document.createElement("canvas").getContext("2d");
  if (!ctx) return text.length * 6;
  ctx.font = `13px ${FONT_FAMILY}`;
  return ctx.measureText(text).width;
}

/** Write each line's label at its right-hand end, in the line's colour. */
function drawEndLabels(u: Plot, labels: { idx: number; text: string; color: string }[]) {
  const pr: number = u.pxRatio ?? window.devicePixelRatio ?? 1;
  const ctx: CanvasRenderingContext2D = u.ctx;
  const { left, top, width, height } = u.bbox;
  const pts: { y: number; text: string; color: string }[] = [];
  for (const l of labels) {
    const vals = u.data[l.idx] as (number | null)[];
    let j = vals.length - 1;
    while (j >= 0 && (vals[j] === null || !Number.isFinite(vals[j] as number))) j--;
    if (j < 0) continue;
    const y = u.valToPos(vals[j], "y", true);
    pts.push({ y: Math.min(Math.max(y, top), top + height), text: l.text, color: l.color });
  }
  const ys = spread(pts.map((p) => p.y), 14 * pr, top + 6 * pr, top + height);
  ctx.save();
  ctx.font = `${13 * pr}px ${FONT_FAMILY}`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";  // uPlot leaves its axis alignment on the context
  pts.forEach((p, i) => {
    ctx.fillStyle = p.color;
    ctx.fillText(p.text, left + width + 6 * pr, ys[i]);
  });
  ctx.restore();
}

interface SpectrumSeries {
  label: string;
  color: string;     // stroke + fill (rgba'd)
  values: number[];  // C floats matching wavelengths.length
  fillBetween?: { lower: number[]; upper: number[] };  // optional band (e.g., p10..p90)
  dashed?: boolean;
}

interface SpectrumChartProps {
  wavelengths: number[];   // C floats (nm)
  series: SpectrumSeries[]; // 1..N lines
  height?: number;
  yLabel?: string;
  /** Hard upper bound on the y-scale. Useful when one noisy series
   * (small-pixel-count class with high SWIR variance) would otherwise
   * dominate auto-scale and crush the others flat. */
  yMax?: number;
  yMin?: number;
  /** Write each series' label at the end of its line instead of a legend above the chart. */
  directLabels?: boolean;
}

export function SpectrumChart({
  wavelengths,
  series,
  height = 220,
  yLabel = "reflectance",
  yMax,
  yMin,
  directLabels = false,
}: SpectrumChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const container = ref.current;
    let dispose: (() => void) | null = null;

    (async () => {
      const [{ default: uPlot }] = await Promise.all([
        import("uplot"),
        import("uplot/dist/uPlot.min.css"),
      ]);
      const width = container.clientWidth || 320;
      const c = readInk();

      // uPlot data shape: [xs, y1, y2, ...]
      // Plus optional band-fill series (lower, upper) per spec.
      const xs = wavelengths;
      const ys: number[][] = series.map((s) => s.values);

      // Insert per-band fill series between existing series so that
      // band shading sits *behind* the corresponding mean line.
      const bandSeries: { idx: number; lo: number[]; hi: number[]; color: string }[] = [];
      for (let i = 0; i < series.length; i++) {
        if (series[i].fillBetween) {
          bandSeries.push({
            idx: i,
            lo: series[i].fillBetween!.lower,
            hi: series[i].fillBetween!.upper,
            color: series[i].color,
          });
        }
      }

      // Build series + data:
      //   slot 0: x (wavelength)
      //   slots 1..n series.length: each line
      //   then per band: lower, upper
      const data: (number[])[] = [xs, ...ys];
      const plotSeries: object[] = [{}];
      series.forEach((s) => {
        plotSeries.push({
          label: s.label,
          stroke: s.color,
          width: 1.25,
          dash: s.dashed ? [3, 3] : undefined,
        });
      });
      bandSeries.forEach((b) => {
        data.push(b.lo, b.hi);
        plotSeries.push({
          label: `${series[b.idx].label} p10`,
          stroke: "transparent",
          show: false,
        });
        plotSeries.push({
          label: `${series[b.idx].label} p90`,
          stroke: "transparent",
          show: false,
        });
      });

      const yScale: object =
        yMax !== undefined || yMin !== undefined
          ? { auto: false, range: [yMin ?? 0, yMax ?? 1] }
          : { auto: true };
      // Room on the right for the end-of-line labels.
      const labelRoom = directLabels
        ? Math.min(160, Math.max(...series.map((s) => measureLabel(s.label))) + 12)
        : 12;
      const opts: object = {
        width,
        height,
        padding: [10, labelRoom, 0, 0],
        scales: { x: { time: false }, y: yScale },
        // uPlot's built-in legend gets unwieldy with 5+ series; we
        // either label line ends directly or render a compact legend.
        legend: { show: false },
        cursor: { drag: { x: true, y: false }, points: { show: true } },
        series: plotSeries,
        axes: [tufteAxis(c, "wavelength (nm)"), tufteAxis(c, yLabel)],
        hooks: {
          draw: [
            (u: Plot) => {
              drawRangeFrame(u, c, 1, series.length);
              if (directLabels) {
                drawEndLabels(u, series.map((s, i) => ({ idx: i + 1, text: s.label, color: s.color })));
              }
            },
          ],
        },
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const inst = new (uPlot as any)(opts, data, container);
      dispose = () => inst.destroy();
    })();

    return () => {
      dispose?.();
      while (container.firstChild) container.removeChild(container.firstChild);
    };
  }, [wavelengths, series, height, yLabel, yMax, yMin, directLabels]);

  return (
    <div className="diagnostic-chart-wrap">
      {series.length > 1 && !directLabels && (
        <div className="diagnostic-chart-legend">
          {series.map((s) => (
            <span key={s.label} className="diagnostic-chart-legend__item">
              <span
                className="diagnostic-chart-legend__swatch"
                style={{
                  background: s.color,
                  borderStyle: s.dashed ? "dashed" : "solid",
                }}
              />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <div ref={ref} className="diagnostic-chart" />
    </div>
  );
}

interface HistogramChartProps {
  edges: number[];   // bins+1
  counts: number[];  // bins
  color?: string;
  height?: number;
  label?: string;
  threshold?: number; // draws a vertical reference line at x=threshold
}

export function HistogramChart({
  edges,
  counts,
  color = "#1f5f3d",
  height = 130,
  label = "count",
  threshold,
}: HistogramChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const container = ref.current;
    let dispose: (() => void) | null = null;

    (async () => {
      const [{ default: uPlot }] = await Promise.all([
        import("uplot"),
        import("uplot/dist/uPlot.min.css"),
      ]);
      const width = container.clientWidth || 300;
      const c = readInk();

      const xs: number[] = [];
      for (let i = 0; i < counts.length; i++) {
        xs.push((edges[i] + edges[i + 1]) / 2);
      }
      const data: (number[])[] = [xs, counts];
      const opts: object = {
        width,
        height,
        padding: [18, 12, 0, 0],
        scales: { x: { time: false }, y: { auto: true } },
        legend: { show: false },
        cursor: { drag: { x: false, y: false }, points: { show: false } },
        series: [
          {},
          {
            label,
            stroke: hexToRgba(color, 0.55),
            width: 0,
            fill: hexToRgba(color, 0.55),
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            paths: (uPlot as any).paths.bars({ size: [0.92, 100], align: 0 }),
          },
        ],
        axes: [tufteAxis(c), tufteAxis(c, undefined, { space: 30 })],
        hooks: {
          draw: [
            (u: Plot) => {
              const ctx = u.ctx as CanvasRenderingContext2D;
              const pr: number = u.pxRatio ?? window.devicePixelRatio ?? 1;
              // Direct title: the quantity, top-left, instead of a legend.
              ctx.save();
              ctx.font = `italic ${13 * pr}px ${FONT_FAMILY}`;
              ctx.fillStyle = c.ink2;
              ctx.textBaseline = "bottom";
              ctx.textAlign = "left";  // uPlot leaves its axis alignment on the context
              ctx.fillText(label, u.bbox.left + 4 * pr, u.bbox.top - 2 * pr);
              ctx.restore();
              if (threshold === undefined) return;
              const x = u.valToPos(threshold, "x", true);
              if (!Number.isFinite(x)) return;
              // The human-set threshold: one thin line in the selection colour, labelled.
              ctx.save();
              ctx.strokeStyle = c.select;
              ctx.lineWidth = pr;
              ctx.beginPath();
              ctx.moveTo(x, u.bbox.top);
              ctx.lineTo(x, u.bbox.top + u.bbox.height);
              ctx.stroke();
              ctx.font = `${12 * pr}px ${FONT_FAMILY}`;
              ctx.fillStyle = c.ink2;
              ctx.textBaseline = "bottom";
              const t = `threshold ${threshold.toPrecision(3)}`;
              const tw = ctx.measureText(t).width;
              const flip = x + 4 * pr + tw > u.bbox.left + u.bbox.width;
              ctx.textAlign = flip ? "right" : "left";
              ctx.fillText(t, flip ? x - 4 * pr : x + 4 * pr, u.bbox.top - 2 * pr);
              ctx.restore();
            },
          ],
        },
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const inst = new (uPlot as any)(opts, data, container);
      dispose = () => inst.destroy();
    })();

    return () => {
      dispose?.();
      while (container.firstChild) container.removeChild(container.firstChild);
    };
  }, [edges, counts, color, height, label, threshold]);

  return <div ref={ref} className="diagnostic-chart" />;
}

function hexToRgba(hex: string, alpha: number): string {
  // Accept either #rgb / #rrggbb / already-rgba()
  if (hex.startsWith("rgba(") || hex.startsWith("rgb(")) return hex;
  let h = hex.replace("#", "");
  if (h.length === 3) {
    h = h.split("").map((c) => c + c).join("");
  }
  const num = Number.parseInt(h, 16);
  const r = (num >> 16) & 0xff;
  const g = (num >> 8) & 0xff;
  const b = num & 0xff;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// === ROC curves =====================================================
//
// One curve per model. X = false positive rate, Y = true positive rate.
// Diagonal reference line for random-classifier baseline.

interface ROCCurve {
  label: string;
  color: string;
  fpr: number[];
  tpr: number[];
  auc: number;
}

interface ROCChartProps {
  curves: ROCCurve[];
  height?: number;
}

export function ROCChart({ curves, height = 280 }: ROCChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current || curves.length === 0) return;
    const container = ref.current;
    let dispose: (() => void) | null = null;

    (async () => {
      const [{ default: uPlot }] = await Promise.all([
        import("uplot"),
        import("uplot/dist/uPlot.min.css"),
      ]);
      const width = container.clientWidth || 320;
      const ink = readInk();

      // Resample each curve onto a shared 0..1 FPR grid so uPlot can
      // share the X axis cleanly. 101 points = 0, 0.01, ..., 1.0.
      const xs: number[] = Array.from({ length: 101 }, (_, i) => i / 100);
      const seriesData: number[][] = curves.map((c) => resampleROC(c.fpr, c.tpr, xs));
      // Diagonal reference (random-classifier baseline).
      const chance = xs.slice();
      const data: number[][] = [xs, ...seriesData, chance];

      const plotSeries: object[] = [{}];
      curves.forEach((c) => {
        plotSeries.push({
          label: `${c.label} · AUC=${c.auc.toFixed(3)}`,
          stroke: c.color,
          width: 1.5,
        });
      });
      plotSeries.push({
        label: "chance",
        stroke: ink.rule,
        width: 1,
      });

      const opts: object = {
        width,
        height,
        scales: {
          x: { time: false, range: [0, 1] },
          y: { auto: false, range: [0, 1] },
        },
        legend: { show: false },
        cursor: { drag: { x: true, y: false }, points: { show: true } },
        series: plotSeries,
        axes: [
          tufteAxis(ink, "false positive rate", { splits: () => [0, 0.5, 1] }),
          tufteAxis(ink, "true positive rate", { splits: () => [0, 0.5, 1] }),
        ],
        hooks: {
          draw: [
            (u: Plot) => {
              const ctx = u.ctx as CanvasRenderingContext2D;
              const pr: number = u.pxRatio ?? window.devicePixelRatio ?? 1;
              drawRangeFrame(u, ink, 1, curves.length);
              // Labels in the empty lower-right triangle, each in its curve's colour, in
              // place of a legend; "chance" sits on the diagonal it names.
              ctx.save();
              ctx.font = `${13 * pr}px ${FONT_FAMILY}`;
              ctx.textBaseline = "middle";
              ctx.textAlign = "left";  // uPlot leaves its axis alignment on the context
              const x = u.valToPos(0.42, "x", true);
              let y = u.valToPos(0.34, "y", true);
              for (const cv of curves) {
                ctx.fillStyle = cv.color;
                ctx.fillText(`${cv.label}  AUC ${cv.auc.toFixed(3)}`, x, y);
                y += 16 * pr;
              }
              ctx.fillStyle = ink.ink3;
              ctx.font = `italic ${12 * pr}px ${FONT_FAMILY}`;
              ctx.fillText("chance", u.valToPos(0.8, "x", true), u.valToPos(0.72, "y", true));
              ctx.restore();
            },
          ],
        },
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const inst = new (uPlot as any)(opts, data, container);
      dispose = () => inst.destroy();
    })();

    return () => {
      dispose?.();
      while (container.firstChild) container.removeChild(container.firstChild);
    };
  }, [curves, height]);

  // Curves are labelled inside the plot by the draw hook; no legend.
  return (
    <div className="diagnostic-chart-wrap">
      <div ref={ref} className="diagnostic-chart" />
    </div>
  );
}

function resampleROC(fpr: number[], tpr: number[], targetXs: number[]): number[] {
  // ROC curves come in monotonic FPR order. Linear interpolation onto
  // `targetXs`. Anchor endpoints at (0, 0) and (1, 1) if missing.
  const xs = fpr.slice();
  const ys = tpr.slice();
  if (xs.length === 0 || xs[0] > 0) {
    xs.unshift(0);
    ys.unshift(0);
  }
  if (xs[xs.length - 1] < 1) {
    xs.push(1);
    ys.push(1);
  }
  const out: number[] = [];
  let j = 0;
  for (const tx of targetXs) {
    while (j + 1 < xs.length && xs[j + 1] < tx) j++;
    const x0 = xs[j];
    const x1 = xs[Math.min(j + 1, xs.length - 1)];
    const y0 = ys[j];
    const y1 = ys[Math.min(j + 1, ys.length - 1)];
    if (x1 === x0) {
      out.push(y1);
    } else {
      const t = (tx - x0) / (x1 - x0);
      out.push(y0 + t * (y1 - y0));
    }
  }
  return out;
}
