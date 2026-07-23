// uPlot-based charts for action Output viewers.
//
// SpectrumChart   — multi-series line chart over wavelength (nm).
// HistogramChart  — bar chart of bin counts.
//
// Both lazy-import uPlot so the bundle stays light when no Action is
// selected. Pattern matches SceneVisualizations.tsx.

import { useEffect, useRef } from "react";

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
}

export function SpectrumChart({
  wavelengths,
  series,
  height = 220,
  yLabel = "reflectance",
  yMax,
  yMin,
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
          width: 1.5,
          dash: s.dashed ? [4, 4] : undefined,
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
      const opts: object = {
        width,
        height,
        scales: { x: { time: false }, y: yScale },
        // uPlot's built-in legend gets unwieldy with 5+ series; we
        // render a compact custom legend above the chart instead.
        legend: { show: false },
        cursor: { drag: { x: true, y: false }, points: { show: true } },
        series: plotSeries,
        axes: [
          {
            stroke: "#888",
            label: "wavelength (nm)",
            labelSize: 14,
            labelGap: 4,
          },
          {
            stroke: "#888",
            label: yLabel,
            labelSize: 14,
            labelGap: 4,
          },
        ],
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const inst = new (uPlot as any)(opts, data, container);
      dispose = () => inst.destroy();
    })();

    return () => {
      dispose?.();
      while (container.firstChild) container.removeChild(container.firstChild);
    };
  }, [wavelengths, series, height, yLabel, yMax, yMin]);

  return (
    <div className="diagnostic-chart-wrap">
      {series.length > 1 && (
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

      const xs: number[] = [];
      for (let i = 0; i < counts.length; i++) {
        xs.push((edges[i] + edges[i + 1]) / 2);
      }
      const data: (number[])[] = [xs, counts];
      const opts: object = {
        width,
        height,
        scales: { x: { time: false }, y: { auto: true } },
        legend: { show: false },
        cursor: { drag: { x: false, y: false }, points: { show: false } },
        series: [
          {},
          {
            label,
            stroke: color,
            fill: hexToRgba(color, 0.25),
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            paths: (uPlot as any).paths.bars({ size: [0.92, 100], align: 0 }),
          },
        ],
        axes: [{ stroke: "#888" }, { stroke: "#888" }],
        ...(threshold !== undefined
          ? {
              hooks: {
                draw: [
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  (u: any) => {
                    const ctx = u.ctx as CanvasRenderingContext2D;
                    const x = u.valToPos(threshold, "x", true);
                    if (Number.isFinite(x)) {
                      ctx.save();
                      ctx.strokeStyle = "#ef4444";
                      ctx.setLineDash([4, 4]);
                      ctx.lineWidth = 1;
                      ctx.beginPath();
                      ctx.moveTo(x, u.bbox.top);
                      ctx.lineTo(x, u.bbox.top + u.bbox.height);
                      ctx.stroke();
                      ctx.restore();
                    }
                  },
                ],
              },
            }
          : {}),
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
          width: 1.8,
        });
      });
      plotSeries.push({
        label: "chance",
        stroke: "#cbd5e1",
        dash: [4, 4],
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
          {
            stroke: "#888",
            label: "false positive rate",
            labelSize: 14,
            labelGap: 4,
          },
          {
            stroke: "#888",
            label: "true positive rate",
            labelSize: 14,
            labelGap: 4,
          },
        ],
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

  return (
    <div className="diagnostic-chart-wrap">
      <div className="diagnostic-chart-legend">
        {curves.map((c) => (
          <span key={c.label} className="diagnostic-chart-legend__item">
            <span
              className="diagnostic-chart-legend__swatch"
              style={{ background: c.color, borderStyle: "solid" }}
            />
            {c.label} · AUC <strong>{c.auc.toFixed(3)}</strong>
          </span>
        ))}
        <span className="diagnostic-chart-legend__item">
          <span
            className="diagnostic-chart-legend__swatch"
            style={{ background: "#cbd5e1", borderStyle: "dashed" }}
          />
          chance
        </span>
      </div>
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
