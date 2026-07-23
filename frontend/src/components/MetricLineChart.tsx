// Large time-series chart for a single host metric (CPU / RAM / GPU %).
//
// What makes this chart unusual: we draw vertical "marker" lines for
// each Job-start event the parent passes in, so the operator can scrub
// the line and say "that 80% spike at 10:14 was action_run for
// Chakshu". uPlot's draw hook is the right place — runs after the
// line is rendered, has the scale ready, no React re-render cost.

import { useEffect, useRef } from "react";

export interface MarkerEvent {
  t: number; // unix seconds
  label: string;
  /** Optional CSS color override; defaults to the chart's marker tint. */
  color?: string;
}

interface Props {
  title: string;
  /** Unit shown in the y-axis label + hover tooltip. Usually "%". */
  unit?: string;
  /** Stroke color of the main line. */
  color: string;
  /** Optional translucent fill below the line. */
  fillColor?: string;
  /** Unix seconds, parallel to `values`. Must be the same length. */
  timestamps: number[];
  /** Metric values, percent (0–100) when `unit === "%"`. */
  values: number[];
  /** Vertical-line annotations: job starts, deletions, etc. */
  markers: MarkerEvent[];
  /** Height in px. Defaults to 200. */
  height?: number;
  /** Y-axis min/max. Defaults to [0, 100] for percent series. */
  yMin?: number;
  yMax?: number;
  /** Optional explicit x-axis window in unix seconds. When set,
   *  overrides uPlot's default "fit-to-data" behavior — useful for
   *  expanding the visible range to include historical markers. */
  xMin?: number;
  xMax?: number;
}

export function MetricLineChart({
  title,
  unit = "%",
  color,
  fillColor,
  timestamps,
  values,
  markers,
  height = 200,
  yMin = 0,
  yMax = 100,
  xMin,
  xMax,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<{
    destroy: () => void;
    setData: (d: number[][]) => void;
    setSize: (s: { width: number; height: number }) => void;
    setScale: (key: string, range: { min: number; max: number }) => void;
  } | null>(null);
  const markersRef = useRef<MarkerEvent[]>(markers);
  markersRef.current = markers;
  const titleRef = useRef(title);
  titleRef.current = title;
  const unitRef = useRef(unit);
  unitRef.current = unit;
  // Refs hold the latest data so the async build path can snapshot
  // whatever samples arrived before uPlot finished loading.
  const tsRef = useRef<number[]>(timestamps);
  tsRef.current = timestamps;
  const valsRef = useRef<number[]>(values);
  valsRef.current = values;

  // (Re)build the chart once on mount. Subsequent data changes go
  // through plot.setData so we don't tear down and rebuild every tick.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    let cleanup = () => {};

    (async () => {
      const [{ default: uPlot }] = await Promise.all([
        import("uplot"),
        import("uplot/dist/uPlot.min.css"),
      ]);
      if (cancelled) return;

      const widthOf = () => container.clientWidth || 600;

      const opts = {
        width: widthOf(),
        height,
        scales: {
          x: { time: true },
          y: { range: [yMin, yMax] as [number, number] },
        },
        legend: { show: false },
        cursor: {
          points: { show: true, size: 7 },
          drag: { x: false, y: false, setScale: false },
          // Pointer tracks the nearest x; that's our hover tooltip.
        },
        series: [
          { label: "time" },
          {
            label: "value",
            stroke: color,
            width: 1.7,
            fill: fillColor,
            points: { show: false },
            value: (_self: unknown, raw: number | null) =>
              raw === null || raw === undefined
                ? "—"
                : `${raw.toFixed(1)}${unitRef.current}`,
          },
        ],
        axes: [
          {
            stroke: "#9ca3af",
            grid: { stroke: "#e5e7eb" },
            // Compact "10:14" labels — drop date until the window spans a day.
            values: (_self: unknown, ticks: number[]) =>
              ticks.map((t) => {
                const d = new Date(t * 1000);
                return d.toLocaleTimeString(undefined, {
                  hour: "2-digit",
                  minute: "2-digit",
                });
              }),
          },
          {
            stroke: "#9ca3af",
            grid: { stroke: "#e5e7eb" },
            label: unit,
            labelGap: 4,
          },
        ],
        hooks: {
          // Overlay the marker lines after the line plot is drawn. We
          // read from a ref so updating `markers` after the chart is
          // built doesn't require rebuilding it.
          draw: [
            (u: unknown) => {
              const plot = u as {
                ctx: CanvasRenderingContext2D;
                bbox: { left: number; top: number; width: number; height: number };
                valToPos: (v: number, scale: string, can?: boolean) => number;
                scales: Record<string, { min?: number; max?: number }>;
              };
              const list = markersRef.current;
              if (!list.length) return;
              const xMin = plot.scales.x.min ?? 0;
              const xMax = plot.scales.x.max ?? 0;
              if (xMax <= xMin) return;
              const ctx = plot.ctx;
              const top = plot.bbox.top;
              const bottom = top + plot.bbox.height;
              ctx.save();
              for (const m of list) {
                if (m.t < xMin || m.t > xMax) continue;
                const color = m.color ?? "rgba(99, 102, 241, 0.85)";
                const x = plot.valToPos(m.t, "x", true);
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.setLineDash([5, 4]);
                ctx.lineWidth = 1.4;
                ctx.moveTo(x, top);
                ctx.lineTo(x, bottom);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.moveTo(x - 4, top);
                ctx.lineTo(x + 4, top);
                ctx.lineTo(x, top + 6);
                ctx.closePath();
                ctx.fill();
              }
              ctx.restore();
            },
          ],
        },
      };

      // Snapshot the freshest data possible — handles the case where
      // a few ticks arrive while uPlot's lazy import was in flight.
      const data: number[][] = [tsRef.current, valsRef.current];
      const inst = new (uPlot as unknown as {
        new (o: object, d: unknown[], el: HTMLElement): {
          destroy: () => void;
          setData: (d: number[][]) => void;
          setSize: (s: { width: number; height: number }) => void;
          setScale: (key: string, range: { min: number; max: number }) => void;
        };
      })(opts, data, container);
      plotRef.current = inst;

      const onResize = () => inst.setSize({ width: widthOf(), height });
      window.addEventListener("resize", onResize);
      // ResizeObserver catches flex/grid layout changes that don't
      // fire a window resize (e.g. card wrapping to a new row).
      const ro = new ResizeObserver(() => onResize());
      ro.observe(container);
      cleanup = () => {
        window.removeEventListener("resize", onResize);
        ro.disconnect();
        inst.destroy();
        plotRef.current = null;
      };
    })();

    return () => {
      cancelled = true;
      cleanup();
    };
    // We intentionally only build once; data updates flow via setData.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push new data into the existing chart instance every render.
  useEffect(() => {
    const inst = plotRef.current;
    if (!inst) return;
    inst.setData([timestamps, values]);
  }, [timestamps, values]);

  // Drive the x-scale from props when the caller provides an explicit
  // window. Without this uPlot fits to the data, which hides markers
  // older than the earliest sample.
  useEffect(() => {
    const inst = plotRef.current;
    if (!inst) return;
    if (xMin === undefined || xMax === undefined) return;
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMax <= xMin) return;
    inst.setScale("x", { min: xMin, max: xMax });
  }, [xMin, xMax, timestamps, values]);

  const recent = [...markers].slice(-5).reverse();

  return (
    <div className="metric-chart">
      <header className="metric-chart__header">
        <h4 className="metric-chart__title">{title}</h4>
        <span className="small metric-chart__legend">
          <span
            className="metric-chart__legend-swatch"
            style={{ background: color }}
          />
          live · <span className="metric-chart__legend-mark" /> job start
        </span>
      </header>
      <div ref={containerRef} className="metric-chart__canvas" />
      {recent.length > 0 && (
        <ul className="metric-chart__markers">
          {recent.map((m, i) => (
            <li key={`${m.t}-${i}`} className="small">
              <span className="mono">
                {new Date(m.t * 1000).toLocaleTimeString(undefined, {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })}
              </span>{" "}
              {m.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
