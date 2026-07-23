// Monitoring destination (Step 20).
//
// Two stacked sections: host (CPU / RAM / disk / GPU) and workload
// (queue depth, by-status / by-type breakdowns, rolling throughput,
// latency mean/p95).
//
// Metric collection is owned by `HostMetricsProvider` mounted in
// `App.tsx` — that means polling starts at login and the buffered
// history survives navigating between pages. This component reads the
// provider's state and renders.

import { useState } from "react";

import {
  HOST_TICK_MS as _HOST_TICK_MS,
  useHostMetrics,
  WORKLOAD_TICK_MS as _WORKLOAD_TICK_MS,
} from "../components/HostMetricsProvider";
import { MetricLineChart } from "../components/MetricLineChart";
import { Sparkline } from "../components/Sparkline";

type RangeKey = "1m" | "10m" | "1h" | "all";
const RANGE_SECONDS: Record<RangeKey, number | null> = {
  "1m": 60,
  "10m": 600,
  "1h": 3600,
  all: null,
};
const RANGE_OPTIONS: Array<{ key: RangeKey; label: string }> = [
  { key: "1m", label: "1m" },
  { key: "10m", label: "10m" },
  { key: "1h", label: "1h" },
  { key: "all", label: "All" },
];

function bytes(n: number): string {
  if (n >= 1024 ** 4) return `${(n / 1024 ** 4).toFixed(2)} TiB`;
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KiB`;
  return `${n} B`;
}

function pct(n: number): string {
  return `${n.toFixed(1)}%`;
}

function secs(n: number | null): string {
  if (n === null || n === undefined) return "—";
  if (n < 60) return `${n.toFixed(1)}s`;
  const m = Math.floor(n / 60);
  const r = n - m * 60;
  return `${m}m ${r.toFixed(0)}s`;
}

export function MonitoringPage() {
  // Pull everything the page needs from the app-level provider; the
  // provider owns all polling, so navigating away and back is a no-op
  // for the history buffers. `error` surfaces transport problems.
  const {
    host,
    workload,
    tsBig,
    cpuBig,
    ramBig,
    gpuBig,
    cpuH,
    ramH,
    gpuH,
    throughputH,
    markers,
    error,
  } = useHostMetrics();

  // Range picker — controls the x-axis window of all three charts.
  // "all" expands to cover every sample + every marker, so historical
  // job starts stay visible across long sessions.
  const [range, setRange] = useState<RangeKey>("10m");

  // Resolve the visible x-window. The default rolling-window modes
  // clip to the chosen seconds-back-from-now; "all" expands to include
  // every marker and every sample so nothing stays hidden.
  const nowSec = Date.now() / 1000;
  const earliestSample = tsBig[0];
  const latestSample = tsBig[tsBig.length - 1] ?? nowSec;
  const earliestMarker = markers.length > 0 ? markers[0].t : undefined;
  let xMin: number | undefined;
  let xMax: number | undefined;
  if (tsBig.length > 0) {
    if (range === "all") {
      xMin = Math.min(earliestSample ?? nowSec, earliestMarker ?? nowSec);
      xMax = latestSample;
      // Pad ~5% on each side so edge markers don't clip against the axis.
      const span = Math.max(60, xMax - xMin);
      xMin -= span * 0.03;
      xMax += span * 0.03;
    } else {
      const seconds = RANGE_SECONDS[range] ?? 600;
      xMax = latestSample;
      xMin = xMax - seconds;
    }
  }

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Monitoring</h1>
          <p className="page__subtitle">
            Real-time host and workload metrics. Top-bar sparklines pull
            from the same endpoint at 1 Hz; this page polls at 2 / 5 s
            for the deeper breakdown.
          </p>
        </div>
      </div>

      {error && (
        <div className="page__error" role="alert">
          metrics fetch failed: {error}
        </div>
      )}

      {/* ─── Big time-series charts ─────────────────────────────── */}
      <section className="metrics-section">
        <header className="metrics-section__header">
          <h2 className="metrics-section__title">
            Host ·{" "}
            {range === "all"
              ? "all observed"
              : range === "1h"
                ? "last hour"
                : range === "1m"
                  ? "last minute"
                  : "last 10 min"}
          </h2>
          <div className="metrics-section__header-right">
            <div
              className="metrics-range"
              role="group"
              aria-label="Time range"
            >
              {RANGE_OPTIONS.map((o) => (
                <button
                  key={o.key}
                  type="button"
                  className="metrics-range__btn"
                  data-active={range === o.key ? "true" : "false"}
                  onClick={() => setRange(o.key)}
                >
                  {o.label}
                </button>
              ))}
            </div>
            <span className="small metric-chart__legend">
              Dashed vertical lines mark job starts — colored by job type.
            </span>
          </div>
        </header>
        <div className="metrics-chart-stack">
          <MetricLineChart
            title="CPU utilisation"
            unit="%"
            color="#2563eb"
            fillColor="rgba(37, 99, 235, 0.10)"
            timestamps={tsBig}
            values={cpuBig}
            markers={markers}
            xMin={xMin}
            xMax={xMax}
          />
          <MetricLineChart
            title="RAM usage"
            unit="%"
            color="#0d9488"
            fillColor="rgba(13, 148, 136, 0.10)"
            timestamps={tsBig}
            values={ramBig}
            markers={markers}
            xMin={xMin}
            xMax={xMax}
          />
          {host?.gpu.available ? (
            <MetricLineChart
              title={`GPU · ${host.gpu.devices[0]?.name ?? "device 0"}`}
              unit="%"
              color="#1f5f3d"
              fillColor="rgba(31, 95, 61, 0.10)"
              timestamps={tsBig}
              values={gpuBig}
              markers={markers}
              xMin={xMin}
              xMax={xMax}
            />
          ) : (
            <div className="metric-chart metric-chart--inert">
              <header className="metric-chart__header">
                <h4 className="metric-chart__title">GPU utilisation</h4>
              </header>
              <p className="small metric-chart__placeholder">
                {host?.gpu.note ??
                  "GPU not available — Mac Docker has no MPS/CUDA passthrough; worker runs on CPU."}
              </p>
            </div>
          )}
        </div>
        {tsBig.length < 3 && (
          <p className="small metric-chart__warmup">
            Charts populate as samples arrive — first few ticks needed.
          </p>
        )}
        {markers.length === 0 && tsBig.length >= 3 && (
          <p className="small metric-chart__hint">
            Run an Action or onboard a Scene to see job-start markers
            appear on the charts in real time.
          </p>
        )}
      </section>

      {/* ─── Summary cards (smaller sparklines) ──────────────────── */}
      <section className="metrics-section">
        <header className="metrics-section__header">
          <h2 className="metrics-section__title">Host</h2>
          {host && (
            <span className="small">
              {host.cpu.count_logical} logical cores
              {host.cpu.load_average_1m !== null && (
                <>
                  {" "}
                  · load{" "}
                  <span className="mono">
                    {host.cpu.load_average_1m.toFixed(2)}
                  </span>
                </>
              )}
            </span>
          )}
        </header>
        <div className="metrics-section__grid">
          <div className="metrics-card">
            <h4>CPU</h4>
            <Sparkline label="CPU" values={cpuH} current={cpuH.at(-1) ?? 0} />
            <p className="mono">{host ? pct(host.cpu.percent) : "—"}</p>
          </div>
          <div className="metrics-card">
            <h4>RAM</h4>
            <Sparkline label="RAM" values={ramH} current={ramH.at(-1) ?? 0} />
            <p className="mono">
              {host ? `${bytes(host.memory.used_bytes)} / ${bytes(host.memory.total_bytes)} · ${pct(host.memory.percent)}` : "—"}
            </p>
          </div>
          <div className="metrics-card">
            <h4>GPU</h4>
            {host?.gpu.available && host.gpu.devices.length > 0 ? (
              <>
                <Sparkline label="GPU" values={gpuH} current={gpuH.at(-1) ?? 0} />
                <p className="mono">
                  {host.gpu.devices[0].name} · util{" "}
                  {host.gpu.devices[0].utilization_percent.toFixed(0)}% · mem{" "}
                  {host.gpu.devices[0].memory_used_mb.toFixed(0)} /{" "}
                  {host.gpu.devices[0].memory_total_mb.toFixed(0)} MiB ·{" "}
                  {host.gpu.devices[0].temperature_c.toFixed(0)}°C
                </p>
              </>
            ) : (
              <p className="small">
                {host?.gpu.note ?? "GPU not available — worker runs on CPU."}
              </p>
            )}
          </div>
        </div>
        {host && host.disks.length > 0 && (
          <div className="metrics-disks">
            <h4>Volumes</h4>
            <table className="jobs-table">
              <thead>
                <tr>
                  <th>Mount</th>
                  <th>Used</th>
                  <th>Free</th>
                  <th>Total</th>
                  <th>% used</th>
                </tr>
              </thead>
              <tbody>
                {host.disks.map((d) => (
                  <tr key={d.mountpoint}>
                    <td className="mono">{d.mountpoint}</td>
                    <td className="mono">{bytes(d.used_bytes)}</td>
                    <td className="mono">{bytes(d.free_bytes)}</td>
                    <td className="mono">{bytes(d.total_bytes)}</td>
                    <td className="mono">{pct(d.percent)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="metrics-section">
        <header className="metrics-section__header">
          <h2 className="metrics-section__title">Workload</h2>
          {workload && (
            <span className="small">
              queue depth{" "}
              <span className="mono">{workload.queue_depth}</span> · oldest
              queued{" "}
              <span className="mono">{secs(workload.oldest_queued_age_seconds)}</span>
            </span>
          )}
        </header>
        <div className="metrics-section__grid">
          <div className="metrics-card">
            <h4>Status</h4>
            {workload ? (
              <ul className="metrics-counts">
                <li><span>queued</span><span className="mono">{workload.by_status.queued}</span></li>
                <li><span>running</span><span className="mono">{workload.by_status.running}</span></li>
                <li><span>complete</span><span className="mono">{workload.by_status.complete}</span></li>
                <li><span>failed</span><span className="mono">{workload.by_status.failed}</span></li>
                <li><span>cancelled</span><span className="mono">{workload.by_status.cancelled}</span></li>
              </ul>
            ) : (
              <p className="small">Loading…</p>
            )}
          </div>
          <div className="metrics-card">
            <h4>Throughput</h4>
            <Sparkline
              label="jobs / min"
              values={throughputH}
              current={throughputH.at(-1) ?? 0}
            />
            <p className="mono small">
              last 1m{" "}
              <strong>{workload?.completed_last_minute ?? "—"}</strong> · 10m{" "}
              <strong>{workload?.completed_last_10_minutes ?? "—"}</strong> · 1h{" "}
              <strong>{workload?.completed_last_hour ?? "—"}</strong>
            </p>
          </div>
          <div className="metrics-card">
            <h4>Latency (last hour)</h4>
            <p className="mono">
              mean {secs(workload?.completed_mean_seconds ?? null)} · p95{" "}
              {secs(workload?.completed_p95_seconds ?? null)}
            </p>
            <p className="small">
              Computed from <code>completed_at − started_at</code> on rows
              that finished in the past hour.
            </p>
          </div>
        </div>
        {workload && (
          <div className="metrics-section__by-type">
            <h4>By type</h4>
            <table className="jobs-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Queued</th>
                  <th>Running</th>
                </tr>
              </thead>
              <tbody>
                {Array.from(
                  new Set([
                    ...Object.keys(workload.by_type_queued),
                    ...Object.keys(workload.by_type_running),
                  ]),
                )
                  .sort()
                  .map((t) => (
                    <tr key={t}>
                      <td className="mono small">{t}</td>
                      <td className="mono">{workload.by_type_queued[t] ?? 0}</td>
                      <td className="mono">{workload.by_type_running[t] ?? 0}</td>
                    </tr>
                  ))}
                {Object.keys(workload.by_type_queued).length === 0 &&
                  Object.keys(workload.by_type_running).length === 0 && (
                    <tr>
                      <td colSpan={3} className="small">
                        Nothing queued or running right now.
                      </td>
                    </tr>
                  )}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
