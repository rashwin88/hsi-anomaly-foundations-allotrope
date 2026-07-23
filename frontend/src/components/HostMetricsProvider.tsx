// App-level host-metrics provider.
//
// Started once at the root of the app (App.tsx wraps its children with
// this) and runs for the full session — leaves a single polling loop
// running so the user can navigate away from /monitoring and come
// back to find the history intact. MonitoringPage subscribes via
// `useHostMetrics()` and renders against the buffered series.
//
// Buffer policy: keep ~1 hour of host metrics at 2 s cadence (= 1800
// samples). For job-start markers we accumulate everything we see —
// each job_id is only counted once, so the marker list grows
// monotonically with the session. This matches Monitoring's "All"
// range expanding to cover every historical marker.

import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

import { listJobs } from "../api/jobs";
import {
  getHostMetrics,
  getWorkloadMetrics,
  type HostMetrics,
  type WorkloadMetrics,
} from "../api/metrics";
import type { MarkerEvent } from "./MetricLineChart";

export const HOST_TICK_MS = 2000;
export const WORKLOAD_TICK_MS = 5000;
// 1 hour of context at 2 s cadence. Pages choose their visible
// window via the range picker; we just buffer the full hour.
export const BIG_HISTORY = 1800;
// Short-history sparklines used inside the summary cards.
export const HISTORY = 60;

// Marker palette per job type. Matches what MonitoringPage uses today.
const JOB_TYPE_COLOR: Record<string, string> = {
  action_run: "rgba(124, 58, 237, 0.78)",
  scene_onboard: "rgba(37, 99, 235, 0.78)",
  annotation_attach: "rgba(180, 83, 9, 0.78)",
  project_export: "rgba(13, 148, 136, 0.78)",
};
function jobMarkerColor(type: string): string {
  return JOB_TYPE_COLOR[type] ?? "rgba(99, 102, 241, 0.7)";
}

export interface HostMetricsState {
  host: HostMetrics | null;
  workload: WorkloadMetrics | null;
  // Big-chart history — parallel arrays. Rebuilt as a new array
  // reference each tick so chart consumers re-render correctly.
  tsBig: number[];
  cpuBig: number[];
  ramBig: number[];
  gpuBig: number[];
  // Short-history sparklines.
  cpuH: number[];
  ramH: number[];
  gpuH: number[];
  throughputH: number[];
  // Job-start markers accumulated across the session.
  markers: MarkerEvent[];
  error: string | null;
  // The moment the provider mounted — useful for chart x-axis
  // "since session start" defaults.
  sessionStartSec: number;
}

const initialState: HostMetricsState = {
  host: null,
  workload: null,
  tsBig: [],
  cpuBig: [],
  ramBig: [],
  gpuBig: [],
  cpuH: Array(HISTORY).fill(0),
  ramH: Array(HISTORY).fill(0),
  gpuH: Array(HISTORY).fill(0),
  throughputH: Array(HISTORY).fill(0),
  markers: [],
  error: null,
  sessionStartSec: Date.now() / 1000,
};

const HostMetricsCtx = createContext<HostMetricsState>(initialState);

export function useHostMetrics(): HostMetricsState {
  return useContext(HostMetricsCtx);
}

export function HostMetricsProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<HostMetricsState>(() => ({
    ...initialState,
    sessionStartSec: Date.now() / 1000,
  }));

  // De-dup seen job_ids across the session so re-polling /jobs doesn't
  // re-emit markers for jobs we've already recorded.
  const seenStartsRef = useRef<Set<string>>(new Set());

  // Host tick — runs forever once the provider mounts. Each iteration
  // appends one sample to the big buffers and shifts the short
  // sparkline rolling windows by one.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const snap = await getHostMetrics();
        if (cancelled) return;
        const tSec = new Date(snap.timestamp).getTime() / 1000;
        const gpuPct =
          snap.gpu.available && snap.gpu.devices.length > 0
            ? snap.gpu.devices[0].utilization_percent
            : 0;
        setState((prev) => {
          const cap = (arr: number[], v: number) => {
            const next =
              arr.length >= BIG_HISTORY ? arr.slice(1) : arr.slice();
            next.push(v);
            return next;
          };
          const shortRoll = (arr: number[], v: number) => [...arr.slice(1), v];
          return {
            ...prev,
            host: snap,
            tsBig: cap(prev.tsBig, tSec),
            cpuBig: cap(prev.cpuBig, snap.cpu.percent),
            ramBig: cap(prev.ramBig, snap.memory.percent),
            gpuBig: cap(prev.gpuBig, gpuPct),
            cpuH: shortRoll(prev.cpuH, snap.cpu.percent),
            ramH: shortRoll(prev.ramH, snap.memory.percent),
            gpuH: shortRoll(prev.gpuH, gpuPct),
            error: null,
          };
        });
      } catch (err) {
        if (!cancelled) {
          setState((prev) => ({ ...prev, error: (err as Error).message }));
        }
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), HOST_TICK_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // Workload tick — drives the small summary cards + the throughput
  // sparkline + job-start marker collection.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const [wl, jobs] = await Promise.all([
          getWorkloadMetrics(),
          listJobs({ limit: 100 }),
        ]);
        if (cancelled) return;
        const newMarkers: MarkerEvent[] = [];
        for (const j of jobs.items) {
          if (!j.started_at) continue;
          if (seenStartsRef.current.has(j.id)) continue;
          seenStartsRef.current.add(j.id);
          newMarkers.push({
            t: new Date(j.started_at).getTime() / 1000,
            label: `${j.type} · ${j.id.slice("job_".length, "job_".length + 8)}…`,
            color: jobMarkerColor(j.type),
          });
        }
        setState((prev) => ({
          ...prev,
          workload: wl,
          throughputH: [...prev.throughputH.slice(1), wl.completed_last_minute],
          markers:
            newMarkers.length > 0
              ? [...prev.markers, ...newMarkers]
              : prev.markers,
        }));
      } catch {
        /* swallow — host error already surfaces */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), WORKLOAD_TICK_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return (
    <HostMetricsCtx.Provider value={state}>{children}</HostMetricsCtx.Provider>
  );
}
