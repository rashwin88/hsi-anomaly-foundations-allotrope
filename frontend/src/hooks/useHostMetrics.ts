// Real host-metrics polling hook (Step 20).
//
// Replaces useFakeMetrics — keeps the same MetricSeries shape so
// TopBar's render layer doesn't change. We poll once per second and
// roll the last 60 values into history arrays for the sparklines.

import { useEffect, useRef, useState } from "react";

import { getHostMetrics, type HostMetrics } from "../api/metrics";

const HISTORY_LENGTH = 60;
const TICK_MS = 1000;

export interface MetricSeries {
  cpu: number[];
  gpu: number[];
  ram: number[];
  // The latest snapshot — exposed so the topbar can show whether GPU
  // is genuinely unavailable vs "0%" with hardware present.
  latest: HostMetrics | null;
  error: string | null;
}

function emptyHistory(): number[] {
  return Array(HISTORY_LENGTH).fill(0);
}

export function useHostMetrics(): MetricSeries {
  const [series, setSeries] = useState<MetricSeries>({
    cpu: emptyHistory(),
    gpu: emptyHistory(),
    ram: emptyHistory(),
    latest: null,
    error: null,
  });
  // Used to drop in-flight responses on unmount; also gates against
  // overlapping ticks if the network is slow.
  const inFlight = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (inFlight.current) return;
      inFlight.current = true;
      try {
        const snap = await getHostMetrics();
        if (cancelled) return;
        setSeries((prev) => {
          const gpuPct =
            snap.gpu.available && snap.gpu.devices.length > 0
              ? snap.gpu.devices[0].utilization_percent
              : 0;
          return {
            cpu: [...prev.cpu.slice(1), snap.cpu.percent],
            gpu: [...prev.gpu.slice(1), gpuPct],
            ram: [...prev.ram.slice(1), snap.memory.percent],
            latest: snap,
            error: null,
          };
        });
      } catch (err) {
        if (cancelled) return;
        setSeries((prev) => ({
          ...prev,
          error: (err as Error).message ?? "metrics_unavailable",
        }));
      } finally {
        inFlight.current = false;
      }
    };
    void tick(); // first read immediately
    const id = window.setInterval(() => void tick(), TICK_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return series;
}
