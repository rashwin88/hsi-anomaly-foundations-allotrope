// Synthetic CPU / GPU / RAM history for the top-bar sparklines.
//
// Step 4c: no metrics backend yet, so we generate plausible-looking values
// that drift smoothly. When we add `GET /metrics/host` (or a websocket
// stream) later, the swap is hook-only — TopBar's render stays the same.
//
// History length = 60 ticks at 1Hz = last 60 seconds.

import { useEffect, useState } from "react";

const HISTORY_LENGTH = 60;
const TICK_MS = 1000;

export interface MetricSeries {
  cpu: number[];
  gpu: number[];
  ram: number[];
}

function clamp(n: number, min = 0, max = 100): number {
  return Math.max(min, Math.min(max, n));
}

function nextValue(prev: number, drift: number, noise: number): number {
  // drift = slow random walk; noise = high-frequency jitter on top.
  return clamp(
    prev + (Math.random() - 0.5) * drift + (Math.random() - 0.5) * noise,
  );
}

function seedHistory(base: number, jitter: number): number[] {
  return Array.from({ length: HISTORY_LENGTH }, () =>
    clamp(base + (Math.random() - 0.5) * jitter),
  );
}

export function useFakeMetrics(): MetricSeries {
  const [m, setM] = useState<MetricSeries>(() => ({
    cpu: seedHistory(22, 8),
    gpu: seedHistory(8, 6),
    ram: seedHistory(38, 4),
  }));

  useEffect(() => {
    const id = window.setInterval(() => {
      setM((prev) => ({
        cpu: [...prev.cpu.slice(1), nextValue(prev.cpu.at(-1)!, 8, 4)],
        gpu: [...prev.gpu.slice(1), nextValue(prev.gpu.at(-1)!, 14, 6)],
        ram: [...prev.ram.slice(1), nextValue(prev.ram.at(-1)!, 2, 1)],
      }));
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, []);

  return m;
}
