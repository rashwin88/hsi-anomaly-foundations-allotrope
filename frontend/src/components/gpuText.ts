// Plain-text GPU readouts for the top bar and the Monitoring page (class-driven flow, addition E).
//
// /metrics/host reports the api host's nvidia-smi, else the freshest worker snapshot (the GPU is
// given to the worker only). torch_device says where the models actually run, so "GPU present
// but jobs on CPU" is visible. nvidia-smi can print [N/A] for a field: those come back null.

import type { HostMetrics } from "../api/metrics";

type Gpu = HostMetrics["gpu"];
type Device = Gpu["devices"][number];

const num = (v: number | null | undefined, digits = 0) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "n/a";

/** "NVIDIA H200 · util 87% · mem 61234 / 143771 MiB · 54°C" */
export function gpuDeviceLine(d: Device): string {
  return `${d.name} · util ${num(d.utilization_percent)}% · mem ${num(d.memory_used_mb)} / ${num(
    d.memory_total_mb,
  )} MiB · ${num(d.temperature_c)}°C`;
}

/** "Models on GPU (cuda:0 ...)" / "Models on CPU" / "" when no worker has reported. */
export function modelsOnLine(gpu: Gpu): string {
  const dev = gpu.torch_device;
  if (!dev) return "";
  return dev.startsWith("cuda") ? `Models on GPU (${dev})` : `Models on ${dev.toUpperCase()}`;
}

/** Where the numbers come from, e.g. "from worker abc/1234, 3 s ago". */
export function gpuSourceLine(gpu: Gpu): string {
  if (gpu.source === "api") return "nvidia-smi on the api host";
  if (gpu.source === "worker" && gpu.worker_id)
    return `from worker ${gpu.worker_id}, ${num(gpu.age_seconds)} s ago`;
  return "";
}

/** True when a GPU is present but the worker's models run on the CPU. */
export function gpuIdleButPresent(gpu: Gpu): boolean {
  return gpu.available && !!gpu.torch_device && !gpu.torch_device.startsWith("cuda");
}
