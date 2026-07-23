// Host + workload metrics client (Step 20).

import { fetchJson } from "./client";

export interface HostMetrics {
  timestamp: string;
  cpu: {
    percent: number;
    count_logical: number;
    load_average_1m: number | null;
  };
  memory: {
    total_bytes: number;
    used_bytes: number;
    available_bytes: number;
    percent: number;
  };
  disks: Array<{
    mountpoint: string;
    total_bytes: number;
    used_bytes: number;
    free_bytes: number;
    percent: number;
  }>;
  gpu: {
    available: boolean;
    note: string;
    devices: Array<{
      name: string;
      utilization_percent: number;
      memory_used_mb: number;
      memory_total_mb: number;
      temperature_c: number;
    }>;
  };
}

export interface WorkloadMetrics {
  timestamp: string;
  queue_depth: number;
  by_status: {
    queued: number;
    running: number;
    complete: number;
    failed: number;
    cancelled: number;
  };
  by_type_running: Record<string, number>;
  by_type_queued: Record<string, number>;
  completed_last_minute: number;
  completed_last_10_minutes: number;
  completed_last_hour: number;
  completed_mean_seconds: number | null;
  completed_p95_seconds: number | null;
  oldest_queued_age_seconds: number | null;
}

export async function getHostMetrics(): Promise<HostMetrics> {
  return fetchJson<HostMetrics>("/metrics/host");
}

export async function getWorkloadMetrics(): Promise<WorkloadMetrics> {
  return fetchJson<WorkloadMetrics>("/metrics/workload");
}
