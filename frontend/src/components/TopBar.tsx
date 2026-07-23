// Top bar (storyboard-spec § 4).
//
// Left:  active project chip — "no project" placeholder until we have a
//        Project context (Step 6+).
// Right: GPU pill (from /metrics/host) → CPU / RAM sparklines → UserMenu.
//
// Step 20: useHostMetrics polls /metrics/host once per second and rolls
// a 60-tick history. GPU availability is now genuine — populated from
// nvidia-smi on Linux hosts, false on Mac Docker (no passthrough).

import { useHostMetrics } from "../hooks/useHostMetrics";
import { Sparkline } from "./Sparkline";
import { UserMenu } from "./UserMenu";

export function TopBar() {
  const { cpu, ram, gpu, latest, error } = useHostMetrics();
  const gpuAvailable = latest?.gpu.available ?? false;
  const gpuTitle = error
    ? `metrics: ${error}`
    : gpuAvailable
      ? `${latest!.gpu.devices[0]?.name ?? "GPU"} · util ${
          latest!.gpu.devices[0]?.utilization_percent.toFixed(0) ?? "0"
        }%`
      : latest?.gpu.note ||
        "No GPU passthrough into the Docker VM on Mac · worker runs on CPU";

  return (
    <header className="topbar">
      <div className="topbar__project">
        <span className="topbar__project-label">Active project</span>
        <span className="topbar__project-value topbar__project-value--dim">
          — none —
        </span>
      </div>

      <span className="topbar__spacer" />

      <div className="topbar__metrics">
        <span
          className={
            gpuAvailable
              ? "gpu-status gpu-status--online"
              : "gpu-status gpu-status--offline"
          }
          title={gpuTitle}
        >
          <span className="gpu-status__dot" aria-hidden="true" />
          <span className="gpu-status__label">
            {gpuAvailable ? "GPU" : "no GPU"}
          </span>
        </span>
        {gpuAvailable && (
          <Sparkline label="GPU" values={gpu} current={gpu.at(-1) ?? 0} />
        )}
        <Sparkline label="CPU" values={cpu} current={cpu.at(-1) ?? 0} />
        <Sparkline label="RAM" values={ram} current={ram.at(-1) ?? 0} />
      </div>

      <UserMenu />
    </header>
  );
}
