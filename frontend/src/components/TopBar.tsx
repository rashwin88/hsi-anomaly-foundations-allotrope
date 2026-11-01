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
import { APP_VERSION } from "../version";
import { gpuDeviceLine, gpuIdleButPresent, gpuSourceLine, modelsOnLine } from "./gpuText";
import { Sparkline } from "./Sparkline";
import { UserMenu } from "./UserMenu";

export function TopBar() {
  const { cpu, ram, gpu, latest, error } = useHostMetrics();
  const gpuAvailable = latest?.gpu.available ?? false;
  const gpuSnap = latest?.gpu;
  // Name, utilisation, memory and temperature, where the numbers come from, and where the
  // models run (the worker's torch device).
  const gpuTitle = error
    ? `metrics: ${error}`
    : gpuSnap && gpuAvailable
      ? [gpuSnap.devices.map(gpuDeviceLine).join("\n"), gpuSourceLine(gpuSnap), modelsOnLine(gpuSnap)]
          .filter(Boolean)
          .join("\n")
      : [gpuSnap?.note || "No GPU reported", gpuSnap ? modelsOnLine(gpuSnap) : ""].filter(Boolean).join("\n");
  const gpuLabel = !gpuAvailable
    ? "no GPU"
    : gpuSnap && gpuIdleButPresent(gpuSnap)
      ? "GPU · jobs on CPU"
      : `GPU ${gpuSnap?.devices[0]?.utilization_percent?.toFixed(0) ?? "–"}%`;

  return (
    <header className="topbar">
      <span className="small" style={{ color: "var(--ink-3)" }} title="App version">
        Allotrope v{APP_VERSION}
      </span>
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
            {gpuLabel}
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
