// Right-edge slide-out drawer showing in-flight scene_onboard jobs.
//
// Pulled into ScenesPage. Polls /jobs?type=scene_onboard&status=...
// every 3 s while open; throttles to 8 s while collapsed so the badge
// stays current without thrashing the api. Calls `onJobFinished` once
// each job leaves the in-flight set, so the parent can refresh the
// scenes table to show the freshly-onboarded row.
//
// A scene_onboard job's payload carries `files` (upload-order paths)
// and `sensor_type`. We render the first file's basename as the
// row's primary label; sensor pill + status badge + elapsed timer
// round it out.

import { useEffect, useMemo, useRef, useState } from "react";

import { listJobs } from "../api/jobs";
import type { Job, JobStatus } from "../types";

interface Props {
  /** Bumped when the panel's ingest form kicks off a new job, so the
   *  drawer fetches immediately instead of waiting for its poll tick. */
  refreshSignal?: number;
  /** Called once per job-id when that job transitions out of the
   *  in-flight set (queued/running → complete/failed/cancelled). */
  onJobFinished?: (job: Job) => void;
}

const IN_FLIGHT_STATUSES: ReadonlySet<JobStatus> = new Set([
  "queued",
  "running",
] as const);

const POLL_OPEN_MS = 3000;
const POLL_CLOSED_MS = 8000;

const SENSOR_LABELS: Record<string, string> = {
  prisma: "PRISMA",
  landsat9: "Landsat 9",
  enmap: "EnMAP",
  aviris_ng: "AVIRIS-NG",
  hotsat1: "HotSAT-1",
};

function basename(p: string): string {
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(i + 1) : p;
}

function primaryLabel(job: Job): string {
  const files = (job.payload?.files as string[] | undefined) ?? [];
  if (files.length === 0) return "(no filename)";
  const first = basename(files[0]);
  return files.length > 1 ? `${first} +${files.length - 1}` : first;
}

function sensorOf(job: Job): string | null {
  const s = job.payload?.sensor_type as string | undefined;
  return s ?? null;
}

function elapsed(job: Job): string {
  const start = job.started_at ?? job.created_at;
  if (!start) return "—";
  const ms = Date.now() - new Date(start).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return "<1s";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
}

export function OnboardingDrawer({ refreshSignal = 0, onJobFinished }: Props) {
  const [open, setOpen] = useState(false);
  const [inFlight, setInFlight] = useState<Job[]>([]);
  // We need to remember which ids were in-flight last tick so we can fire
  // onJobFinished exactly once per transition.
  const lastIdsRef = useRef<Set<string>>(new Set());
  // Tick state nudges the elapsed-timer column so it ticks even while
  // the underlying jobs payload doesn't change.
  const [, setNowTick] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => setNowTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        // We can't pass two statuses through the api, so pull a small
        // recent page across all statuses + filter client-side. Cheap
        // (≤ 50 rows, indexed scan).
        const page = await listJobs({ type: "scene_onboard", limit: 50 });
        if (cancelled) return;
        const active = page.items.filter((j) =>
          IN_FLIGHT_STATUSES.has(j.status),
        );
        const activeIds = new Set(active.map((j) => j.id));

        // Fire onJobFinished for ids that were in-flight last tick but
        // aren't anymore. Lookup their settled row from the same page.
        if (onJobFinished) {
          for (const prevId of lastIdsRef.current) {
            if (!activeIds.has(prevId)) {
              const settled = page.items.find((j) => j.id === prevId);
              if (settled) onJobFinished(settled);
            }
          }
        }
        lastIdsRef.current = activeIds;
        setInFlight(active);
      } catch {
        // network blip — keep the last list visible, try again next tick
      }
    };
    void tick(); // immediate fetch on open / refreshSignal bump
    const ms = open ? POLL_OPEN_MS : POLL_CLOSED_MS;
    const id = window.setInterval(() => void tick(), ms);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [open, refreshSignal, onJobFinished]);

  const count = inFlight.length;

  // Pull-tab is always rendered; drawer panel slides over from the right
  // when `open`. Click-outside closes via a scrim that only renders when
  // open (to avoid swallowing pointer events on the main page).
  return (
    <div className="onboarding-drawer" data-open={open ? "true" : "false"}>
      <button
        type="button"
        className="onboarding-drawer__tab"
        onClick={() => setOpen((v) => !v)}
        title={
          count === 0
            ? "No onboarding jobs in flight"
            : `${count} scene${count === 1 ? "" : "s"} onboarding`
        }
      >
        <span className="onboarding-drawer__tab-label">
          {open ? "›" : "‹"} Onboarding
        </span>
        {count > 0 && (
          <span className="onboarding-drawer__badge" aria-label={`${count} active`}>
            {count}
          </span>
        )}
      </button>

      {open && (
        <>
          <div
            className="onboarding-drawer__scrim"
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          <aside className="onboarding-drawer__panel" role="dialog" aria-label="Onboarding jobs in flight">
            <header className="onboarding-drawer__header">
              <h3>Onboarding</h3>
              <button
                type="button"
                className="onboarding-drawer__close"
                onClick={() => setOpen(false)}
                aria-label="Close"
              >
                ×
              </button>
            </header>

            {inFlight.length === 0 ? (
              <div className="onboarding-drawer__empty">
                <p>No scenes onboarding right now.</p>
                <p className="small">
                  When you start an ingest, its progress lands here — close the
                  ingest panel and watch the drawer instead.
                </p>
              </div>
            ) : (
              <ul className="onboarding-drawer__list">
                {inFlight.map((j) => (
                  <OnboardingRow key={j.id} job={j} />
                ))}
              </ul>
            )}
          </aside>
        </>
      )}
    </div>
  );
}

function OnboardingRow({ job }: { job: Job }) {
  const sensor = sensorOf(job);
  const label = useMemo(() => primaryLabel(job), [job]);
  return (
    <li className="onboarding-row" data-status={job.status}>
      <div className="onboarding-row__head">
        <span className="onboarding-row__name" title={label}>{label}</span>
        <span
          className="action-card__status"
          data-tone={job.status}
        >
          ● {job.status}
        </span>
      </div>
      <div className="onboarding-row__meta small">
        {sensor && (
          <span className="sensor-pill">
            {SENSOR_LABELS[sensor] ?? sensor}
          </span>
        )}
        <span className="mono">{elapsed(job)}</span>
      </div>
    </li>
  );
}
