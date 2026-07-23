// Jobs destination (Step 19 + polish pass).
//
// Live snapshot of the Postgres-backed work queue. Layout:
//
//   ┌─ KPI strip ─────────────────────────────────────────────────────┐
//   │  in-flight │ queued │ done last hour │ failed last hour         │
//   └─────────────────────────────────────────────────────────────────┘
//   ┌─ status chips ──────────────────────────────────────────────────┐
//   │  [ All ] [ Queued ] [ Running ] [ Complete ] [ Failed ] …       │
//   │  Type ▾   ⟳ polling                                             │
//   └─────────────────────────────────────────────────────────────────┘
//   ┌─ table (sticky header) ─────────────────────────────────────────┐
//   │  Job              · Type pill           · Project / Target      │
//   │  Started / Elapsed                      · Status                │
//   │  ▸ failure_reason expansion (when present)                      │
//   └─────────────────────────────────────────────────────────────────┘
//
// Sequence diagram: final design/diagrams/jobs-list.drawio

import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError } from "../api/client";
import { listJobs, type JobsPage } from "../api/jobs";
import type { Job, JobStatus, JobType } from "../types";

const PAGE_SIZE = 10;
const POLL_MS = 3000;

const STATUS_CHIPS: Array<{ value: JobStatus | "all"; label: string }> = [
  { value: "all", label: "All" },
  { value: "queued", label: "Queued" },
  { value: "running", label: "Running" },
  { value: "complete", label: "Complete" },
  { value: "failed", label: "Failed" },
  { value: "cancelled", label: "Cancelled" },
];

const TYPE_OPTIONS: Array<{ value: JobType | "all"; label: string }> = [
  { value: "all", label: "All types" },
  { value: "action_run", label: "action_run" },
  { value: "scene_onboard", label: "scene_onboard" },
  { value: "annotation_attach", label: "annotation_attach" },
  { value: "project_export", label: "project_export" },
];

const TYPE_TONE: Record<string, string> = {
  action_run: "violet",
  scene_onboard: "blue",
  annotation_attach: "amber",
  project_export: "teal",
};

function fmt(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function relative(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const sec = (Date.now() - d.getTime()) / 1000;
  if (sec < 60) return `${Math.max(1, Math.floor(sec))}s ago`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function elapsed(j: Job): string {
  const start = j.started_at ?? j.created_at;
  const end = j.completed_at ?? j.last_heartbeat_at ?? new Date().toISOString();
  if (!start) return "—";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return "<1s";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function shortIdTail(wire: string | null | undefined): string {
  if (!wire) return "—";
  const i = wire.indexOf("_");
  const tail = i >= 0 ? wire.slice(i + 1) : wire;
  return tail.slice(0, 8);
}

function CopyableId({
  wire,
  prefix,
}: {
  wire: string | null;
  prefix?: string;
}) {
  const [copied, setCopied] = useState(false);
  if (!wire) return <span className="jobs__id jobs__id--empty">—</span>;
  const tail = shortIdTail(wire);
  return (
    <button
      type="button"
      className="jobs__id"
      title={`${wire} — click to copy`}
      onClick={async (e) => {
        e.stopPropagation();
        try {
          await navigator.clipboard.writeText(wire);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1100);
        } catch {
          /* clipboard denied — fall through silently */
        }
      }}
    >
      {prefix && <span className="jobs__id-prefix">{prefix}</span>}
      <span className="jobs__id-tail mono">{tail}</span>
      {copied && <span className="jobs__id-copied">copied</span>}
    </button>
  );
}

export function JobsPage() {
  const [statusFilter, setStatusFilter] = useState<JobStatus | "all">("all");
  const [typeFilter, setTypeFilter] = useState<JobType | "all">("all");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<JobsPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // KPI strip pulls a tiny "all jobs, no filter" page so its counts
  // don't shift when the user narrows the table. We over-fetch once per
  // poll cycle — still cheap (one indexed scan) and decouples the
  // top-of-page summary from the bottom-of-page filter state.
  const [kpiSource, setKpiSource] = useState<Job[]>([]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listJobs({
      limit: PAGE_SIZE,
      offset,
      status: statusFilter === "all" ? undefined : statusFilter,
      type: typeFilter === "all" ? undefined : typeFilter,
    })
      .then((result) => {
        if (cancelled) return;
        setPage(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.detail ?? `HTTP ${err.status}`
            : "Could not reach the server.",
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [statusFilter, typeFilter, offset, reloadTick]);

  // KPI strip — unfiltered, last 200 rows is enough to compute the
  // "in last hour" buckets without dragging the whole table down.
  useEffect(() => {
    let cancelled = false;
    listJobs({ limit: 200 })
      .then((res) => {
        if (!cancelled) setKpiSource(res.items);
      })
      .catch(() => {
        // Non-fatal — main table's error block already surfaces failures.
      });
    return () => {
      cancelled = true;
    };
  }, [reloadTick]);

  // Poll while any visible row is in flight.
  const inFlight = useMemo(
    () =>
      (page?.items ?? []).some(
        (j) => j.status === "queued" || j.status === "running",
      ),
    [page],
  );
  useEffect(() => {
    if (!inFlight) return;
    const id = window.setInterval(() => setReloadTick((t) => t + 1), POLL_MS);
    return () => window.clearInterval(id);
  }, [inFlight]);

  // Tick once per second so elapsed timers refresh visually.
  const [, setNow] = useState(Date.now());
  const tickerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!inFlight) return;
    tickerRef.current = window.setInterval(() => setNow(Date.now()), 1000);
    return () => {
      if (tickerRef.current !== null) window.clearInterval(tickerRef.current);
    };
  }, [inFlight]);

  const onChangeStatus = (v: JobStatus | "all") => {
    setStatusFilter(v);
    setOffset(0);
  };
  const onChangeType = (v: JobType | "all") => {
    setTypeFilter(v);
    setOffset(0);
  };

  const totalPages = page ? Math.ceil(page.total / page.limit) : 0;
  const currentPage = page ? Math.floor(page.offset / page.limit) + 1 : 0;

  // KPI buckets ------------------------------------------------------
  const kpi = useMemo(() => {
    const oneHourAgo = Date.now() - 60 * 60 * 1000;
    let running = 0;
    let queued = 0;
    let completedHour = 0;
    let failedHour = 0;
    for (const j of kpiSource) {
      if (j.status === "running") running++;
      else if (j.status === "queued") queued++;
      else if (j.status === "complete" && j.completed_at) {
        if (new Date(j.completed_at).getTime() >= oneHourAgo) completedHour++;
      } else if (j.status === "failed" && j.completed_at) {
        if (new Date(j.completed_at).getTime() >= oneHourAgo) failedHour++;
      }
    }
    return { running, queued, completedHour, failedHour };
  }, [kpiSource]);

  return (
    <section className="page jobs-page">
      <header className="page__header">
        <h1 className="page__title">Jobs</h1>
        <p className="page__subtitle">
          The Postgres-backed work queue. Worker claims rows via{" "}
          <code>FOR UPDATE SKIP LOCKED</code>; this page mirrors live state and
          auto-polls while anything is in flight.
        </p>
      </header>

      {/* ─── KPI strip ────────────────────────────────────────── */}
      <div className="jobs__kpi">
        <KpiCard
          tone="running"
          value={kpi.running}
          label="Running"
          hint={kpi.running > 0 ? "worker is busy" : "worker idle"}
        />
        <KpiCard
          tone="queued"
          value={kpi.queued}
          label="Queued"
          hint={
            kpi.queued > 0
              ? "waiting to claim"
              : "no backlog"
          }
        />
        <KpiCard
          tone="complete"
          value={kpi.completedHour}
          label="Done · last hour"
        />
        <KpiCard
          tone="failed"
          value={kpi.failedHour}
          label="Failed · last hour"
        />
      </div>

      {/* ─── Filter row ────────────────────────────────────────── */}
      <div className="jobs__filters">
        <div className="jobs__chips" role="group" aria-label="Filter by status">
          {STATUS_CHIPS.map((c) => (
            <button
              key={c.value}
              type="button"
              className="jobs__chip"
              data-active={statusFilter === c.value ? "true" : "false"}
              data-tone={c.value === "all" ? undefined : c.value}
              onClick={() => onChangeStatus(c.value)}
            >
              {c.label}
            </button>
          ))}
        </div>
        <div className="jobs__type-filter">
          <label className="form__label form__label--inline small">
            Type
            <select
              className="form__select"
              value={typeFilter}
              onChange={(e) => onChangeType(e.target.value as JobType | "all")}
            >
              {TYPE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <span className="jobs__poll-state small">
            {inFlight && <span className="jobs__poll-dot" aria-hidden="true" />}
            {page
              ? `${page.total.toLocaleString()} job${page.total === 1 ? "" : "s"}`
              : ""}
            {inFlight ? " · live" : ""}
          </span>
        </div>
      </div>

      {error && (
        <p className="form__error" role="alert">
          {error}
        </p>
      )}

      {loading && !page && <p className="page__hint">Loading…</p>}

      {page && page.items.length === 0 && !loading && (
        <div className="jobs__empty">
          <p>No jobs match those filters.</p>
        </div>
      )}

      {page && page.items.length > 0 && (
        <div className="jobs-table-wrap">
          <table className="jobs-table jobs-table--polished">
            <thead>
              <tr>
                <th>Job</th>
                <th>Project / Target</th>
                <th>Started</th>
                <th>Elapsed</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {page.items.map((j) => {
                const isOpen = expandedId === j.id;
                const hasFailure = Boolean(j.failure_reason);
                return (
                  <Row
                    key={j.id}
                    job={j}
                    isOpen={isOpen}
                    hasFailure={hasFailure}
                    onToggle={() =>
                      setExpandedId(isOpen ? null : j.id)
                    }
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {page && totalPages > 1 && (
        <div className="jobs__pagination">
          <button
            type="button"
            className="jobs__page-btn"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
          >
            ← Previous
          </button>
          <span className="small">
            Page {currentPage} / {totalPages}
          </span>
          <button
            type="button"
            className="jobs__page-btn"
            disabled={offset + PAGE_SIZE >= page.total}
            onClick={() => setOffset(offset + PAGE_SIZE)}
          >
            Next →
          </button>
        </div>
      )}
    </section>
  );
}

// --- KPI card --------------------------------------------------------

function KpiCard({
  tone,
  value,
  label,
  hint,
}: {
  tone: JobStatus;
  value: number;
  label: string;
  hint?: string;
}) {
  return (
    <div className="jobs__kpi-card" data-tone={tone} data-zero={value === 0 ? "true" : "false"}>
      <div className="jobs__kpi-value">{value.toLocaleString()}</div>
      <div className="jobs__kpi-label">{label}</div>
      {hint && <div className="jobs__kpi-hint small">{hint}</div>}
    </div>
  );
}

// --- Row -------------------------------------------------------------

function Row({
  job,
  isOpen,
  hasFailure,
  onToggle,
}: {
  job: Job;
  isOpen: boolean;
  hasFailure: boolean;
  onToggle: () => void;
}) {
  const tone = TYPE_TONE[job.type] ?? "neutral";
  return (
    <>
      <tr
        className="jobs-row"
        data-status={job.status}
        data-expandable={hasFailure ? "true" : "false"}
        onClick={hasFailure ? onToggle : undefined}
        role={hasFailure ? "button" : undefined}
      >
        <td>
          <div className="jobs-row__primary">
            <span className="jobs-row__type" data-tone={tone}>
              {job.type}
            </span>
            <CopyableId wire={job.id} prefix="job_" />
          </div>
          <div className="jobs-row__created small">
            created {relative(job.created_at)}
          </div>
        </td>
        <td>
          <div className="jobs-row__stack">
            <CopyableId wire={job.project_id} prefix="project_" />
            <CopyableId wire={job.target_id} prefix="" />
          </div>
        </td>
        <td>
          {job.started_at ? (
            <div className="jobs-row__stack">
              <span className="small">{fmt(job.started_at)}</span>
              <span className="small jobs-row__rel">{relative(job.started_at)}</span>
            </div>
          ) : (
            <span className="small">— not yet —</span>
          )}
        </td>
        <td className="mono">
          {elapsed(job)}
        </td>
        <td>
          <div className="jobs-row__status-cell">
            <StatusPill status={job.status as JobStatus} />
            {job.cancellation_requested && (
              <span className="jobs-row__cancel-flag" title="Cancellation requested">
                cancel?
              </span>
            )}
            {hasFailure && (
              <span className="jobs-row__chev" aria-hidden="true">
                {isOpen ? "▾" : "▸"}
              </span>
            )}
          </div>
        </td>
      </tr>
      {isOpen && hasFailure && (
        <tr className="jobs-row__expansion">
          <td colSpan={5}>
            <div className="jobs-row__failure">
              <div className="jobs-row__failure-label small">failure_reason</div>
              <pre className="jobs-row__failure-body">{job.failure_reason}</pre>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function StatusPill({ status }: { status: JobStatus }) {
  return (
    <span className="jobs-status" data-status={status}>
      <span className="jobs-status__dot" aria-hidden="true" />
      <span>{status}</span>
    </span>
  );
}
