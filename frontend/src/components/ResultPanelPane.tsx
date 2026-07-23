// Project workspace Result tab (Step 17).
//
// Computed view: pulls /projects/{id}/result and shows the live roll-up
// of completed Actions + counts of curated entities. The "Create export"
// button enqueues a project_export job; while one is in flight, a small
// status block polls until the new bundle row lands.

import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError } from "../api/client";
import { getJob } from "../api/jobs";
import {
  createProjectExport,
  getProjectResult,
  listProjectExports,
  type ExportPublic,
  type ResultPublic,
} from "../api/result";

interface Props {
  projectId: string;
}

const RESULT_POLL_MS = 4000;
const EXPORT_POLL_MS = 2000;

function bytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KiB`;
  return `${n} B`;
}

function dt(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

export function ResultPanelPane({ projectId }: Props) {
  const [result, setResult] = useState<ResultPublic | null>(null);
  const [exports, setExports] = useState<ExportPublic[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingJobId, setPendingJobId] = useState<string | null>(null);
  const [pendingStatus, setPendingStatus] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  // ---- Live Result ---------------------------------------------------
  const inFlight = useMemo(
    () =>
      result?.actions.some(
        (a) => a.status === "queued" || a.status === "running",
      ) ?? false,
    [result],
  );

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const [r, ex] = await Promise.all([
          getProjectResult(projectId),
          listProjectExports(projectId),
        ]);
        if (cancelled) return;
        setResult(r);
        setExports(ex.items);
        setError(null);
      } catch (err) {
        if (!cancelled)
          setError(
            err instanceof ApiError
              ? err.detail ?? `HTTP ${err.status}`
              : "Could not reach the server.",
          );
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), RESULT_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [projectId]);

  // ---- Pending export polling ---------------------------------------
  useEffect(() => {
    if (!pendingJobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const job = await getJob(pendingJobId);
        if (cancelled) return;
        setPendingStatus(job.status);
        if (job.status === "complete" || job.status === "failed" || job.status === "cancelled") {
          if (job.status === "failed") {
            setExportError(job.failure_reason ?? "export_failed");
          }
          setPendingJobId(null);
          // re-fetch exports list to surface the new row
          try {
            const ex = await listProjectExports(projectId);
            if (!cancelled) setExports(ex.items);
          } catch {
            // ignore — next result tick will pick it up
          }
        }
      } catch {
        // transient error; keep polling
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), EXPORT_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pendingJobId, projectId]);

  const onCreateExport = useCallback(async () => {
    setExportError(null);
    try {
      const accepted = await createProjectExport(projectId);
      setPendingJobId(accepted.job_id);
      setPendingStatus("queued");
    } catch (err) {
      setExportError(
        err instanceof ApiError
          ? err.detail ?? `HTTP ${err.status}`
          : "Could not enqueue export.",
      );
    }
  }, [projectId]);

  return (
    <section className="workspace__card">
      <div className="workspace__card-header">
        <h3 className="workspace__card-title">Result</h3>
        {result && (
          <span className="small">
            last completed{" "}
            <strong>{dt(result.last_action_completed_at)}</strong>
            {inFlight ? " · polling" : ""}
          </span>
        )}
      </div>

      {error && <p className="form__error" role="alert">{error}</p>}

      {!result && !error && (
        <div className="workspace__card-body workspace__card-body--loading">
          Loading…
        </div>
      )}

      {result && (
        <>
          <dl className="result-meta">
            <div>
              <dt>Project</dt>
              <dd>{result.project.name}</dd>
            </div>
            <div>
              <dt>Scene</dt>
              <dd>
                {result.project.scene_name}{" "}
                <code className="small">{result.project.scene_sensor_type}</code>
              </dd>
            </div>
            <div>
              <dt>Visualizations</dt>
              <dd className="mono">{result.visualization_count}</dd>
            </div>
            <div>
              <dt>Notes</dt>
              <dd className="mono">{result.note_count}</dd>
            </div>
            <div>
              <dt>Annotations</dt>
              <dd className="mono">{result.annotation_count}</dd>
            </div>
          </dl>

          {result.actions.length === 0 ? (
            <div className="workspace__empty">
              <p>
                No Actions yet. Submit one from the <strong>Action</strong>{" "}
                tab and it'll appear here as it runs and completes.
              </p>
            </div>
          ) : (
            <table className="jobs-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Started</th>
                  <th>Completed</th>
                  <th>Output</th>
                </tr>
              </thead>
              <tbody>
                {result.actions.map((a) => (
                  <tr key={a.id}>
                    <td className="mono small">{a.type}</td>
                    <td>
                      <span
                        className="action-card__status"
                        data-tone={a.status}
                      >
                        ● {a.status}
                      </span>
                    </td>
                    <td className="small">{dt(a.started_at)}</td>
                    <td className="small">{dt(a.completed_at)}</td>
                    <td className="mono small">
                      {a.output_id ?? (a.failure_reason ? "—" : "pending")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div className="result-export">
            <h4>Export</h4>
            <p className="small">
              Bundle the current Result + every completed Action output,
              every saved Visualization, every Note, and the Scene
              thumbnail into a single zip you can hand off.
            </p>
            <button
              type="button"
              className="anomaly-viewer__tool-btn"
              data-active="true"
              onClick={() => void onCreateExport()}
              disabled={pendingJobId !== null}
            >
              {pendingJobId
                ? `Building export… (${pendingStatus ?? "queued"})`
                : "Create export"}
            </button>
            {exportError && (
              <p className="form__error" role="alert">
                {exportError}
              </p>
            )}

            {exports && exports.length > 0 && (
              <table className="jobs-table">
                <thead>
                  <tr>
                    <th>Created</th>
                    <th>Snapshot</th>
                    <th>Size</th>
                    <th>Format</th>
                    <th>Download</th>
                  </tr>
                </thead>
                <tbody>
                  {exports.map((e) => (
                    <tr key={e.id}>
                      <td className="small">{dt(e.created_at)}</td>
                      <td className="small">{dt(e.snapshot_at)}</td>
                      <td className="mono">{bytes(e.size_bytes)}</td>
                      <td className="mono small">{e.format}</td>
                      <td>
                        <a
                          className="anomaly-viewer__tool-btn"
                          href={`/api${e.download_url}`}
                        >
                          Download
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </section>
  );
}
