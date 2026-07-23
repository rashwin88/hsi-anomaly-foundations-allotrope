// Inline Ingest panel for the Scenes page.
//
// Renders sensor picker → file/folder input → submit. After submit, polls
// the resulting job via useJobStatus and reports progress. On success,
// notifies the parent so the Scenes table can refresh.
//
// File-input mode flips with sensor:
//   prisma   → single .he5
//   landsat9 → single .tif
//   enmap    → directory (webkitdirectory) — multi-file
//
// Storyboard-spec § 5 (Scenes destination) — Ingest is the data-in
// surface; everything else on this page is read.
// Sequence diagram for the underlying call: scene-onboard.drawio (Step 7e)

import { type FormEvent, useEffect, useRef, useState } from "react";

import { ApiError } from "../api/client";
import { onboardScene } from "../api/scenes";
import { useJobStatus } from "../hooks/useJobStatus";
import type { SensorType } from "../types";
import { useToast } from "./Toast";

interface IngestPanelProps {
  onClose: () => void;
  onComplete: () => void;
}

const SENSOR_OPTIONS: { value: SensorType; label: string; hint: string }[] = [
  { value: "prisma", label: "PRISMA", hint: "single .he5 file" },
  { value: "landsat9", label: "Landsat 9", hint: "single .tif file" },
  { value: "enmap", label: "EnMAP", hint: "folder containing METADATA.XML + TIFs" },
  { value: "aviris_ng", label: "AVIRIS-NG", hint: "ang<date>t<time> folder containing _corr_.bin + .hdr" },
  { value: "hotsat1", label: "HotSAT-1", hint: "SatVu L2 Visual folder with metadata.json + _ortho.tiff + _udm.tiff" },
];

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function IngestPanel({ onClose, onComplete }: IngestPanelProps) {
  const [sensor, setSensor] = useState<SensorType>("prisma");
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  // Upload progress (XHR upload.onprogress). Null while no upload in
  // flight; { loaded, total } while bytes ship; total==0 means the
  // browser can't determine size (rare for File-backed FormData).
  const [uploadProgress, setUploadProgress] = useState<
    { loaded: number; total: number } | null
  >(null);
  // True after the browser has shipped the last byte but before the api
  // returns 202 — flips the UI from "Uploading XX%" to "Staging files…".
  // The api spends 1–5 s spooling the multipart body to disk in this window.
  const [staging, setStaging] = useState(false);
  const uploadStartRef = useRef<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const { job, error: pollError } = useJobStatus(jobId);

  // When the job lands in "complete", notify the parent so the table
  // re-fetches. We do this exactly once per terminal transition.
  const notifiedRef = useRef(false);
  useEffect(() => {
    if (job?.status === "complete" && !notifiedRef.current) {
      notifiedRef.current = true;
      onComplete();
    }
  }, [job?.status, onComplete]);

  // Reset file selection when the sensor switches — different inputs are
  // mounted (single-file vs. directory) so the previous selection isn't
  // semantically valid anymore.
  const onSensorChange = (next: SensorType) => {
    setSensor(next);
    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onFilesChosen = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFiles(Array.from(e.target.files ?? []));
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;
    setSubmitError(null);
    setSubmitting(true);
    setUploadProgress({ loaded: 0, total: totalBytes });
    uploadStartRef.current = performance.now();
    try {
      const res = await onboardScene(
        sensor,
        files,
        (p) => {
          setUploadProgress({ loaded: p.loaded, total: p.total || totalBytes });
        },
        () => {
          // Browser is done sending bytes. The api is now reading the
          // multipart spool and writing to the staging volume — typically
          // a few seconds. Flip the label so the user doesn't think the
          // app froze at 100%.
          setStaging(true);
        },
      );
      setJobId(res.job_id);
      // Upload finished — keep the bar at 100% until the polling phase
      // takes over (so the user sees the transition rather than an
      // abrupt unmount).
      setUploadProgress({ loaded: totalBytes, total: totalBytes });
      setStaging(false);
      // Fire-and-forget: lets the user dismiss the panel and still track
      // the job from anywhere in the app via the link. Today the link
      // goes to /jobs (currently a placeholder); when the per-job route
      // lands we'll switch to /jobs/<id>.
      toast.push({
        title: "Onboarding job started",
        message: `${SENSOR_OPTIONS.find((o) => o.value === sensor)?.label} · ${res.staged_count} file${res.staged_count === 1 ? "" : "s"} staged`,
        variant: "info",
        link: { to: "/jobs", label: "View in Jobs" },
        durationMs: 8000,
      });
    } catch (err) {
      if (err instanceof ApiError) {
        setSubmitError(err.detail ?? `Error: HTTP ${err.status}`);
      } else {
        setSubmitError("Could not reach the server.");
      }
    } finally {
      setSubmitting(false);
      setStaging(false);
    }
  };

  // While a job is running we hide the form and show progress instead.
  const inFlight = jobId !== null && job?.status !== "complete" && job?.status !== "failed" && job?.status !== "cancelled";
  const finished = job?.status === "complete";
  const failed = job?.status === "failed" || job?.status === "cancelled";

  const totalBytes = files.reduce((acc, f) => acc + f.size, 0);
  const isFolderSensor = sensor === "enmap" || sensor === "aviris_ng" || sensor === "hotsat1";
  const accept = sensor === "prisma" ? ".he5" : sensor === "landsat9" ? ".tif,.tiff" : undefined;

  return (
    <section className="ingest panel">
      <header className="ingest__header">
        <h2 className="panel__heading">Ingest scene</h2>
        <button type="button" className="ingest__close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </header>

      {!jobId && (
        <form onSubmit={onSubmit} className="ingest__form">
          <fieldset className="ingest__fieldset">
            <legend className="form__label">Sensor</legend>
            <div className="ingest__sensors">
              {SENSOR_OPTIONS.map((opt) => (
                <label
                  key={opt.value}
                  className={
                    "ingest__sensor" + (sensor === opt.value ? " is-active" : "")
                  }
                >
                  <input
                    type="radio"
                    name="sensor"
                    value={opt.value}
                    checked={sensor === opt.value}
                    onChange={() => onSensorChange(opt.value)}
                  />
                  <span className="ingest__sensor-label">{opt.label}</span>
                  <span className="ingest__sensor-hint">{opt.hint}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <label className="form__field">
            <span className="form__label">
              {isFolderSensor ? "Pick the scene folder" : "Pick the scene file"}
            </span>
            {/*
              Re-mount the input when sensor changes so the
              `webkitdirectory` attribute is applied/removed cleanly —
              the React DOM diff doesn't toggle that attribute correctly
              if the same element is reused.
            */}
            {isFolderSensor ? (
              <input
                key="enmap-dir"
                ref={fileInputRef}
                type="file"
                onChange={onFilesChosen}
                /* @ts-expect-error — webkitdirectory is non-standard but supported in Chromium/WebKit */
                webkitdirectory=""
                directory=""
                multiple
              />
            ) : (
              <input
                key={`single-${sensor}`}
                ref={fileInputRef}
                type="file"
                onChange={onFilesChosen}
                accept={accept}
              />
            )}
          </label>

          {files.length > 0 && (
            <p className="ingest__summary">
              {files.length} file{files.length === 1 ? "" : "s"} selected
              {" · "}
              {formatBytes(totalBytes)}
            </p>
          )}

          {submitError && (
            <p className="form__error" role="alert">
              {submitError}
            </p>
          )}

          {submitting && uploadProgress && (
            <UploadProgressBar
              loaded={uploadProgress.loaded}
              total={uploadProgress.total || totalBytes}
              startedAt={uploadStartRef.current}
            />
          )}
          {submitting && staging && (
            <p className="ingest__staging-note small muted">
              Staging files on the server… this can take a few seconds for
              multi-GB uploads.
            </p>
          )}

          <div className="ingest__actions">
            <button
              type="button"
              className="ingest__cancel"
              onClick={onClose}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="form__submit"
              disabled={submitting || files.length === 0}
            >
              {submitting
                ? staging
                  ? "Staging…"
                  : "Uploading…"
                : "Start onboarding"}
            </button>
          </div>
        </form>
      )}

      {jobId && (
        <div className="ingest__progress">
          <p className="ingest__job-id">
            Job: <span className="mono">{jobId}</span>
          </p>
          <p className="ingest__status">
            Status:{" "}
            <span className={`status-pill status-pill--${job?.status ?? "queued"}`}>
              {job?.status ?? "submitting"}
            </span>
          </p>
          {pollError && (
            <p className="ingest__poll-error">
              Polling: {pollError} (still trying)
            </p>
          )}
          {inFlight && (
            <p className="ingest__hint">
              The worker is parsing metadata and rendering a thumbnail. This
              can take a few seconds for small files, longer for full
              hyperspectral scenes.
            </p>
          )}
          {finished && (
            <>
              <p className="ingest__success">
                Scene onboarded successfully.
                {job?.target_id && (
                  <>
                    {" "}
                    <span className="mono">{job.target_id}</span>
                  </>
                )}
              </p>
              <button type="button" className="form__submit" onClick={onClose}>
                Close
              </button>
            </>
          )}
          {failed && (
            <>
              <p className="form__error">
                {job?.failure_reason?.split("\n")[0] ?? "Onboarding failed."}
              </p>
              <button type="button" className="form__submit" onClick={onClose}>
                Close
              </button>
            </>
          )}
        </div>
      )}
    </section>
  );
}

// --- UploadProgressBar -----------------------------------------------
//
// Renders the live byte / percent / ETA view above the action row while
// the multipart upload is in flight. Pure presentation — state lives in
// the parent.

function UploadProgressBar({
  loaded,
  total,
  startedAt,
}: {
  loaded: number;
  total: number;
  startedAt: number | null;
}) {
  const pct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0;
  const elapsedMs =
    startedAt !== null ? Math.max(1, performance.now() - startedAt) : 0;
  const bytesPerSec = elapsedMs > 0 ? (loaded / elapsedMs) * 1000 : 0;
  const remaining = total - loaded;
  const etaSec =
    bytesPerSec > 0 && remaining > 0 ? remaining / bytesPerSec : null;
  return (
    <div
      className="ingest__upload"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      aria-label="Uploading scene files"
    >
      <div className="ingest__upload-bar">
        <div
          className="ingest__upload-bar-fill"
          style={{ width: `${pct.toFixed(1)}%` }}
        />
      </div>
      <div className="ingest__upload-meta small">
        <span>
          {formatBytes(loaded)} / {formatBytes(total)} ·{" "}
          <strong>{pct.toFixed(0)}%</strong>
        </span>
        <span>
          {bytesPerSec > 0 ? `${formatBytes(bytesPerSec)}/s` : "—"}
          {etaSec !== null ? ` · ETA ${formatEta(etaSec)}` : ""}
        </span>
      </div>
    </div>
  );
}

function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  return `${m}m ${s}s`;
}
