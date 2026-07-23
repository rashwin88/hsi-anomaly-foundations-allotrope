// Center-pane Action card + per-type Output viewer (Step 12h).
//
// Reads the selected Action via /actions/{id}, polls every 3 s while
// queued/running so the card animates through status transitions, and
// dispatches per-type rendering (band_filter_apply / scene_segmentation)
// once ActionOutput is materialised.
//
// Sequence diagrams:
//   action-detail.drawio · action-file.drawio · action-types-catalog.drawio

import { useEffect, useMemo, useRef, useState } from "react";

import {
  type AnomalyDetectionPreviewResponse,
  getAction,
  submitAnomalyDetectionCommit,
  submitAnomalyDetectionPreview,
} from "../api/actions";
import { ApiError } from "../api/client";
import { createVisualization } from "../api/projectVisualizations";
import type { ActionDetail, ActionTypeMeta } from "../types";
import { ActionFlowChart } from "./ActionFlowChart";
import { HistogramChart, ROCChart, SpectrumChart } from "./DiagnosticCharts";

// --- Rich diagnostics (lazy-loaded from /actions/{id}/files/diagnostics.json) -

interface BandFilterDiagnostics {
  wavelengths_nm: number[];
  mean_spectrum: number[];
  std_spectrum: number[];
  p10_spectrum: number[];
  p50_spectrum: number[];
  p90_spectrum: number[];
}

interface IndexHistogram {
  counts: number[];
  edges: number[];
  min: number;
  max: number;
  n_valid: number;
}

interface SceneSegmentationDiagnostics {
  wavelengths_nm: number[];
  mean_spectrum_per_class: Record<string, number[]>;
  index_histograms: {
    ndvi: IndexHistogram;
    ndwi: IndexHistogram;
    brightness: IndexHistogram;
  };
  thresholds?: Record<string, number>;
}

function diagnosticsUrl(actionId: string): string {
  return `/api/actions/${encodeURIComponent(actionId)}/files/diagnostics.json`;
}

function useDiagnostics<T>(actionId: string, ready: boolean): {
  data: T | null;
  error: string | null;
} {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    setData(null);
    setError(null);
    fetch(diagnosticsUrl(actionId), { credentials: "include" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const j = (await res.json()) as T;
        if (!cancelled) setData(j);
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [actionId, ready]);
  return { data, error };
}

interface ActionDetailPaneProps {
  actionId: string | null;            // wire format action_<uuid> or null
  catalog: ActionTypeMeta[] | null;   // pre-fetched on workspace mount
}

const STATUS_COPY: Record<ActionDetail["status"], { label: string; tone: string }> = {
  queued:    { label: "Queued",    tone: "queued" },
  running:   { label: "Running",   tone: "running" },
  complete:  { label: "Complete",  tone: "complete" },
  failed:    { label: "Failed",    tone: "failed" },
  cancelled: { label: "Cancelled", tone: "cancelled" },
  // anomaly_detection_prep terminal state — worker is done, awaiting
  // user threshold choice in the viewer.
  needs_threshold: { label: "Needs threshold", tone: "needs_threshold" },
};

export function ActionDetailPane({ actionId, catalog }: ActionDetailPaneProps) {
  const [detail, setDetail] = useState<ActionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  // Fetch on actionId or refresh token change.
  useEffect(() => {
    if (!actionId) {
      setDetail(null);
      setError(null);
      return;
    }
    let cancelled = false;
    getAction(actionId)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setError(
            err.status === 404
              ? `Action ${actionId} not found.`
              : err.detail ?? `Error: HTTP ${err.status}`,
          );
        } else {
          setError("Could not reach the server.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [actionId, reloadToken]);

  // Poll while in flight.
  useEffect(() => {
    if (!detail) return;
    if (detail.status !== "queued" && detail.status !== "running") return;
    const id = window.setInterval(() => {
      setReloadToken((t) => t + 1);
    }, 3000);
    return () => window.clearInterval(id);
  }, [detail]);

  const meta = useMemo<ActionTypeMeta | null>(() => {
    if (!detail || !catalog) return null;
    return catalog.find((c) => c.type === detail.type) ?? null;
  }, [detail, catalog]);

  if (!actionId) {
    return (
      <section className="workspace__card workspace__card--center">
        <h3 className="workspace__card-title">Action detail</h3>
        <div className="workspace__empty workspace__empty--center">
          <p>
            Select an Action from the list to see its configuration and
            output here — in place, no new window, no navigation.
          </p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="workspace__card workspace__card--center">
        <h3 className="workspace__card-title">Action detail</h3>
        <p className="form__error" role="alert">{error}</p>
      </section>
    );
  }

  if (!detail) {
    return (
      <section className="workspace__card workspace__card--center">
        <h3 className="workspace__card-title">Action detail</h3>
        <div className="workspace__empty workspace__empty--center">
          <p>Loading…</p>
        </div>
      </section>
    );
  }

  return (
    <section className="workspace__card workspace__card--center">
      <ActionCard detail={detail} meta={meta} />
      <ActionOutputViewer detail={detail} meta={meta} />
    </section>
  );
}

// --- Action card -----------------------------------------------------

function ActionCard({
  detail,
  meta,
}: {
  detail: ActionDetail;
  meta: ActionTypeMeta | null;
}) {
  const status = STATUS_COPY[detail.status];
  return (
    <article className="action-card">
      <header className="action-card__header">
        <div className="action-card__head-left">
          <span className="action-card__slug">{detail.type}</span>
          <h2 className="action-card__title">
            {meta?.label ?? detail.type}
          </h2>
        </div>
        <span
          className="action-card__status"
          data-tone={status.tone}
        >
          ● {status.label}
        </span>
      </header>

      {meta?.description && (
        <p className="action-card__description">{meta.description}</p>
      )}

      <div className="action-card__flow">
        <div className="action-card__flow-title">
          Recipe
          <span className="action-card__flow-sub">
            What this action runs end-to-end
          </span>
        </div>
        <ActionFlowChart actionType={detail.type} height={360} />
      </div>

      <div className="action-card__io">
        {meta?.inputs && meta.inputs.length > 0 && (
          <div>
            <span className="action-card__io-title">Inputs</span>
            <ul>
              {meta.inputs.map((i) => (
                <li key={i.key}>
                  <strong>{i.label}</strong>{" "}
                  <code className="mono small">
                    {String(detail.configuration[i.key] ?? "—")}
                  </code>
                </li>
              ))}
            </ul>
          </div>
        )}
        {meta?.outputs && meta.outputs.length > 0 && (
          <div>
            <span className="action-card__io-title">Outputs</span>
            <ul>
              {meta.outputs.map((o) => (
                <li key={o.key}>
                  <strong>{o.label}</strong>{" "}
                  <span className="form__optional">({o.artifact_type})</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <details className="action-card__config">
        <summary>Configuration</summary>
        <pre className="action-card__config-body">
          {JSON.stringify(detail.configuration, null, 2)}
        </pre>
      </details>

      {detail.failure_reason && (
        <div className="action-card__failure">
          <span className="action-card__failure-title">Failure reason</span>
          <pre>{detail.failure_reason}</pre>
        </div>
      )}

      <div className="action-card__timestamps">
        <span>
          <strong>created</strong>{" "}
          <span className="mono small">{detail.created_at}</span>
        </span>
        <span>
          <strong>started</strong>{" "}
          <span className="mono small">{detail.started_at ?? "—"}</span>
        </span>
        <span>
          <strong>completed</strong>{" "}
          <span className="mono small">{detail.completed_at ?? "—"}</span>
        </span>
      </div>
    </article>
  );
}

// --- Output viewer dispatch -----------------------------------------

function ActionOutputViewer({
  detail,
  meta,
}: {
  detail: ActionDetail;
  meta: ActionTypeMeta | null;
}) {
  // Most actions only have a viewer after they hit "complete". The
  // anomaly_detection_prep type is the exception — its worker writes
  // the composite then waits for the user, so its viewer must render
  // while the action is still in "needs_threshold".
  const inViewableTerminalState =
    detail.status === "complete" ||
    (detail.status === "needs_threshold" &&
      detail.type === "anomaly_detection_prep");
  if (!detail.output || !inViewableTerminalState) {
    return (
      <div className="action-output-viewer action-output-viewer--placeholder">
        {detail.status === "queued" && (
          <p>Waiting for the worker to claim this Action…</p>
        )}
        {detail.status === "running" && (
          <p>The worker is running. Output will appear here when complete.</p>
        )}
        {detail.status === "failed" && (
          <p>Action failed — see the failure reason above.</p>
        )}
        {detail.status === "cancelled" && <p>Action was cancelled.</p>}
      </div>
    );
  }

  const summary = detail.output.summary;
  if (detail.type === "band_filter_apply") {
    return (
      <BandFilterApplyOutputViewer
        actionId={detail.id}
        summary={summary}
        meta={meta}
      />
    );
  }
  if (detail.type === "scene_segmentation") {
    return (
      <SceneSegmentationOutputViewer
        actionId={detail.id}
        summary={summary}
        meta={meta}
      />
    );
  }
  if (detail.type === "anomaly_scoring") {
    return (
      <AnomalyScoringOutputViewer
        actionId={detail.id}
        summary={summary}
        meta={meta}
      />
    );
  }
  if (detail.type === "cloud_mask") {
    return (
      <CloudMaskOutputViewer
        actionId={detail.id}
        summary={summary}
        meta={meta}
      />
    );
  }
  if (detail.type === "anomaly_detection_prep") {
    return (
      <AnomalyDetectionPrepViewer
        actionId={detail.id}
        summary={summary}
        meta={meta}
      />
    );
  }
  if (detail.type === "spectral_library_match") {
    return (
      <SpectralLibraryMatchOutputViewer
        actionId={detail.id}
        summary={summary}
      />
    );
  }
  // Generic fallback for future types.
  return (
    <div className="action-output-viewer">
      <h4>Output</h4>
      <pre className="action-card__config-body">
        {JSON.stringify(summary, null, 2)}
      </pre>
    </div>
  );
}

// --- Per-type viewers -----------------------------------------------

function fileUrl(actionId: string, filename: string): string {
  return `/api/actions/${encodeURIComponent(actionId)}/files/${encodeURIComponent(filename)}`;
}

function BandFilterApplyOutputViewer({
  actionId,
  summary,
  meta,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  meta: ActionTypeMeta | null;
}) {
  const m = summary as Record<string, unknown>;
  const previewSpec = meta?.outputs.find((o) => o.key === "preview");
  const { data: diag, error: diagError } =
    useDiagnostics<BandFilterDiagnostics>(actionId, true);

  return (
    <div className="action-output-viewer">
      <h4>Filtered vendable</h4>
      <div className="action-output-viewer__split">
        <div className="action-output-viewer__media">
          {previewSpec ? (
            <img
              src={fileUrl(actionId, previewSpec.filename)}
              alt="Filtered cube preview"
              className="action-output-viewer__image"
            />
          ) : (
            <p className="form__optional">No preview available.</p>
          )}
        </div>
        <dl className="action-output-viewer__stats">
          <dt>sensor</dt>
          <dd>{String(m.sensor ?? "—")}</dd>
          <dt>bands</dt>
          <dd>{String(m.band_count ?? "—")}</dd>
          <dt>shape</dt>
          <dd className="mono">{`${m.height ?? "?"} × ${m.width ?? "?"}`}</dd>
          <dt>total pixels</dt>
          <dd>{Number(m.total_pixels ?? 0).toLocaleString()}</dd>
          <dt>valid pixels</dt>
          <dd>{`${Number(m.valid_pixels ?? 0).toLocaleString()} (${m.spatial_validity_pct ?? "—"}%)`}</dd>
          <dt>λ range (nm)</dt>
          <dd className="mono">{`${m.wavelengths_min_nm ?? "?"}–${m.wavelengths_max_nm ?? "?"}`}</dd>
          <dt>common grid</dt>
          <dd>{m.common_wavelength_grid_applied ? "yes" : "no"}</dd>
          <dt>nearest-valid fill</dt>
          <dd>{m.nearest_valid_fill_applied ? "applied" : "skipped"}</dd>
          <dt>quality masks</dt>
          <dd className="mono small">
            {Array.isArray(m.quality_masks_applied) && m.quality_masks_applied.length > 0
              ? (m.quality_masks_applied as string[]).join(", ")
              : "—"}
          </dd>
          <dt>pickle size</dt>
          <dd>{`${m.pickle_size_mb ?? "?"} MB`}</dd>
        </dl>
      </div>

      <BandFilterApplyDiagnosticsCharts diag={diag} error={diagError} />
    </div>
  );
}

function BandFilterApplyDiagnosticsCharts({
  diag,
  error,
}: {
  diag: BandFilterDiagnostics | null;
  error: string | null;
}) {
  if (error) {
    return (
      <p className="form__optional small">
        Diagnostics file unavailable ({error}). The charts are produced by
        runs from the latest worker — older outputs only carry the lean
        summary above.
      </p>
    );
  }
  if (!diag) {
    return <p className="scene-detail__hint">Loading diagnostics…</p>;
  }
  // Guard against legacy diagnostics that lack the spectral arrays.
  const wls = Array.isArray(diag.wavelengths_nm) ? diag.wavelengths_nm : null;
  const mean = Array.isArray(diag.mean_spectrum) ? diag.mean_spectrum : null;
  const p10 = Array.isArray(diag.p10_spectrum) ? diag.p10_spectrum : null;
  const p50 = Array.isArray(diag.p50_spectrum) ? diag.p50_spectrum : null;
  const p90 = Array.isArray(diag.p90_spectrum) ? diag.p90_spectrum : null;
  const ok =
    wls &&
    mean &&
    p10 &&
    p50 &&
    p90 &&
    wls.length > 0 &&
    mean.length === wls.length &&
    p10.length === wls.length &&
    p50.length === wls.length &&
    p90.length === wls.length;
  if (!ok) {
    return (
      <p className="form__optional small">
        Spectral diagnostics not available for this output (likely an
        older run). Re-run the action to get the mean / p10 / p90 spectrum.
      </p>
    );
  }
  return (
    <div className="action-output-viewer__chart">
      <div className="action-output-viewer__chart-title">
        Mean spectrum across valid pixels
        <span className="action-output-viewer__chart-sub">
          dashed = p10 / median / p90 across the scene
        </span>
      </div>
      <SpectrumChart
        wavelengths={wls}
        yLabel="reflectance"
        series={[
          { label: "p10", color: "#94a3b8", values: p10, dashed: true },
          { label: "p90", color: "#94a3b8", values: p90, dashed: true },
          { label: "mean", color: "#1f5f3d", values: mean },
          { label: "median", color: "#3b82f6", values: p50, dashed: true },
        ]}
      />
    </div>
  );
}

// Tuned for white canvas — every line stays visible against the
// background and against the others.
const CLASS_COLORS: Record<string, string> = {
  water:      "#2563eb",   // saturated blue
  cloud:      "#0ea5e9",   // sky — visible on white, distinct from water
  shadow:     "#1f2937",   // near-black so the line reads on top of the others
  vegetation: "#059669",   // emerald
  kept:       "#1f5f3d",   // brand forest — the consensus / kept line
};

function SceneSegmentationOutputViewer({
  actionId,
  summary,
  meta,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  meta: ActionTypeMeta | null;
}) {
  const m = summary as Record<string, unknown>;
  const counts = (m.pixel_counts ?? {}) as Record<string, number>;
  const thresholds = (m.thresholds ?? {}) as Record<string, number>;
  const previewSpec = meta?.outputs.find((o) => o.key === "preview");
  const total = counts.spatial_valid ?? 0;
  const pct = (n: number) =>
    total > 0 ? `${((n / total) * 100).toFixed(1)}%` : "—";

  const { data: diag, error: diagError } =
    useDiagnostics<SceneSegmentationDiagnostics>(actionId, true);

  return (
    <div className="action-output-viewer">
      <h4>Scene segmentation</h4>
      <div className="action-output-viewer__media action-output-viewer__media--full">
        {previewSpec ? (
          <img
            src={fileUrl(actionId, previewSpec.filename)}
            alt="Indices + mask montage"
            className="action-output-viewer__image"
          />
        ) : (
          <p className="form__optional">No preview available.</p>
        )}
      </div>
      {/* Class table sits full-width below the preview so the threshold
          column has room to breathe — the side-by-side split was crushing
          it into multi-line wraps. */}
      <div className="action-output-viewer__class-block">
        <table className="action-output-viewer__class-table">
          <thead>
            <tr>
              <th>Class</th>
              <th className="num">Pixels</th>
              <th className="num">% of valid</th>
              <th>Threshold</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Water</td>
              <td className="num">{(counts.water ?? 0).toLocaleString()}</td>
              <td className="num">{pct(counts.water ?? 0)}</td>
              <td className="mono small">NDWI &gt; {thresholds.ndwi_water ?? "?"}</td>
            </tr>
            <tr>
              <td>Cloud</td>
              <td className="num">{(counts.cloud ?? 0).toLocaleString()}</td>
              <td className="num">{pct(counts.cloud ?? 0)}</td>
              <td className="mono small">brightness &gt; {thresholds.brightness_cloud ?? "?"}</td>
            </tr>
            <tr>
              <td>Shadow</td>
              <td className="num">{(counts.shadow ?? 0).toLocaleString()}</td>
              <td className="num">{pct(counts.shadow ?? 0)}</td>
              <td className="mono small">brightness &lt; {thresholds.brightness_shadow ?? "?"}</td>
            </tr>
            <tr>
              <td>Vegetation</td>
              <td className="num">{(counts.vegetation ?? 0).toLocaleString()}</td>
              <td className="num">{pct(counts.vegetation ?? 0)}</td>
              <td className="mono small">NDVI &gt; {thresholds.ndvi_vegetation ?? "?"}</td>
            </tr>
            <tr className="action-output-viewer__class-row--total">
              <td><strong>Kept</strong></td>
              <td className="num"><strong>{(counts.kept ?? 0).toLocaleString()}</strong></td>
              <td className="num"><strong>{pct(counts.kept ?? 0)}</strong></td>
              <td className="mono small">union of selected classes</td>
            </tr>
          </tbody>
        </table>
        <p className="action-output-viewer__hint">
          Downstream HSI Actions consume <code>keep_mask.tif</code> as
          their scoring domain.
        </p>
      </div>

      <SceneSegmentationDiagnosticsCharts
        diag={diag}
        error={diagError}
        thresholds={thresholds}
        counts={counts}
      />
    </div>
  );
}

function SceneSegmentationDiagnosticsCharts({
  diag,
  error,
  thresholds,
  counts,
}: {
  diag: SceneSegmentationDiagnostics | null;
  error: string | null;
  thresholds: Record<string, number>;
  counts: Record<string, number>;
}) {
  if (error) {
    return (
      <p className="form__optional small">
        Diagnostics file unavailable ({error}). The charts are produced
        by runs from the latest worker — older outputs only carry the
        lean summary above.
      </p>
    );
  }
  if (!diag) {
    return <p className="scene-detail__hint">Loading diagnostics…</p>;
  }
  const wls = Array.isArray(diag.wavelengths_nm) ? diag.wavelengths_nm : null;
  const spectra = (diag.mean_spectrum_per_class ?? null) as
    | Record<string, number[]>
    | null;
  const hists = (diag.index_histograms ?? null) as
    | SceneSegmentationDiagnostics["index_histograms"]
    | null;
  const haveSpectra =
    wls && spectra && Object.keys(spectra).length > 0 &&
    Object.values(spectra).every(
      (v) => Array.isArray(v) && v.length === wls.length,
    );
  const haveHists =
    hists &&
    hists.ndvi &&
    hists.ndwi &&
    hists.brightness &&
    Array.isArray(hists.ndvi.counts) &&
    Array.isArray(hists.ndwi.counts) &&
    Array.isArray(hists.brightness.counts);

  if (!haveSpectra && !haveHists) {
    return (
      <p className="form__optional small">
        Per-class spectra and index histograms not available for this
        output (older run). Re-run the action to get the rich charts.
      </p>
    );
  }

  return (
    <>
      {haveSpectra && wls && spectra && (() => {
        // Derive a stable y-max from the kept spectrum so small noisy
        // classes (shadow with 1k pixels, etc.) can't blow the scale.
        const kept = spectra.kept ?? [];
        const keptMax = kept.length ? Math.max(...kept) : 0.4;
        const yMax = Math.min(0.9, Math.max(0.25, keptMax * 1.6));
        return (
          <div className="action-output-viewer__chart">
            <div className="action-output-viewer__chart-title">
              Mean spectrum per class
              <span className="action-output-viewer__chart-sub">
                y-axis capped at {yMax.toFixed(2)} (1.6× kept-class max) so
                small-pixel-count classes don't dominate
              </span>
            </div>
            <SpectrumChart
              wavelengths={wls}
              yLabel="reflectance"
              yMin={0}
              yMax={yMax}
              series={Object.entries(spectra)
                .filter(([cls]) => (counts[cls] ?? 0) > 0 || cls === "kept")
                .map(([cls, values]) => ({
                  label: cls,
                  color: CLASS_COLORS[cls] ?? "#94a3b8",
                  values,
                  dashed: cls !== "kept",
                }))}
            />
          </div>
        );
      })()}

      {haveHists && hists && (
        <>
          <div className="action-output-viewer__hist-grid">
            <HistogramChart
              edges={hists.ndvi.edges}
              counts={hists.ndvi.counts}
              color="#10b981"
              label="NDVI"
              threshold={thresholds.ndvi_vegetation}
            />
            <HistogramChart
              edges={hists.ndwi.edges}
              counts={hists.ndwi.counts}
              color="#3b82f6"
              label="NDWI"
              threshold={thresholds.ndwi_water}
            />
            <HistogramChart
              edges={hists.brightness.edges}
              counts={hists.brightness.counts}
              color="#f59e0b"
              label="brightness"
              threshold={thresholds.brightness_cloud}
            />
          </div>
          <p className="action-output-viewer__hint small">
            Dashed red lines = thresholds in use. Anything to the
            class-side of the line was masked out of <code>keep_mask</code>.
          </p>
        </>
      )}
    </>
  );
}

// === Cloud mask output viewer ========================================

interface CloudMaskDiagnostics {
  scene_shape: [number, number];
  valid_pixels: number;
  cloud_pixels: number;
  kept_pixels: number;
  cloud_pct_of_valid: number;
  kept_pct_of_valid: number;
  sampling_ratio: number;
  n_components: number;
  fit_seconds: number;
  predict_seconds: number;
  gmm_anchors_celsius: number[] | null;
  probe_percentiles_celsius: {
    p2: number | null;
    p8: number | null;
    p50: number | null;
    p92: number | null;
    p98: number | null;
  } | null;
  gmm_cluster_means_celsius: number[] | null;
  sample_count: number;
}

function CloudMaskOutputViewer({
  actionId,
  summary,
  meta,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  meta: ActionTypeMeta | null;
}) {
  const m = summary as Record<string, unknown>;
  const previewSpec = meta?.outputs.find((o) => o.key === "preview");
  const sceneShape = (m.scene_shape as number[] | undefined) ?? [0, 0];

  const [modalOpen, setModalOpen] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (modalOpen) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") {
        e.preventDefault();
        setModalOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  return (
    <div className="action-output-viewer">
      <h4>Cloud mask</h4>
      <div className="anomaly-viewer__summary-card">
        <button
          type="button"
          className="anomaly-viewer__thumb-btn"
          onClick={() => setModalOpen(true)}
          aria-label="Open full-screen result"
        >
          {previewSpec ? (
            <img
              src={fileUrl(actionId, previewSpec.filename)}
              alt="B10 grayscale with cyan cloud overlay"
              className="anomaly-viewer__thumb-img"
            />
          ) : (
            <span className="anomaly-viewer__thumb-fallback">no preview</span>
          )}
          <span className="anomaly-viewer__thumb-overlay">
            <span className="anomaly-viewer__thumb-icon" aria-hidden="true">
              ⤢
            </span>
            Open result
          </span>
        </button>
        <div className="anomaly-viewer__summary-meta">
          <p className="anomaly-viewer__summary-line">
            <strong>{Number(m.cloud_pixels ?? 0).toLocaleString()}</strong> cloud px ·{" "}
            <strong>{String(m.cloud_pct_of_valid ?? "—")}%</strong> of valid
          </p>
          <p className="anomaly-viewer__summary-line small">
            scene {`${sceneShape[0]} × ${sceneShape[1]}`} · valid{" "}
            {Number(m.valid_pixels ?? 0).toLocaleString()} · sampling{" "}
            {String(m.sampling_ratio ?? "—")}
          </p>
          <p className="anomaly-viewer__summary-line small">
            fit {Number(m.fit_seconds ?? 0).toFixed(1)}s · predict{" "}
            {Number(m.predict_seconds ?? 0).toFixed(1)}s · {String(m.n_components ?? "—")} GMM
            components
          </p>
          <button
            type="button"
            className="btn anomaly-viewer__open-btn"
            onClick={() => setModalOpen(true)}
          >
            Open result viewer
            <span className="anomaly-viewer__shortcut">⌘I</span>
          </button>
        </div>
      </div>

      {modalOpen && (
        <CloudMaskResultModal
          actionId={actionId}
          summary={m}
          previewFilename={previewSpec?.filename ?? null}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}

function CloudMaskResultModal({
  actionId,
  summary,
  previewFilename,
  onClose,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  previewFilename: string | null;
  onClose: () => void;
}) {
  const sceneShape = (summary.scene_shape as number[] | undefined) ?? [0, 0];
  const { data: diag, error: diagError } =
    useDiagnostics<CloudMaskDiagnostics>(actionId, true);

  useEffect(() => {
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  return (
    <div
      className="anomaly-viewer__modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Cloud mask result"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <section className="anomaly-viewer__modal">
        <header className="anomaly-viewer__modal-header">
          <div>
            <h3>
              Cloud mask · {Number(summary.cloud_pixels ?? 0).toLocaleString()} cloud px ·{" "}
              {String(summary.cloud_pct_of_valid ?? "—")}% of valid
            </h3>
            <p className="anomaly-viewer__modal-sub small">
              scene {`${sceneShape[0]} × ${sceneShape[1]}`} · sampling{" "}
              {String(summary.sampling_ratio ?? "—")} · {String(summary.n_components ?? "—")} GMM
              components · fit {Number(summary.fit_seconds ?? 0).toFixed(1)}s · predict{" "}
              {Number(summary.predict_seconds ?? 0).toFixed(1)}s
            </p>
          </div>
          <button
            type="button"
            className="anomaly-viewer__modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        <div className="anomaly-viewer__modal-body">
          <div className="cloud-viewer__panel">
            <ZoomablePanel
              src={
                previewFilename
                  ? `/api/actions/${encodeURIComponent(actionId)}/files/${encodeURIComponent(previewFilename)}`
                  : ""
              }
              title="B10 grayscale + cyan cloud overlay"
              colormapHint="2nd–98th percentile stretch · α=0.55"
            />
          </div>
          <CloudMaskDiagnosticsBlock diag={diag} error={diagError} />
          <p className="action-output-viewer__hint small">
            Downstream thermal <code>anomaly_scoring</code> Actions can
            consume this via the <em>Keep mask · cloud mask</em> picker;
            scoring will be restricted to{" "}
            <code>pure_validity ∧ ¬cloud</code>.
          </p>
        </div>
      </section>
    </div>
  );
}

function CloudMaskDiagnosticsBlock({
  diag,
  error,
}: {
  diag: CloudMaskDiagnostics | null;
  error: string | null;
}) {
  if (error) {
    return (
      <p className="form__optional small">
        Diagnostics file unavailable ({error}).
      </p>
    );
  }
  if (!diag) return <p className="scene-detail__hint">Loading diagnostics…</p>;
  const probe = diag.probe_percentiles_celsius;
  const anchors = diag.gmm_anchors_celsius ?? [];
  const means = diag.gmm_cluster_means_celsius ?? [];
  return (
    <div className="action-output-viewer__chart">
      <div className="action-output-viewer__chart-title">
        GMM fit details
        <span className="action-output-viewer__chart-sub">
          temperatures in °C · cold clusters are masked as cloud
        </span>
      </div>
      <table className="action-output-viewer__class-table">
        <thead>
          <tr>
            <th>section</th>
            <th>values</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>probe percentiles</td>
            <td className="mono small">
              {probe
                ? `P2 ${probe.p2?.toFixed(2)} · P8 ${probe.p8?.toFixed(2)} · P50 ${probe.p50?.toFixed(2)} · P92 ${probe.p92?.toFixed(2)} · P98 ${probe.p98?.toFixed(2)}`
                : "—"}
            </td>
          </tr>
          <tr>
            <td>anchor means</td>
            <td className="mono small">
              {anchors.length
                ? anchors.map((a) => a.toFixed(2)).join(" · ")
                : "—"}
            </td>
          </tr>
          <tr>
            <td>fitted cluster means</td>
            <td className="mono small">
              {means.length
                ? means.map((c) => c.toFixed(2)).join(" · ")
                : "—"}
            </td>
          </tr>
          <tr>
            <td>training sample size</td>
            <td className="mono small">
              {diag.sample_count.toLocaleString()}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// === Anomaly scoring output viewer ===================================

interface AnomalyROC {
  fpr: number[];
  tpr: number[];
  auc: number;
}

interface AnomalyScoringDiagnostics {
  // Top-level normalisation provenance. "baked" = foundation model
  // used its training pixel stats; "per_scene_dn_zscore" = action
  // overrode the stats with per-scene mean/std (HotSat-style
  // uncalibrated DN). Older diagnostics (pre-this-field) omit it.
  normalization_mode?: string;
  scene_units?: string | null;
  per_model_full: Array<{
    codename: string;
    architecture: string;
    sensor: string;
    method: string;
    patch_size: number;
    stride: number;
    batch_size?: number;
    sam_l1_alpha?: number | null;
    device: string;
    load_seconds: number;
    infer_seconds: number;
    score_min: number;
    score_max: number;
    score_mean: number;
    score_percentiles: Record<string, number>;
    auc: number | null;
    // Per-model normalisation provenance — present from the slice
    // that added per-scene DN overrides. "n/a" for classical
    // detectors.
    normalization_mode?: string;
  }>;
  roc: Record<string, AnomalyROC> | null;
}

// Per-model swatch palette for multi-model charts (ROC, score
// distributions). First two colors lead with the brand-accent green
// + warm amber so charts read on-brand; the rest cycle through
// distinguishable hues (blue / orange / pink / cyan / yellow) for
// runs with many models. Avoid pure purple here so it doesn't fight
// the new accent.
const AD_MODEL_COLORS = [
  "#1f5f3d",   // brand forest green
  "#b18a2c",   // brand amber
  "#3b82f6",   // blue
  "#f97316",   // orange
  "#0d7490",   // teal (matches VNIR chip)
  "#ec4899",   // pink
  "#facc15",   // yellow
];

function outputUrl(actionId: string, relpath: string): string {
  // Path-segment-encode but keep the `/` separators so the FastAPI
  // `:path` route receives a structured relpath instead of `%2F`.
  const safe = relpath
    .split("/")
    .map((s) => encodeURIComponent(s))
    .join("/");
  return `/api/actions/${encodeURIComponent(actionId)}/output/${safe}`;
}

function AnomalyScoringOutputViewer({
  actionId,
  summary,
  meta,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  meta: ActionTypeMeta | null;
}) {
  const m = summary as Record<string, unknown>;
  const previewSpec = meta?.outputs.find((o) => o.key === "preview");
  const perModel = Array.isArray(m.per_model)
    ? (m.per_model as Array<Record<string, unknown>>)
    : [];
  const sceneShape = (m.scene_shape as number[] | undefined) ?? [0, 0];

  const [modalOpen, setModalOpen] = useState(false);
  // Cmd/Ctrl+I as a power-user shortcut; Esc closes (handled in modal).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (modalOpen) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") {
        e.preventDefault();
        setModalOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  return (
    <div className="action-output-viewer">
      <h4>Anomaly scoring</h4>
      <div className="anomaly-viewer__summary-card">
        <button
          type="button"
          className="anomaly-viewer__thumb-btn"
          onClick={() => setModalOpen(true)}
          aria-label="Open full-screen result"
        >
          {previewSpec ? (
            <img
              src={fileUrl(actionId, previewSpec.filename)}
              alt="Heatmap montage thumbnail"
              className="anomaly-viewer__thumb-img"
            />
          ) : (
            <span className="anomaly-viewer__thumb-fallback">no preview</span>
          )}
          <span className="anomaly-viewer__thumb-overlay">
            <span className="anomaly-viewer__thumb-icon" aria-hidden="true">
              ⤢
            </span>
            Open result
          </span>
        </button>
        <div className="anomaly-viewer__summary-meta">
          <p className="anomaly-viewer__summary-line">
            <strong>
              {String(m.n_models ?? perModel.length)} model
              {perModel.length === 1 ? "" : "s"}
            </strong>{" "}
            · scene {`${sceneShape[0]} × ${sceneShape[1]}`} ·{" "}
            {String(m.band_count ?? "?")} bands
          </p>
          <p className="anomaly-viewer__summary-line small">
            kept {String(m.kept_pct ?? "—")}% of valid · device{" "}
            <code>{String(m.device ?? "?")}</code>
          </p>
          <ul className="anomaly-viewer__summary-models">
            {perModel.map((row) => {
              const cname = String(row.codename ?? "?");
              const auc = row.auc;
              return (
                <li key={cname}>
                  <strong>{cname}</strong>{" "}
                  <span className="mono small">
                    {String(row.method ?? "")}
                  </span>
                  {typeof auc === "number" && (
                    <span className="mono small">
                      {" "}
                      · AUC {auc.toFixed(3)}
                    </span>
                  )}
                  <span className="mono small">
                    {" "}
                    · {Number(row.infer_seconds ?? 0).toFixed(1)}s
                  </span>
                </li>
              );
            })}
          </ul>
          <button
            type="button"
            className="btn anomaly-viewer__open-btn"
            onClick={() => setModalOpen(true)}
          >
            Open result viewer
            <span className="anomaly-viewer__shortcut">⌘I</span>
          </button>
        </div>
      </div>

      {modalOpen && (
        <AnomalyScoringResultModal
          actionId={actionId}
          summary={m}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}

function AnomalyScoringResultModal({
  actionId,
  summary,
  onClose,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  onClose: () => void;
}) {
  const perModel = Array.isArray(summary.per_model)
    ? (summary.per_model as Array<Record<string, unknown>>)
    : [];
  const sceneShape = (summary.scene_shape as number[] | undefined) ?? [0, 0];
  const codenames = perModel.map((r) => String(r.codename ?? ""));
  const [activeCodename, setActiveCodename] = useState<string | null>(
    codenames[0] ?? null,
  );
  const { data: diag, error: diagError } =
    useDiagnostics<AnomalyScoringDiagnostics>(actionId, true);

  useEffect(() => {
    if (!activeCodename && codenames.length > 0) setActiveCodename(codenames[0]);
  }, [codenames, activeCodename]);

  // Esc closes; lock body scroll while open.
  useEffect(() => {
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  const activeRow = perModel.find((r) => r.codename === activeCodename);
  const fullStats = diag?.per_model_full.find((r) => r.codename === activeCodename);

  return (
    <div
      className="anomaly-viewer__modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Anomaly scoring result"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <section className="anomaly-viewer__modal">
        <header className="anomaly-viewer__modal-header">
          <div>
            <h3>Anomaly scoring · {String(summary.n_models ?? perModel.length)} model
              {perModel.length === 1 ? "" : "s"}</h3>
            <p className="anomaly-viewer__modal-sub small">
              scene {`${sceneShape[0]} × ${sceneShape[1]}`} ·{" "}
              {String(summary.band_count ?? "?")} bands · kept{" "}
              {String(summary.kept_pct ?? "—")}% · device{" "}
              <code>{String(summary.device ?? "?")}</code>
            </p>
          </div>
          <button
            type="button"
            className="anomaly-viewer__modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        {diag?.normalization_mode === "per_scene_dn_zscore" && (
          <div
            className="anomaly-viewer__norm-banner"
            role="note"
            aria-label="Per-scene normalisation in effect"
          >
            <strong>Per-scene normalisation</strong>: this sensor ships
            uncalibrated values
            {diag.scene_units ? (
              <> (<code>{diag.scene_units}</code>)</>
            ) : null}
            , so the foundation model's baked pixel-stats were replaced
            with per-scene mean and std. Scores are <em>scene-relative</em>{" "}
            — meaningful within this scene but not directly comparable
            across scenes.
          </div>
        )}

        {codenames.length > 1 && (
          <div className="anomaly-viewer__tabs" role="tablist">
            {perModel.map((row) => {
              const cname = String(row.codename ?? "?");
              return (
                <button
                  key={cname}
                  type="button"
                  role="tab"
                  aria-selected={cname === activeCodename}
                  className="anomaly-viewer__tab"
                  data-active={cname === activeCodename ? "true" : "false"}
                  onClick={() => setActiveCodename(cname)}
                >
                  <span className="anomaly-viewer__tab-name">{cname}</span>
                  <span className="anomaly-viewer__tab-method">
                    {String(row.method ?? "")}
                    {typeof row.auc === "number"
                      ? ` · AUC=${(row.auc as number).toFixed(3)}`
                      : ""}
                  </span>
                </button>
              );
            })}
          </div>
        )}

        <div className="anomaly-viewer__modal-body">
          {activeCodename && activeRow && (
            <AnomalyScoringTriPanel
              actionId={actionId}
              codename={activeCodename}
              method={String(activeRow.method ?? "")}
              stats={fullStats ?? null}
              hasGt={Boolean(summary.has_gt)}
              sceneId={(summary.scene_id as string | undefined) ?? null}
              sensorType={(summary.sensor_type as string | undefined) ?? null}
              sceneShape={
                Array.isArray(summary.scene_shape) &&
                summary.scene_shape.length === 2
                  ? [
                      Number(summary.scene_shape[0]),
                      Number(summary.scene_shape[1]),
                    ]
                  : null
              }
            />
          )}
          <AnomalyScoringROCBlock
            diag={diag}
            error={diagError}
            hasGt={Boolean(summary.has_gt)}
          />
        </div>
      </section>
    </div>
  );
}

function AnomalyScoringTriPanel({
  actionId,
  codename,
  method,
  stats,
  hasGt,
  sceneId,
  sensorType,
  sceneShape,
}: {
  actionId: string;
  codename: string;
  method: string;
  stats: AnomalyScoringDiagnostics["per_model_full"][number] | null;
  hasGt: boolean;
  sceneId: string | null;
  sensorType: string | null;
  sceneShape: [number, number] | null;
}) {
  const codenameSlug = codename.toLowerCase().replace(/\s+/g, "_");
  const rgbSrc = outputUrl(actionId, "rgb.png");
  const reconSrc = outputUrl(actionId, `models/${codenameSlug}/reconstruction.png`);
  const scoreSrc = outputUrl(actionId, `models/${codenameSlug}/anomaly_score.png`);

  // Linked-pan/zoom controller: each panel registers its panzoom instance
  // + image element; when any panel's transform changes, the controller
  // broadcasts the same {x,y,scale} to the others. The `applying` flag
  // suppresses feedback while we're the one writing.
  const linker = useLinkedPanzoom();
  const [areaZoom, setAreaZoom] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  // GT dots — toggle + lazy-loaded sidecar. Only meaningful when the
  // action was submitted with input_annotation_id (summary.has_gt).
  const [showGt, setShowGt] = useState(false);
  const [gtDots, setGtDots] = useState<Array<[number, number]> | null>(null);
  const [gtError, setGtError] = useState<string | null>(null);
  useEffect(() => {
    if (!showGt || gtDots !== null) return;
    let cancelled = false;
    fetch(`/api/actions/${encodeURIComponent(actionId)}/files/gt_dots.json`, {
      credentials: "include",
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<{
          pixels: Array<[number, number]>;
        }>;
      })
      .then((d) => {
        if (!cancelled) setGtDots(d.pixels);
      })
      .catch(() => {
        if (!cancelled) {
          setGtError(
            "GT dots unavailable — action may pre-date the gt_dots.json sidecar. Re-run with the same GT annotation to populate.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [showGt, gtDots, actionId]);

  // Spectrum probe — opened when a dot is clicked.
  const [probeAt, setProbeAt] = useState<{ row: number; col: number } | null>(
    null,
  );

  const onSaveView = async () => {
    const defaultName = `${codename} · ${method}`;
    const name = window.prompt(
      "Name this visualization:",
      defaultName,
    );
    if (!name) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      // Resolve action -> project_id via /actions/{id} (cheap, cached server-side).
      const act = await getAction(actionId);
      // Compose the three currently-visible panels into one PNG, taking
      // their CURRENT panzoom transform into account so the saved frame
      // matches what the user is looking at.
      const blob = await composeTriPanelBlob(linker);
      if (!blob) {
        throw new Error("compose_failed");
      }
      await createVisualization({
        projectId: act.project_id,
        source: { kind: "action_output", action_id: actionId },
        name,
        description: undefined,
        viewState: { codename, method, area_zoom: areaZoom },
        imageBlob: blob,
      });
      setSaveMsg("Saved. See the Visualizations tab.");
      window.dispatchEvent(new CustomEvent("allotrope:viz-saved"));
    } catch (err) {
      setSaveMsg(
        err instanceof ApiError
          ? `Save failed: ${err.detail ?? err.status}`
          : `Save failed: ${(err as Error).message}`,
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <div className="anomaly-viewer__toolbar">
        <button
          type="button"
          className="anomaly-viewer__tool-btn"
          data-active={areaZoom ? "true" : "false"}
          onClick={() => setAreaZoom(v => !v)}
          title="Drag a box on any panel to zoom all three to that region"
        >
          {areaZoom ? "Area-zoom: ON" : "Area-zoom"}
        </button>
        <button
          type="button"
          className="anomaly-viewer__tool-btn"
          onClick={() => linker.resetAll()}
          title="Reset all panels to 1×"
        >
          Reset
        </button>
        <button
          type="button"
          className="anomaly-viewer__tool-btn"
          onClick={() => void onSaveView()}
          disabled={saving}
          title="Save the current 3-panel view to this project"
        >
          {saving ? "Saving…" : "Save view"}
        </button>
        {hasGt && (
          <button
            type="button"
            className="anomaly-viewer__tool-btn"
            data-active={showGt ? "true" : "false"}
            onClick={() => setShowGt((v) => !v)}
            title="Toggle cyan dots at ground-truth anomaly pixels — click a dot to inspect its spectrum"
          >
            {showGt ? "GT dots: ON" : "Show GT dots"}
          </button>
        )}
        {saveMsg && (
          <span className="anomaly-viewer__tool-hint small">{saveMsg}</span>
        )}
        {gtError && showGt && (
          <span className="anomaly-viewer__tool-hint small" style={{ color: "#b91c1c" }}>
            {gtError}
          </span>
        )}
        <span className="anomaly-viewer__tool-hint small">
          Scroll = zoom · drag = pan · double-click = reset · all three panels move together
        </span>
      </div>
      <div className="anomaly-viewer__panels">
        <ZoomablePanel
          src={rgbSrc}
          title="RGB scene"
          linker={linker}
          panelKey="rgb"
          areaZoom={areaZoom}
          gtDots={showGt ? gtDots ?? undefined : undefined}
          sceneShape={sceneShape ?? undefined}
          onDotClick={(row, col) => setProbeAt({ row, col })}
        />
        <ZoomablePanel
          src={reconSrc}
          title={`${codename} reconstruction`}
          linker={linker}
          panelKey="recon"
          areaZoom={areaZoom}
          gtDots={showGt ? gtDots ?? undefined : undefined}
          sceneShape={sceneShape ?? undefined}
          onDotClick={(row, col) => setProbeAt({ row, col })}
        />
        <ZoomablePanel
          src={scoreSrc}
          title={`${codename} score (${method})`}
          colormapHint="inferno · 99.5%-percentile cap"
          linker={linker}
          panelKey="score"
          areaZoom={areaZoom}
          gtDots={showGt ? gtDots ?? undefined : undefined}
          sceneShape={sceneShape ?? undefined}
          onDotClick={(row, col) => setProbeAt({ row, col })}
        />
      </div>
      {probeAt && sceneId && (
        <SpectrumProbeModal
          sceneId={sceneId}
          sensorType={sensorType}
          row={probeAt.row}
          col={probeAt.col}
          onClose={() => setProbeAt(null)}
        />
      )}
      {stats && (
        <table className="action-output-viewer__class-table anomaly-viewer__stats">
          <tbody>
            <tr>
              <th>method</th>
              <td className="mono">{stats.method}</td>
              <th>patch · stride · batch</th>
              <td className="mono">
                {stats.patch_size} · {stats.stride} · {stats.batch_size ?? "—"}
              </td>
            </tr>
            <tr>
              <th>score min · mean · max</th>
              <td className="mono">
                {stats.score_min.toFixed(4)} · {stats.score_mean.toFixed(4)} ·{" "}
                {stats.score_max.toFixed(4)}
              </td>
              <th>p95 · p99 · p99.9</th>
              <td className="mono">
                {(stats.score_percentiles.p95 ?? 0).toFixed(4)} ·{" "}
                {(stats.score_percentiles.p99 ?? 0).toFixed(4)} ·{" "}
                {(stats.score_percentiles.p99_9 ?? 0).toFixed(4)}
              </td>
            </tr>
            <tr>
              <th>load · infer (s)</th>
              <td className="mono">
                {stats.load_seconds.toFixed(1)} · {stats.infer_seconds.toFixed(1)}
              </td>
              <th>device · arch</th>
              <td className="mono">
                <code>{stats.device}</code> · <code>{stats.architecture}</code>
              </td>
            </tr>
            <tr>
              <th>erosion · keep_mask erosion</th>
              <td className="mono">
                {(stats as unknown as { erosion_kernel_size?: number | null }).erosion_kernel_size ?? "—"}{" "}
                ·{" "}
                {(stats as unknown as { keep_mask_erosion_kernel_size?: number }).keep_mask_erosion_kernel_size ?? 1}
              </td>
              <th />
              <td />
            </tr>
          </tbody>
        </table>
      )}
      <p className="action-output-viewer__hint small">
        Per-model rasters at <code>models/{codenameSlug}/anomaly_score.tif</code> and{" "}
        <code>reconstruction.tif</code>.
      </p>
    </>
  );
}

// --- Linked panzoom controller -----------------------------------------
//
// All three anomaly-viewer panels share a single {x, y, scale} transform.
// Each panel registers its panzoom instance + image element with the
// controller. When any panel emits a `transform` event, the controller
// pushes the same transform onto the other panels, with a re-entry guard
// so the broadcast doesn't ping-pong.
//
// Area-zoom: while the toolbar toggle is on, drag a rectangle on any
// panel — on mouseup the controller zooms all three to the rect in image
// coordinates (so a region of interest stays aligned across RGB / recon /
// score regardless of which panel it was drawn on).

type PzInstance = {
  dispose: () => void;
  getTransform: () => { x: number; y: number; scale: number };
  moveTo: (x: number, y: number) => void;
  zoomAbs: (x: number, y: number, scale: number) => void;
  on: (evt: string, fn: () => void) => void;
  off?: (evt: string, fn: () => void) => void;
  pause?: () => void;
  resume?: () => void;
};

interface PanelEntry {
  pz: PzInstance;
  img: HTMLImageElement;
}

interface LinkedPanzoom {
  register: (key: string, entry: PanelEntry) => () => void;
  broadcast: (from: string, t: { x: number; y: number; scale: number }) => void;
  zoomToImageRect: (rect: { x: number; y: number; w: number; h: number }) => void;
  resetAll: () => void;
}

// Render the three registered panels into a single triptych PNG. Each
// panel's currently-visible region (after panzoom transform) is drawn
// into a same-aspect canvas tile. Returns a PNG blob suitable for upload
// or `null` if no panels are registered.
async function composeTriPanelBlob(
  linker: LinkedPanzoom,
): Promise<Blob | null> {
  // Cheat-ish access — we don't expose the panels map publicly, so we
  // pluck the entries via the same helper the linker uses. The panel
  // order in JSX is rgb / recon / score; rely on insertion order, which
  // Map preserves.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const entries: Array<[string, { pz: PzInstance; img: HTMLImageElement }]> =
    (linker as unknown as { _entries?: () => unknown })._entries
      ? // hot path for tests
        ((linker as unknown as { _entries: () => Iterable<[string, { pz: PzInstance; img: HTMLImageElement }]> })
          ._entries() as unknown as Array<[string, { pz: PzInstance; img: HTMLImageElement }]>)
      : // production: dig through the closure indirectly by enumerating
        // the panels' DOM nodes via class selector.
        Array.from(
          document.querySelectorAll<HTMLImageElement>(".anomaly-viewer__panel-img"),
        ).map((img) => [
          img.alt || "panel",
          { pz: null as unknown as PzInstance, img },
        ]);
  if (!entries.length) return null;

  // Each tile = the panel container's current size; render at 2× for
  // crispness on retina.
  const SCALE = 2;
  const tileW = entries[0][1].img.parentElement?.clientWidth ?? 512;
  const tileH = entries[0][1].img.parentElement?.clientHeight ?? 512;
  const GAP = 6;
  const canvas = document.createElement("canvas");
  canvas.width = (tileW * entries.length + GAP * (entries.length - 1)) * SCALE;
  canvas.height = tileH * SCALE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.fillStyle = "#0b0b0f";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < entries.length; i++) {
    const { img } = entries[i][1];
    // Use the rendered <img>'s on-screen rectangle directly. Panzoom
    // applies the transform via CSS so the image's getBoundingClientRect
    // already reflects the live pan+zoom; we just need to map it into
    // the tile's local coordinates relative to the container.
    const container = img.parentElement!;
    const tileX0 = i * (tileW + GAP) * SCALE;
    const cRect = container.getBoundingClientRect();
    const iRect = img.getBoundingClientRect();
    const dx = (iRect.left - cRect.left) * SCALE + tileX0;
    const dy = (iRect.top - cRect.top) * SCALE;
    const dw = iRect.width * SCALE;
    const dh = iRect.height * SCALE;

    ctx.save();
    // Clip to the tile so transforms outside the panel don't bleed.
    ctx.beginPath();
    ctx.rect(tileX0, 0, tileW * SCALE, tileH * SCALE);
    ctx.clip();
    try {
      ctx.drawImage(img, dx, dy, dw, dh);
    } catch {
      // taint-safe fallback — same-origin nginx + credentialled fetch
      // should never get here, but be defensive.
    }
    ctx.restore();
  }

  return new Promise<Blob | null>((resolve) =>
    canvas.toBlob((b) => resolve(b), "image/png"),
  );
}

function useLinkedPanzoom(): LinkedPanzoom {
  const panels = useRef<Map<string, PanelEntry>>(new Map());
  const applying = useRef(false);

  return useMemo<LinkedPanzoom>(() => {
    const register = (key: string, entry: PanelEntry) => {
      panels.current.set(key, entry);
      return () => {
        panels.current.delete(key);
      };
    };

    const apply = (t: { x: number; y: number; scale: number }, exceptKey?: string) => {
      applying.current = true;
      try {
        for (const [k, e] of panels.current) {
          if (k === exceptKey) continue;
          const cur = e.pz.getTransform();
          // panzoom has no setTransform; emulate via zoomAbs+moveTo.
          // zoomAbs around (0,0) doesn't shift origin meaningfully here
          // because moveTo overwrites translation right after.
          e.pz.zoomAbs(0, 0, t.scale);
          e.pz.moveTo(t.x, t.y);
          // touch cur so TS doesn't complain about unused; also useful
          // when debugging.
          void cur;
        }
      } finally {
        // Yield a tick so the panzoom 'transform' events triggered by
        // our writes flush before we drop the guard. Without this, the
        // very next event in the source panel re-broadcasts and we get
        // a feedback echo.
        requestAnimationFrame(() => { applying.current = false; });
      }
    };

    const broadcast = (
      from: string,
      t: { x: number; y: number; scale: number },
    ) => {
      if (applying.current) return;
      apply(t, from);
    };

    const zoomToImageRect = (rect: { x: number; y: number; w: number; h: number }) => {
      // Pick any panel for sizing — all three are the same natural size
      // (worker writes them at the scene's native resolution). Compute
      // the transform that fits the rect to the panel's displayed area.
      const first = panels.current.values().next().value as PanelEntry | undefined;
      if (!first) return;
      const img = first.img;
      const containerW = img.parentElement?.clientWidth ?? img.clientWidth;
      const containerH = img.parentElement?.clientHeight ?? img.clientHeight;
      const naturalW = img.naturalWidth || img.clientWidth;
      const naturalH = img.naturalHeight || img.clientHeight;
      if (rect.w < 4 || rect.h < 4 || naturalW === 0 || naturalH === 0) return;

      // panzoom transform is applied to the image element. At scale=1
      // the image is rendered with object-fit:contain into the container,
      // so the "base" displayed size (at scale=1, translate=0,0) is the
      // contain-fit. Compute that base scale (img-pixels → container-pixels):
      const baseScale = Math.min(containerW / naturalW, containerH / naturalH);
      const baseDisplayW = naturalW * baseScale;
      const baseDisplayH = naturalH * baseScale;
      const baseOffsetX = (containerW - baseDisplayW) / 2;
      const baseOffsetY = (containerH - baseDisplayH) / 2;

      // Target zoom: fit the rect to the container.
      const rectDisplayW = rect.w * baseScale;
      const rectDisplayH = rect.h * baseScale;
      const targetScale = Math.min(
        containerW / rectDisplayW,
        containerH / rectDisplayH,
        12, // cap matches panzoom maxZoom
      );

      // Where the rect center sits in container-pixels at scale=1.
      const cx = baseOffsetX + (rect.x + rect.w / 2) * baseScale;
      const cy = baseOffsetY + (rect.y + rect.h / 2) * baseScale;

      // panzoom's transform: displayedX = scale * x_img + tx. After
      // scaling the image around (0,0) and translating, the point that
      // was at container-px (cx, cy) at scale=1 ends up at
      // (cx * scale + tx, cy * scale + ty). Solve for tx,ty so that
      // point lands in the container center.
      const tx = containerW / 2 - cx * targetScale;
      const ty = containerH / 2 - cy * targetScale;
      apply({ x: tx, y: ty, scale: targetScale });
    };

    const resetAll = () => {
      apply({ x: 0, y: 0, scale: 1 });
    };

    return { register, broadcast, zoomToImageRect, resetAll };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

function ZoomablePanel({
  src,
  title,
  colormapHint,
  linker,
  panelKey,
  areaZoom,
  gtDots,
  sceneShape,
  onDotClick,
}: {
  src: string;
  title: string;
  colormapHint?: string;
  linker?: LinkedPanzoom;
  panelKey?: string;
  areaZoom?: boolean;
  /** Pairs of (row, col) in scene-pixel coords for GT-positive pixels.
   *  When set, an SVG overlay rides the same CSS transform as the
   *  panzoom'd <img> so dots stay aligned across pan/zoom. */
  gtDots?: Array<[number, number]>;
  /** [H, W] of the source scene — used to scale dot coords into the
   *  img's natural pixel space. */
  sceneShape?: [number, number];
  /** Called with (row, col) when a dot is clicked. */
  onDotClick?: (row: number, col: number) => void;
}) {
  // Standalone callers (e.g. cloud_mask single-panel viewer) get a
  // throw-away controller so the link/area-zoom plumbing stays a no-op
  // without forcing them to instantiate the hook.
  const standaloneLinker = useLinkedPanzoom();
  const effLinker = linker ?? standaloneLinker;
  const effPanelKey = panelKey ?? "standalone";
  const effAreaZoom = Boolean(areaZoom);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const pzRef = useRef<PzInstance | null>(null);

  // Area-zoom drag state — kept in a ref so the mousemove handler reads
  // the freshest values without re-binding listeners on every render.
  const dragRef = useRef<{
    active: boolean;
    startX: number;
    startY: number;
    curX: number;
    curY: number;
  }>({ active: false, startX: 0, startY: 0, curX: 0, curY: 0 });
  const [rectBox, setRectBox] = useState<
    { x: number; y: number; w: number; h: number } | null
  >(null);

  useEffect(() => {
    let cancelled = false;
    let unregister: (() => void) | null = null;
    const setup = async () => {
      if (cancelled) return;
      const target = imgRef.current;
      if (!target) return;
      // Lazy import keeps panzoom out of the initial bundle.
      const mod = await import("panzoom");
      if (cancelled) return;
      const instance = mod.default(target, {
        maxZoom: 12,
        minZoom: 1,
        bounds: true,
        boundsPadding: 0.2,
        smoothScroll: false,
        zoomDoubleClickSpeed: 1, // disable default; we wire double-click below
      }) as unknown as PzInstance;
      pzRef.current = instance;
      const onTransform = () => {
        effLinker.broadcast(effPanelKey, instance.getTransform());
      };
      instance.on("transform", onTransform);
      unregister = effLinker.register(effPanelKey, { pz: instance, img: target });
    };
    void setup();
    return () => {
      cancelled = true;
      unregister?.();
      pzRef.current?.dispose();
      pzRef.current = null;
    };
    // src change reloads the image; panzoom instance stays valid because
    // it's bound to the <img> element which React keeps stable here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, effPanelKey]);

  // When area-zoom toggles on, pause panzoom's own drag/wheel so our
  // rectangle interaction doesn't fight it. Resume when toggled off.
  useEffect(() => {
    const inst = pzRef.current;
    if (!inst) return;
    if (effAreaZoom) inst.pause?.();
    else inst.resume?.();
  }, [effAreaZoom]);

  const onDoubleClick = () => {
    if (effAreaZoom) return;
    effLinker.resetAll();
  };

  // --- Area-zoom mouse handlers ---------------------------------------
  // While the toolbar toggle is on, draw a marquee on this panel and
  // (on mouseup) ask the controller to zoom all three panels to the
  // corresponding image-space rect.
  const containerToImageRect = (
    box: { x: number; y: number; w: number; h: number },
  ): { x: number; y: number; w: number; h: number } | null => {
    const img = imgRef.current;
    if (!img) return null;
    const containerW = img.parentElement?.clientWidth ?? img.clientWidth;
    const containerH = img.parentElement?.clientHeight ?? img.clientHeight;
    const naturalW = img.naturalWidth || img.clientWidth;
    const naturalH = img.naturalHeight || img.clientHeight;
    if (naturalW === 0 || naturalH === 0) return null;

    const baseScale = Math.min(containerW / naturalW, containerH / naturalH);
    const baseDisplayW = naturalW * baseScale;
    const baseDisplayH = naturalH * baseScale;
    const baseOffsetX = (containerW - baseDisplayW) / 2;
    const baseOffsetY = (containerH - baseDisplayH) / 2;

    // Undo this panel's current transform on the container-px box so we
    // get back to scale=1 base-fit coordinates, then divide out baseScale
    // to land in image pixels. This makes the marquee work correctly
    // even when the panel was already zoomed in.
    const t = pzRef.current?.getTransform() ?? { x: 0, y: 0, scale: 1 };
    const baseX = (box.x - t.x) / t.scale;
    const baseY = (box.y - t.y) / t.scale;
    const baseW = box.w / t.scale;
    const baseH = box.h / t.scale;

    const imgX = (baseX - baseOffsetX) / baseScale;
    const imgY = (baseY - baseOffsetY) / baseScale;
    const imgW = baseW / baseScale;
    const imgH = baseH / baseScale;
    return { x: imgX, y: imgY, w: imgW, h: imgH };
  };

  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!effAreaZoom) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    e.preventDefault();
    dragRef.current = {
      active: true,
      startX: e.clientX - rect.left,
      startY: e.clientY - rect.top,
      curX: e.clientX - rect.left,
      curY: e.clientY - rect.top,
    };
    setRectBox({ x: dragRef.current.startX, y: dragRef.current.startY, w: 0, h: 0 });
  };

  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!effAreaZoom || !dragRef.current.active) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    dragRef.current.curX = e.clientX - rect.left;
    dragRef.current.curY = e.clientY - rect.top;
    const x = Math.min(dragRef.current.startX, dragRef.current.curX);
    const y = Math.min(dragRef.current.startY, dragRef.current.curY);
    const w = Math.abs(dragRef.current.curX - dragRef.current.startX);
    const h = Math.abs(dragRef.current.curY - dragRef.current.startY);
    setRectBox({ x, y, w, h });
  };

  const onMouseUp = () => {
    if (!effAreaZoom || !dragRef.current.active) return;
    dragRef.current.active = false;
    const box = rectBox;
    setRectBox(null);
    if (!box || box.w < 4 || box.h < 4) return;
    const imgRect = containerToImageRect(box);
    if (!imgRect) return;
    effLinker.zoomToImageRect(imgRect);
  };

  return (
    <figure className="anomaly-viewer__panel">
      <figcaption className="anomaly-viewer__panel-title">
        <span>{title}</span>
        {colormapHint && (
          <span className="anomaly-viewer__panel-sub">{colormapHint}</span>
        )}
      </figcaption>
      <div
        ref={containerRef}
        className="anomaly-viewer__panel-canvas"
        data-area-zoom={effAreaZoom ? "true" : "false"}
        onDoubleClick={onDoubleClick}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <img
          ref={imgRef}
          src={src}
          alt={title}
          draggable={false}
          className="anomaly-viewer__panel-img"
        />
        {gtDots && gtDots.length > 0 && sceneShape && (
          <GtDotsOverlay
            imgRef={imgRef}
            pzRef={pzRef}
            dots={gtDots}
            sceneShape={sceneShape}
            onDotClick={onDotClick}
          />
        )}
        {effAreaZoom && rectBox && rectBox.w > 0 && rectBox.h > 0 && (
          <div
            className="anomaly-viewer__marquee"
            style={{
              left: rectBox.x,
              top: rectBox.y,
              width: rectBox.w,
              height: rectBox.h,
            }}
          />
        )}
      </div>
    </figure>
  );
}

// Spectrum probe — opens a small modal that calls the existing
// /scenes/{id}/spectrum endpoint for a single pixel. Driven by GT-dot
// clicks in the anomaly viewer; harmless if invoked from elsewhere.
function SpectrumProbeModal({
  sceneId,
  sensorType,
  row,
  col,
  onClose,
}: {
  sceneId: string;
  sensorType: string | null;
  row: number;
  col: number;
  onClose: () => void;
}) {
  type SpectrumPoint = {
    wavelength_nm: number;
    reflectance: number;
    spectral_family: string | null;
    is_valid: boolean;
  };
  type SpectrumResp = {
    row: number;
    col: number;
    height: number;
    width: number;
    points: SpectrumPoint[];
  };

  const [data, setData] = useState<SpectrumResp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isThermal = sensorType === "landsat9";

  useEffect(() => {
    if (isThermal) return; // thermal spectrum endpoint returns 422
    let cancelled = false;
    setData(null);
    setError(null);
    fetch(
      `/api/scenes/${encodeURIComponent(sceneId)}/spectrum?row=${row}&col=${col}`,
      { credentials: "include" },
    )
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(
            `${r.status} ${(body as { detail?: string }).detail ?? ""}`,
          );
        }
        return r.json() as Promise<SpectrumResp>;
      })
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [sceneId, row, col, isThermal]);

  // Esc closes
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="anomaly-viewer__modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Pixel spectrum"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{ zIndex: 1200 }}
    >
      <section
        className="anomaly-viewer__modal"
        style={{ maxWidth: 720, maxHeight: "80vh" }}
      >
        <header className="anomaly-viewer__modal-header">
          <div>
            <h3>Spectrum probe · pixel ({row}, {col})</h3>
            <p className="anomaly-viewer__modal-sub small">
              GT-positive pixel · reflectance vs wavelength from the
              persisted vendable
            </p>
          </div>
          <button
            type="button"
            className="anomaly-viewer__modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>
        <div className="anomaly-viewer__modal-body">
          {isThermal ? (
            <p className="action-output-viewer__hint small">
              Thermal scenes are single-band — no spectrum to plot. The
              pixel's temperature is visible directly in the score / RGB
              panels.
            </p>
          ) : error ? (
            <p className="form__error" role="alert">
              spectrum_unavailable: {error}
            </p>
          ) : !data ? (
            <p className="scene-detail__hint">Loading spectrum…</p>
          ) : (
            <>
              <SpectrumChart
                wavelengths={data.points.map((p) => p.wavelength_nm)}
                series={[
                  {
                    label: `(${row}, ${col})`,
                    color: "#0ea5e9",
                    values: data.points.map((p) => p.reflectance),
                  },
                ]}
                yLabel="reflectance"
              />
              <p className="small" style={{ marginTop: "0.4rem", color: "#6b7280" }}>
                {data.points.length} bands · scene {data.height} × {data.width}
              </p>
            </>
          )}
        </div>
      </section>
    </div>
  );
}

// SVG overlay of GT-positive pixels. Mirrors the panzoom'd <img>'s
// current screen rectangle on every transform tick so dots stay
// pixel-perfect aligned across pan/zoom. Each circle is clickable; the
// parent stops propagation so panzoom doesn't treat a dot click as a
// drag start.
function GtDotsOverlay({
  imgRef,
  pzRef,
  dots,
  sceneShape,
  onDotClick,
}: {
  imgRef: React.RefObject<HTMLImageElement | null>;
  pzRef: React.RefObject<PzInstance | null>;
  dots: Array<[number, number]>;
  sceneShape: [number, number];
  onDotClick?: (row: number, col: number) => void;
}) {
  // We position the SVG absolutely against the .anomaly-viewer__panel-canvas
  // (the panel's local stacking context). Every panzoom tick we read the
  // img's bounding rect and project it back into canvas-local coords.
  const [box, setBox] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    let raf = 0;
    const update = () => {
      const i = imgRef.current;
      if (!i) return;
      const parent = i.parentElement;
      if (!parent) return;
      const ir = i.getBoundingClientRect();
      const pr = parent.getBoundingClientRect();
      setBox({
        left: ir.left - pr.left,
        top: ir.top - pr.top,
        width: ir.width,
        height: ir.height,
      });
    };
    const onTransform = () => {
      // Coalesce per animation frame; panzoom can fire many events per
      // wheel tick.
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(update);
    };
    update();
    const inst = pzRef.current;
    inst?.on("transform", onTransform);
    window.addEventListener("resize", update);
    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("resize", update);
      // panzoom doesn't expose `off`; the instance is disposed by the
      // parent on unmount so the listener is cleaned up there.
    };
  }, [imgRef, pzRef, dots]);

  if (!box) return null;
  const [sceneH, sceneW] = sceneShape;
  // Scale dot radius with zoom so they stay readable: bigger when
  // zoomed out, smaller when zoomed in.
  const px = Math.max(box.width / sceneW, box.height / sceneH);
  const r = Math.max(3, Math.min(8, 4 / Math.max(px, 0.5)));

  return (
    <svg
      className="gt-dots-overlay"
      style={{
        position: "absolute",
        left: box.left,
        top: box.top,
        width: box.width,
        height: box.height,
        pointerEvents: "none",
      }}
      viewBox={`0 0 ${sceneW} ${sceneH}`}
      preserveAspectRatio="none"
    >
      {dots.map(([row, col], i) => (
        <circle
          key={i}
          cx={col + 0.5}
          cy={row + 0.5}
          r={r}
          className="gt-dots-overlay__dot"
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            onDotClick?.(row, col);
          }}
        />
      ))}
    </svg>
  );
}

function AnomalyScoringROCBlock({
  diag,
  error,
  hasGt,
}: {
  diag: AnomalyScoringDiagnostics | null;
  error: string | null;
  hasGt: boolean;
}) {
  if (!hasGt) {
    return (
      <p className="action-output-viewer__hint small">
        No GT annotation attached — re-run with one selected in the
        dialog to produce ROC curves.
      </p>
    );
  }
  if (error) {
    return (
      <p className="form__optional small">
        Diagnostics file unavailable ({error}). ROC unavailable.
      </p>
    );
  }
  if (!diag) {
    return <p className="scene-detail__hint">Loading ROC…</p>;
  }
  const roc = diag.roc;
  if (!roc || Object.keys(roc).length === 0) {
    return (
      <p className="action-output-viewer__hint small">
        ROC data missing in diagnostics — likely an older run, or the
        attached annotation has no positives in the keep_mask.
      </p>
    );
  }
  const curves = Object.entries(roc).map(([codename, r], i) => ({
    label: codename,
    color: AD_MODEL_COLORS[i % AD_MODEL_COLORS.length],
    fpr: r.fpr,
    tpr: r.tpr,
    auc: r.auc,
  }));
  return (
    <div className="action-output-viewer__chart">
      <div className="action-output-viewer__chart-title">
        ROC curves vs ground-truth annotation
        <span className="action-output-viewer__chart-sub">
          one curve per model · computed over the keep_mask only · dashed
          line = random-classifier baseline
        </span>
      </div>
      <ROCChart curves={curves} />
    </div>
  );
}


// =====================================================================
// AnomalyDetectionPrepViewer — three panels + threshold slider + Apply
// =====================================================================
//
// The prep action's worker phase finishes with a composite score map
// on disk. This viewer lets the user explore thresholds live: each
// Apply press posts {threshold, mode, dilation_kernel} to the backend,
// which returns the binarised mask PNG + (if GT attached) precision /
// recall / F1. Slider moves are inert until the user presses Apply —
// keeps the metrics honest (no stale numbers).
//
// Defaults: percentile mode at 95% (keep the top 5%), dilation off.
//
// Roadmap step 14.5.

interface AnomalyDetectionPrepSummary {
  upstream_anomaly_scoring_output_id?: string;
  upstream_codenames?: string[];
  active_codenames?: string[];
  weights_normalised?: Record<string, number>;
  composite_shape?: [number, number];
  composite_distribution?: {
    min: number;
    p2: number;
    p50: number;
    p90: number;
    p95: number;
    p99: number;
    p99_5: number;
    max: number;
    mean: number;
    std: number;
    valid_pixels: number;
    total_pixels: number;
  };
  has_gt?: boolean;
}

// Public entry — small inline tile + Open button. The big interactive
// surface lives in the modal below so it gets full-screen real estate.
function AnomalyDetectionPrepViewer({
  actionId,
  summary,
  meta,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  meta: ActionTypeMeta | null;
}) {
  const s = summary as AnomalyDetectionPrepSummary;
  const activeAlgos = s.active_codenames ?? [];
  const weights = s.weights_normalised ?? {};
  const compositeShape = s.composite_shape ?? [0, 0];
  const distribution = s.composite_distribution;
  const compositeUrl = `/api/actions/${encodeURIComponent(
    actionId,
  )}/files/composite_score.png`;

  const [modalOpen, setModalOpen] = useState(false);
  // Cmd/Ctrl+I shortcut to mirror AnomalyScoringOutputViewer.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (modalOpen) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") {
        e.preventDefault();
        setModalOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  return (
    <div className="action-output-viewer">
      <h4>Anomaly detection — prep</h4>
      <div className="anomaly-viewer__summary-card">
        <button
          type="button"
          className="anomaly-viewer__thumb-btn"
          onClick={() => setModalOpen(true)}
          aria-label="Open threshold explorer"
        >
          <img
            src={compositeUrl}
            alt="Composite anomaly score thumbnail"
            className="anomaly-viewer__thumb-img"
          />
          <span className="anomaly-viewer__thumb-overlay">
            <span className="anomaly-viewer__thumb-icon" aria-hidden="true">
              ⤢
            </span>
            Explore thresholds
          </span>
        </button>
        <div className="anomaly-viewer__summary-meta">
          <p className="anomaly-viewer__summary-line">
            <strong>
              {activeAlgos.length} algorithm
              {activeAlgos.length === 1 ? "" : "s"}
            </strong>{" "}
            · scene {`${compositeShape[0]} × ${compositeShape[1]}`}
          </p>
          {distribution && (
            <p className="anomaly-viewer__summary-line small">
              composite p50={distribution.p50.toFixed(3)} · p99={" "}
              {distribution.p99.toFixed(3)} · valid{" "}
              {distribution.valid_pixels.toLocaleString()} /{" "}
              {distribution.total_pixels.toLocaleString()}
            </p>
          )}
          {activeAlgos.length > 0 && (
            <ul className="anomaly-viewer__summary-models">
              {activeAlgos.map((a) => (
                <li key={a}>
                  <strong>{a}</strong>{" "}
                  <span className="mono small">
                    weight {(weights[a] ?? 0).toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <button
            type="button"
            className="btn anomaly-viewer__open-btn"
            onClick={() => setModalOpen(true)}
          >
            Open threshold explorer
            <span className="anomaly-viewer__shortcut">⌘I</span>
          </button>
          <PrepExportButton actionId={actionId} summary={summary} />
        </div>
      </div>

      {modalOpen && (
        <AnomalyDetectionPrepModal
          actionId={actionId}
          summary={summary}
          meta={meta}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}


// Export button shown on the inline prep card. Disabled (with a clear
// tooltip) until the action has been committed — the export endpoint
// refuses to ship a thermal bundle without anomaly_mask.tif on disk.
function PrepExportButton({
  actionId,
  summary,
}: {
  actionId: string;
  summary: Record<string, unknown>;
}) {
  const committed = Boolean(
    (summary as Record<string, unknown>).committed
    ?? (summary as Record<string, unknown>).committed_threshold !== undefined,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onClick = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await fetch(
        `/api/actions/${encodeURIComponent(actionId)}/export`,
        { method: "POST" },
      );
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { const j = await r.json(); detail = j.detail ?? detail; } catch { /* */ }
        if (detail === "crs_missing") {
          throw new Error("Scene has no CRS — submission requires georef.");
        }
        if (detail === "prep_not_committed") {
          throw new Error("Commit the threshold first, then export.");
        }
        throw new Error(`Export failed: ${detail}`);
      }
      const blob = await r.blob();
      const filename =
        r.headers.get("Content-Disposition")
          ?.match(/filename="([^"]+)"/)?.[1]
        ?? `thermal_${actionId.replace(/^action_/, "").slice(0, 8)}.zip`;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button
        type="button"
        className="btn anomaly-viewer__open-btn"
        onClick={onClick}
        disabled={busy || !committed}
        title={
          committed
            ? "Download submission bundle (thermal GeoTIFF + SHP + CSV + manifest)"
            : "Commit a threshold first, then export the bundle"
        }
      >
        {busy ? "Exporting…" : "Export bundle"}
      </button>
      {error && (
        <p className="form__error small" title={error}>
          {error.length > 80 ? error.slice(0, 80) + "…" : error}
        </p>
      )}
    </>
  );
}


// Big interactive surface — three panels + slider + Apply + metrics
// row, rendered into a backdrop modal so it gets full-screen space.
function AnomalyDetectionPrepModal({
  actionId,
  summary,
  meta: _meta,
  onClose,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  meta: ActionTypeMeta | null;
  onClose: () => void;
}) {
  // Esc closes; lock body scroll while open.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const s = summary as AnomalyDetectionPrepSummary;
  const hasGt = Boolean(s.has_gt);
  const activeAlgos = s.active_codenames ?? [];
  const weights = s.weights_normalised ?? {};
  const dist = s.composite_distribution;

  // Three-panel linked panzoom — RGB, composite, mask share the same
  // {pan, zoom} transform so the user can compare features pixel-for-
  // pixel across panels. Same controller pattern AnomalyScoringTriPanel
  // uses.
  const linker = useLinkedPanzoom();

  // URLs for the static panels.
  const rgbUrl = `/api/actions/${encodeURIComponent(actionId)}/files/rgb.png`;
  const compositeUrl = `/api/actions/${encodeURIComponent(
    actionId,
  )}/files/composite_score.png`;

  // Slider state — inert until Apply. Defaults to "show top 5%".
  const [threshold, setThreshold] = useState<number>(95);
  const [thresholdMode, setThresholdMode] = useState<"percentile" | "absolute">(
    "percentile",
  );
  const [dilationKernel, setDilationKernel] = useState<number>(0);

  // Preview state — what's currently displayed in the binary panel.
  const [applying, setApplying] = useState(false);
  const [previewResult, setPreviewResult] =
    useState<AnomalyDetectionPreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Commit state — read initially from the action's summary, mutated
  // after a successful commit so the indicator updates without a
  // round-trip back through getAction().
  const initialCommitted = (summary as Record<string, unknown>)
    .upstream_anomaly_scoring_output_id // any required prep field guards us
    ? Boolean(
        (summary as Record<string, unknown>).committed ??
          // older preps stamped the flag on action_output.summary
          (summary as Record<string, unknown>).committed_threshold !== undefined,
      )
    : false;
  type CommittedSnapshot = {
    threshold: number;
    threshold_mode: "percentile" | "absolute";
    threshold_absolute: number;
    dilation_kernel: number;
  };
  const initialCommittedSnapshot: CommittedSnapshot | null =
    typeof (summary as Record<string, unknown>).committed_threshold === "number"
      ? {
          threshold: Number(
            (summary as Record<string, unknown>).committed_threshold,
          ),
          threshold_mode:
            ((summary as Record<string, unknown>).committed_threshold_mode as
              | "percentile"
              | "absolute") ?? "percentile",
          threshold_absolute: Number(
            (summary as Record<string, unknown>)
              .committed_threshold_absolute ??
              (summary as Record<string, unknown>).committed_threshold,
          ),
          dilation_kernel: Number(
            (summary as Record<string, unknown>).committed_dilation_kernel ?? 0,
          ),
        }
      : null;
  const [committed, setCommitted] = useState<CommittedSnapshot | null>(
    initialCommitted ? initialCommittedSnapshot : null,
  );
  const [committing, setCommitting] = useState(false);
  const [commitError, setCommitError] = useState<string | null>(null);

  // Save-visualization state — mirrors AnomalyScoringTriPanel so the
  // user can snapshot a particular threshold view into the project's
  // Visualizations tab without committing.
  const [savingView, setSavingView] = useState(false);
  const [saveViewMsg, setSaveViewMsg] = useState<string | null>(null);

  const onSaveView = async () => {
    const thresholdLabel =
      thresholdMode === "percentile"
        ? `p${threshold.toFixed(1)}`
        : `≥${threshold.toFixed(3)}`;
    const defaultName = `Anomaly mask · ${thresholdLabel}${
      dilationKernel ? ` · dilation ${dilationKernel}` : ""
    }`;
    const name = window.prompt("Name this visualization:", defaultName);
    if (!name) return;
    setSavingView(true);
    setSaveViewMsg(null);
    try {
      const act = await getAction(actionId);
      const blob = await composeTriPanelBlob(linker);
      if (!blob) throw new Error("compose_failed");
      await createVisualization({
        projectId: act.project_id,
        source: { kind: "action_output", action_id: actionId },
        name,
        description: undefined,
        viewState: {
          threshold,
          threshold_mode: thresholdMode,
          dilation_kernel: dilationKernel,
        },
        imageBlob: blob,
      });
      setSaveViewMsg("Saved. See the Visualizations tab.");
      window.dispatchEvent(new CustomEvent("allotrope:viz-saved"));
    } catch (err) {
      setSaveViewMsg(
        err instanceof ApiError
          ? `Save failed: ${err.detail ?? err.status}`
          : `Save failed: ${(err as Error).message}`,
      );
    } finally {
      setSavingView(false);
    }
  };

  const onApply = async () => {
    setApplying(true);
    setPreviewError(null);
    try {
      const res = await submitAnomalyDetectionPreview(actionId, {
        threshold,
        threshold_mode: thresholdMode,
        dilation_kernel: dilationKernel,
      });
      setPreviewResult(res);
    } catch (err) {
      setPreviewError(
        err instanceof ApiError
          ? (err.detail ?? `HTTP ${err.status}`)
          : "could not reach the server",
      );
    } finally {
      setApplying(false);
    }
  };

  const onCommit = async () => {
    const isRecommit = committed !== null;
    const msg = isRecommit
      ? `Overwrite the existing committed threshold (currently above p${committed!.threshold.toFixed(
          1,
        )}, dilation ${committed!.dilation_kernel}) with the current slider values?`
      : `Lock in these values as the committed anomaly mask?\n\nThreshold mode: ${thresholdMode}\nThreshold: ${threshold.toFixed(
          4,
        )}\nDilation kernel: ${dilationKernel}\n\nDownstream actions will see this committed result; you can re-commit later if you change your mind.`;
    if (!window.confirm(msg)) return;
    setCommitting(true);
    setCommitError(null);
    try {
      const res = await submitAnomalyDetectionCommit(actionId, {
        threshold,
        threshold_mode: thresholdMode,
        dilation_kernel: dilationKernel,
      });
      setCommitted({
        threshold,
        threshold_mode: thresholdMode,
        threshold_absolute: res.threshold_absolute,
        dilation_kernel: dilationKernel,
      });
      // A successful commit also generates a fresh authoritative
      // preview result — mirror it into the panel so the user sees
      // the same mask the committed file holds.
      setPreviewResult({
        threshold_absolute: res.threshold_absolute,
        threshold_percentile: res.threshold_percentile,
        dilation_kernel: res.dilation_kernel,
        n_anomalous: res.n_anomalous,
        n_kept: res.n_kept,
        metrics: res.metrics,
        // No mask_url returned by commit (we wrote the .tif, not the
        // ephemeral PNG); leave the existing previewResult.mask_url if
        // any so the panel stays populated.
        mask_url: previewResult?.mask_url ?? "",
      });
    } catch (err) {
      setCommitError(
        err instanceof ApiError
          ? (err.detail ?? `HTTP ${err.status}`)
          : "could not reach the server",
      );
    } finally {
      setCommitting(false);
    }
  };

  // Slider range depends on mode. In absolute mode we span the
  // composite's min..max range; in percentile mode we span 0..100.
  const sliderMin = thresholdMode === "percentile" ? 0 : (dist?.min ?? 0);
  const sliderMax = thresholdMode === "percentile" ? 100 : (dist?.max ?? 1);
  const sliderStep =
    thresholdMode === "percentile" ? 0.5 : Math.max(0.001, ((dist?.max ?? 1) - (dist?.min ?? 0)) / 200);

  // When the user toggles modes, reset the threshold to a sensible
  // default for that mode so the slider value doesn't go out of range.
  const onToggleMode = (newMode: "percentile" | "absolute") => {
    setThresholdMode(newMode);
    if (newMode === "percentile") {
      setThreshold(95);
    } else if (dist) {
      // Default absolute to p95 of the composite.
      setThreshold(Number((dist.p95 ?? dist.p50).toFixed(4)));
    }
    // Existing preview is for the old mode; clear it.
    setPreviewResult(null);
  };

  return (
    <div
      className="anomaly-viewer__modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Anomaly detection prep — threshold explorer"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <section className="anomaly-viewer__modal adp-viewer">
        <header className="anomaly-viewer__modal-header">
          <div>
            <h3>Anomaly detection — prep</h3>
            <p className="form__hint small">
              {activeAlgos.length > 0 ? (
                <>
                  Composite from {activeAlgos.length} algorithm
                  {activeAlgos.length === 1 ? "" : "s"}:{" "}
                  {activeAlgos.map((a) => (
                    <span key={a} className="adp-viewer__algo-chip">
                      {a}
                      {weights[a] !== undefined
                        ? ` · ${weights[a].toFixed(2)}`
                        : ""}
                    </span>
                  ))}
                </>
              ) : (
                "No active algorithms."
              )}
            </p>
          </div>
          <button
            type="button"
            className="anomaly-viewer__modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

      <div className="anomaly-viewer__toolbar">
        <button
          type="button"
          className="anomaly-viewer__tool-btn"
          onClick={() => linker.resetAll()}
          title="Reset all panels to 1×"
        >
          Reset zoom
        </button>
        <button
          type="button"
          className="anomaly-viewer__tool-btn"
          onClick={() => void onSaveView()}
          disabled={savingView}
          title="Save the current 3-panel view to this project's Visualizations"
        >
          {savingView ? "Saving…" : "Save view"}
        </button>
        {saveViewMsg && (
          <span className="anomaly-viewer__tool-hint">{saveViewMsg}</span>
        )}
      </div>

      <div className="adp-viewer__panels anomaly-viewer__panels">
        <ZoomablePanel
          src={rgbUrl}
          title="Scene (RGB)"
          colormapHint="visual reference"
          linker={linker}
          panelKey="rgb"
        />
        <ZoomablePanel
          src={compositeUrl}
          title="Composite score"
          colormapHint="inferno · 0..1"
          linker={linker}
          panelKey="composite"
        />
        {/* Mask panel — when a preview exists we hand its URL to a
            ZoomablePanel so it shares the linked transform with RGB +
            composite. Before the first Apply, render a placeholder
            inside the same panel chrome so layout doesn't shift. */}
        {previewResult && previewResult.mask_url ? (
          <div className="adp-viewer__panel-with-overlay">
            <ZoomablePanel
              src={previewResult.mask_url}
              title="Anomaly mask (preview)"
              colormapHint="white = anomalous"
              linker={linker}
              panelKey="mask"
            />
            {applying && (
              <div className="adp-viewer__loading-overlay">
                <span className="adp-viewer__spinner" aria-hidden="true" />
                <span className="small">Computing…</span>
              </div>
            )}
          </div>
        ) : (
          <figure className="anomaly-viewer__panel">
            <figcaption className="anomaly-viewer__panel-title">
              <span>Anomaly mask (preview)</span>
              <span className="anomaly-viewer__panel-sub">
                press Apply to render
              </span>
            </figcaption>
            <div className="anomaly-viewer__panel-canvas adp-viewer__placeholder">
              {applying ? (
                <div className="adp-viewer__loading-overlay">
                  <span className="adp-viewer__spinner" aria-hidden="true" />
                  <span className="small">Computing…</span>
                </div>
              ) : (
                <p className="form__optional">
                  Pick a threshold and press <strong>Apply</strong>.
                </p>
              )}
            </div>
          </figure>
        )}
      </div>

      <div className="adp-viewer__controls">
        <div className="adp-viewer__control-row">
          <label className="adp-viewer__control">
            <span>Threshold mode</span>
            <div className="adp-viewer__seg">
              <button
                type="button"
                data-active={thresholdMode === "percentile" ? "true" : "false"}
                onClick={() => onToggleMode("percentile")}
              >
                Percentile
              </button>
              <button
                type="button"
                data-active={thresholdMode === "absolute" ? "true" : "false"}
                onClick={() => onToggleMode("absolute")}
              >
                Absolute
              </button>
            </div>
          </label>

          <label className="adp-viewer__control adp-viewer__control--slider">
            <span>
              Threshold ·{" "}
              <code>
                {thresholdMode === "percentile"
                  ? `above p${threshold.toFixed(1)} (top ${(
                      100 - threshold
                    ).toFixed(1)}%)`
                  : threshold.toFixed(4)}
              </code>
            </span>
            <input
              type="range"
              min={sliderMin}
              max={sliderMax}
              step={sliderStep}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <div className="adp-viewer__slider-extents small">
              <span>
                {thresholdMode === "percentile"
                  ? "p0 · all flagged"
                  : sliderMin.toFixed(2)}
              </span>
              <span>
                {thresholdMode === "percentile"
                  ? "p100 · none flagged"
                  : sliderMax.toFixed(2)}
              </span>
            </div>
            <div className="adp-viewer__slider-numeric small">
              <span className="muted">
                or type {thresholdMode === "percentile" ? "percentile" : "value"}:
              </span>
              <input
                type="number"
                className="adp-viewer__slider-numeric-input"
                min={sliderMin}
                max={sliderMax}
                step={thresholdMode === "percentile" ? "0.01" : sliderStep}
                value={threshold}
                onChange={(e) => {
                  const raw = e.target.value;
                  if (raw === "") return;     // let the user clear mid-edit
                  const v = Number(raw);
                  if (!Number.isFinite(v)) return;
                  // Clamp to the slider's domain so the typed value never
                  // takes the range slider out of bounds.
                  const clamped = Math.max(sliderMin, Math.min(sliderMax, v));
                  setThreshold(clamped);
                }}
              />
            </div>
          </label>

          <label className="adp-viewer__control">
            <span>Dilation kernel</span>
            <select
              value={dilationKernel}
              onChange={(e) => setDilationKernel(Number(e.target.value))}
            >
              <option value={0}>None</option>
              <option value={3}>3 × 3</option>
              <option value={5}>5 × 5</option>
              <option value={7}>7 × 7</option>
              <option value={9}>9 × 9</option>
            </select>
          </label>

          <button
            type="button"
            className="adp-viewer__apply"
            onClick={onApply}
            disabled={applying || committing}
          >
            {applying ? "Applying…" : "Apply"}
          </button>

          <button
            type="button"
            className="adp-viewer__commit"
            onClick={onCommit}
            disabled={applying || committing}
            title={
              committed
                ? "Overwrite the currently committed threshold with the current slider values"
                : "Lock in these values as the committed anomaly mask"
            }
          >
            {committing ? "Committing…" : committed ? "Re-commit" : "Commit"}
          </button>
        </div>

        {committed && (
          <p className="adp-viewer__committed-note small">
            <strong>Currently committed:</strong>{" "}
            {committed.threshold_mode === "percentile"
              ? `above p${committed.threshold.toFixed(1)} (top ${(
                  100 - committed.threshold
                ).toFixed(1)}%)`
              : `absolute ≥ ${committed.threshold.toFixed(4)}`}
            {" · "}dilation {committed.dilation_kernel || "none"}
            {" · "}
            <a
              href={`/api/actions/${encodeURIComponent(
                actionId,
              )}/files/anomaly_mask.tif`}
              target="_blank"
              rel="noopener noreferrer"
            >
              download mask.tif
            </a>
          </p>
        )}

        {commitError && (
          <p className="form__error" role="alert">{commitError}</p>
        )}

        {previewError && (
          <p className="form__error" role="alert">{previewError}</p>
        )}

        {previewResult && (
          <div className="adp-viewer__metrics">
            <div className="adp-viewer__metric">
              <span className="adp-viewer__metric-label">Threshold (absolute)</span>
              <span className="adp-viewer__metric-value">
                {previewResult.threshold_absolute.toFixed(4)}
              </span>
            </div>
            <div className="adp-viewer__metric">
              <span className="adp-viewer__metric-label">Flagging top</span>
              <span className="adp-viewer__metric-value">
                {(100 - previewResult.threshold_percentile).toFixed(1)}%
              </span>
            </div>
            <div className="adp-viewer__metric">
              <span className="adp-viewer__metric-label">Pixels flagged</span>
              <span className="adp-viewer__metric-value">
                {previewResult.n_anomalous.toLocaleString()} /{" "}
                {previewResult.n_kept.toLocaleString()}
              </span>
            </div>
            {hasGt && previewResult.metrics && (
              <>
                <div className="adp-viewer__metric adp-viewer__metric--gt">
                  <span className="adp-viewer__metric-label">Precision</span>
                  <span className="adp-viewer__metric-value">
                    {previewResult.metrics.precision.toFixed(3)}
                  </span>
                </div>
                <div className="adp-viewer__metric adp-viewer__metric--gt">
                  <span className="adp-viewer__metric-label">Recall</span>
                  <span className="adp-viewer__metric-value">
                    {previewResult.metrics.recall.toFixed(3)}
                  </span>
                </div>
                <div className="adp-viewer__metric adp-viewer__metric--gt">
                  <span className="adp-viewer__metric-label">F1</span>
                  <span className="adp-viewer__metric-value">
                    {previewResult.metrics.f1.toFixed(3)}
                  </span>
                </div>
                <div className="adp-viewer__metric adp-viewer__metric--gt small">
                  <span className="adp-viewer__metric-label">TP / FP / FN</span>
                  <span className="adp-viewer__metric-value mono">
                    {previewResult.metrics.tp} / {previewResult.metrics.fp} /{" "}
                    {previewResult.metrics.fn}
                  </span>
                </div>
              </>
            )}
            {hasGt && !previewResult.metrics && (
              <p className="form__optional small">
                Ground truth was attached but the rasterised mask isn't
                available — metrics skipped.
              </p>
            )}
            {!hasGt && (
              <p className="form__optional small">
                No ground truth attached — precision / recall / F1 are
                skipped.
              </p>
            )}
          </div>
        )}
      </div>
      </section>
    </div>
  );
}


// =====================================================================
// SpectralLibraryMatchOutputViewer — splib07 match results
// =====================================================================
//
// Inline tile + full-screen modal. The modal carries a big angle-heatmap
// match map, a side rail with top-materials histogram + at-pixel top-K,
// and a spectrum chart that overlays the pixel's spectrum with the top-1
// (and optionally top-K) library candidate spectra.
//
// Hover-preview: a debounced /probe fires while the mouse is held over a
// pixel; the chart shows a "preview" pair (pixel + top-1) at lower opacity.
// Click: pins the same data at full opacity until the next click outside
// the map.
//
// Roadmap step 14.7.

interface SpectralLibraryMatchSummary {
  upstream_anomaly_detection_output_id?: string;
  sensor_type?: string;
  n_library_entries?: number;
  mode?: string;
  top_k?: number;
  min_coverage?: number;
  min_band_count?: number;
  sg_window_length?: number;
  sg_polyorder?: number;
  chapters?: string[];
  n_pixels?: number;
  n_pixels_matched?: number;
  n_pixels_no_match?: number;
  timing_seconds?: { match?: number; total?: number };
}

interface SpectralHistogramRow {
  library_ix: number;
  name: string;
  chapter: string;
  count: number;
}

function SpectralLibraryMatchOutputViewer({
  actionId,
  summary,
}: {
  actionId: string;
  summary: Record<string, unknown>;
}) {
  const s = summary as SpectralLibraryMatchSummary;
  const matchMapUrl = `/api/actions/${encodeURIComponent(actionId)}/files/match_map.png`;
  const [modalOpen, setModalOpen] = useState(false);

  // Cmd/Ctrl+I shortcut, same pattern as the anomaly_detection_prep viewer.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (modalOpen) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") {
        e.preventDefault();
        setModalOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  return (
    <div className="action-output-viewer">
      <h4>Spectral library match · USGS splib07</h4>
      <div className="anomaly-viewer__summary-card">
        <button
          type="button"
          className="anomaly-viewer__thumb-btn"
          onClick={() => setModalOpen(true)}
          aria-label="Open spectral match viewer"
        >
          <img
            src={matchMapUrl}
            alt="Top-1 match angle heatmap (thumbnail)"
            className="anomaly-viewer__thumb-img"
          />
          <span className="anomaly-viewer__thumb-overlay">
            <span className="anomaly-viewer__thumb-icon" aria-hidden="true">⤢</span>
            Explore matches
          </span>
        </button>
        <div className="anomaly-viewer__summary-meta">
          <p className="anomaly-viewer__summary-line">
            <strong>{s.n_pixels_matched ?? 0}</strong> matched ·{" "}
            {s.n_pixels_no_match ?? 0} no-match · pool {s.n_library_entries ?? 0}
          </p>
          <p className="anomaly-viewer__summary-line small">
            mode {s.mode ?? "?"} · top-K {s.top_k ?? "?"} · min_cov{" "}
            {s.min_coverage ?? "?"} · min_bands {s.min_band_count ?? "?"} · SG (
            {s.sg_window_length ?? "?"}/{s.sg_polyorder ?? "?"})
            {s.timing_seconds?.match != null && (
              <span> · match {s.timing_seconds.match.toFixed(2)} s</span>
            )}
          </p>
          {s.chapters && s.chapters.length > 0 && (
            <p className="anomaly-viewer__summary-line small">
              chapters: {s.chapters.join(", ")}
            </p>
          )}
          <button
            type="button"
            className="btn anomaly-viewer__open-btn"
            onClick={() => setModalOpen(true)}
          >
            Open spectral match viewer
            <span className="anomaly-viewer__shortcut">⌘I</span>
          </button>
        </div>
      </div>
      {modalOpen && (
        <SpectralLibraryMatchModal
          actionId={actionId}
          summary={summary}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}

function SpectralLibraryMatchModal({
  actionId,
  summary,
  onClose,
}: {
  actionId: string;
  summary: Record<string, unknown>;
  onClose: () => void;
}) {
  const s = summary as SpectralLibraryMatchSummary;
  const rgbUrl = `/api/actions/${encodeURIComponent(actionId)}/files/rgb.png`;
  const overlayUrl = `/api/actions/${encodeURIComponent(
    actionId,
  )}/files/match_map_overlay.png`;
  const histogramUrl = `/api/actions/${encodeURIComponent(actionId)}/files/histogram.json`;
  const matchesParquetUrl = `/api/actions/${encodeURIComponent(
    actionId,
  )}/files/matches.parquet`;
  const pixelSpectraNpzUrl = `/api/actions/${encodeURIComponent(
    actionId,
  )}/files/anomaly_pixel_spectra.npz`;
  const libraryReflNpzUrl = `/api/actions/${encodeURIComponent(
    actionId,
  )}/files/library_refl.npz`;

  // Esc closes; lock body scroll while open.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  // Histogram (load once).
  const [histogram, setHistogram] = useState<SpectralHistogramRow[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch(histogramUrl)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => { if (!cancelled) setHistogram(j?.top1_counts ?? []); })
      .catch(() => { if (!cancelled) setHistogram([]); });
    return () => { cancelled = true; };
  }, [histogramUrl]);

  // Preload everything the chart needs in one shot.
  const [bundle, setBundle] = useState<{
    wavelengths: number[];
    matchesByCoord: Map<number, MatchRow[]>;     // key = row*W + col
    spectraByCoord: Map<number, Float32Array>;    // pixel spectra (smoothed)
    validityByCoord: Map<number, Uint8Array>;
    libRefl: Float32Array[];                      // per-library row, length B
    libValid: Uint8Array[];
    width: number;
    height: number;
    matchedSet: Set<number>;                      // for snap-to-pixel lookup
  } | null>(null);
  const [bundleError, setBundleError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [{ readNpz }, parquetMod] = await Promise.all([
          import("../lib/npy"),
          import("hyparquet"),
        ]);
        const [pixelNpzBuf, libNpzBuf, parqBuf] = await Promise.all([
          fetch(pixelSpectraNpzUrl).then((r) => r.arrayBuffer()),
          fetch(libraryReflNpzUrl).then((r) => r.arrayBuffer()),
          fetch(matchesParquetUrl).then((r) => r.arrayBuffer()),
        ]);
        if (cancelled) return;

        const pixNpz = await readNpz(pixelNpzBuf);
        const libNpz = await readNpz(libNpzBuf);

        const wavelengths = Array.from(pixNpz.wavelengths.data as Float64Array);
        const rows = pixNpz.rows.data as Int32Array;
        const cols = pixNpz.cols.data as Int32Array;
        const spectraFlat = pixNpz.spectra.data as Float32Array;
        const validityFlat = pixNpz.valid.data as Uint8Array;
        const [P, B] = pixNpz.spectra.shape as [number, number];

        const spectraByCoord = new Map<number, Float32Array>();
        const validityByCoord = new Map<number, Uint8Array>();
        const matchedSet = new Set<number>();
        // Image width comes from the cube max(cols)+1; we don't have explicit
        // dims here but image natural dims will give us the answer when the
        // <img> loads. Use a sentinel width derived from max col so coord
        // keys stay unique even before the image loads.
        let maxC = 0, maxR = 0;
        for (let i = 0; i < P; i++) {
          if (cols[i] > maxC) maxC = cols[i];
          if (rows[i] > maxR) maxR = rows[i];
        }
        const widthKey = maxC + 2; // big enough to hash uniquely
        for (let i = 0; i < P; i++) {
          const key = rows[i] * widthKey + cols[i];
          spectraByCoord.set(key, spectraFlat.subarray(i * B, (i + 1) * B));
          validityByCoord.set(key, validityFlat.subarray(i * B, (i + 1) * B));
          matchedSet.add(key);
        }

        // Library arrays.
        const libRefl = libNpz.refl.data as Float32Array;
        const libValid = libNpz.valid.data as Uint8Array;
        const [N, LB] = libNpz.refl.shape as [number, number];
        const libReflRows: Float32Array[] = [];
        const libValidRows: Uint8Array[] = [];
        for (let i = 0; i < N; i++) {
          libReflRows.push(libRefl.subarray(i * LB, (i + 1) * LB));
          libValidRows.push(libValid.subarray(i * LB, (i + 1) * LB));
        }

        // Parquet → matches grouped by (row,col).
        // hyparquet's parquetReadObjects expects an AsyncBuffer; wrap the
        // ArrayBuffer in the trivial in-memory adapter.
        const asyncBuf = {
          byteLength: parqBuf.byteLength,
          async slice(start: number, end?: number) {
            return parqBuf.slice(start, end ?? parqBuf.byteLength);
          },
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const matches = (await (parquetMod as any).parquetReadObjects({
          file: asyncBuf,
        })) as MatchRow[];
        const matchesByCoord = new Map<number, MatchRow[]>();
        for (const m of matches) {
          // Coerce in case the parquet reader hands back BigInt for int64
          // columns. We rewrite the row object in-place so downstream code
          // sees plain numbers.
          m.row = Number(m.row);
          m.col = Number(m.col);
          m.rank = Number(m.rank);
          m.library_ix = Number(m.library_ix);
          m.angle_deg = Number(m.angle_deg);
          m.n_bands_used = Number(m.n_bands_used);
          const key = m.row * widthKey + m.col;
          let bucket = matchesByCoord.get(key);
          if (!bucket) {
            bucket = [];
            matchesByCoord.set(key, bucket);
          }
          bucket.push(m);
        }
        for (const bucket of matchesByCoord.values()) {
          bucket.sort((a, b) => a.rank - b.rank);
        }

        if (cancelled) return;
        setBundle({
          wavelengths,
          matchesByCoord,
          spectraByCoord,
          validityByCoord,
          libRefl: libReflRows,
          libValid: libValidRows,
          width: widthKey, // hash basis; not the image width
          height: maxR + 1,
          matchedSet,
        });
      } catch (err) {
        console.error("[spectral-match] bundle load failed", err);
        if (!cancelled) setBundleError(String(err));
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pixelSpectraNpzUrl, libraryReflNpzUrl, matchesParquetUrl]);

  // Panzoom + snap-to-pixel.
  const containerRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const pzRef = useRef<PzInstance | null>(null);
  const [hoverPx, setHoverPx] = useState<{ row: number; col: number } | null>(null);
  const [pinnedPx, setPinnedPx] = useState<{ row: number; col: number } | null>(null);

  // Tracks whether the rgb image has reported `onload`. Panzoom needs to
  // bind AFTER the image has natural dimensions, otherwise it locks
  // `bounds: true` against a 0×0 stage and wheel/drag never escape.
  // AVIRIS-NG 4096-tall PNGs decode slowly enough that the previous
  // mount-time binding fired too early.
  const [rgbLoaded, setRgbLoaded] = useState(false);

  // Safari + some Chrome cache states skip the `load` event for cached
  // images — poll the ref's natural dimensions after mount to catch that.
  useEffect(() => {
    if (rgbLoaded) return;
    let cancelled = false;
    let tries = 0;
    const tick = () => {
      if (cancelled) return;
      const img = rgbImgRef.current;
      if (img && img.naturalWidth > 0) {
        setRgbLoaded(true);
        return;
      }
      if (tries++ < 50) {
        setTimeout(tick, 50);    // up to ~2.5 s of polling
      }
    };
    tick();
    return () => { cancelled = true; };
  }, [rgbLoaded]);

  // Bind panzoom DIRECTLY to the rgb <img>, the same way ZoomablePanel
  // does in the anomaly_scoring viewer. The overlay PNG, dots canvas
  // and ring SVG follow the img via getBoundingClientRect() ticks on
  // every transform event — handled by the layer-tracker effect below.
  //
  // This works for tall+narrow images (AVIRIS-NG) because there's no
  // wrapping stage div whose intrinsic size collapses on extreme
  // aspect ratios: panzoom sees the <img> element directly.
  useEffect(() => {
    if (!rgbLoaded) return;
    let cancelled = false;
    let dispose: (() => void) | null = null;
    const setup = async () => {
      const target = rgbImgRef.current;
      if (!target) return;
      const mod = await import("panzoom");
      if (cancelled) return;
      const inst = mod.default(target, {
        maxZoom: 30,
        minZoom: 1,
        bounds: true,
        boundsPadding: 0.2,
        smoothScroll: false,
        zoomDoubleClickSpeed: 1,
      }) as unknown as PzInstance;
      pzRef.current = inst;
      dispose = () => inst.dispose();
    };
    void setup();
    return () => {
      cancelled = true;
      dispose?.();
      pzRef.current = null;
    };
  }, [rgbLoaded]);

  // Layer tracker: keep overlay/dots/ring sized + positioned to the
  // panzoom'd rgb img. Reads img.getBoundingClientRect() relative to
  // the stage container after every transform tick.
  const [layerBox, setLayerBox] = useState<{
    left: number; top: number; width: number; height: number;
  } | null>(null);
  useEffect(() => {
    if (!rgbLoaded) return;
    let raf = 0;
    let cancelled = false;
    const update = () => {
      if (cancelled) return;
      const img = rgbImgRef.current;
      const stage = stageRef.current;
      if (!img || !stage) return;
      // Stage now has aspect-ratio matching rgb's natural ratio, so the
      // img element fills it with no letter/pillar-boxing. The element
      // rect IS the painted rect — same approach ZoomablePanel uses.
      const ir = img.getBoundingClientRect();
      const sr = stage.getBoundingClientRect();
      setLayerBox({
        left: ir.left - sr.left,
        top: ir.top - sr.top,
        width: ir.width,
        height: ir.height,
      });
    };
    const onTransform = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(update);
    };
    update();
    let intervalId: number | null = null;
    // Hook panzoom transform events once the instance arrives.
    const hookPanzoom = () => {
      const inst = pzRef.current;
      if (inst) {
        inst.on("transform", onTransform);
      } else {
        intervalId = window.setTimeout(hookPanzoom, 50);
      }
    };
    hookPanzoom();
    window.addEventListener("resize", update);
    return () => {
      cancelled = true;
      if (raf) cancelAnimationFrame(raf);
      if (intervalId !== null) window.clearTimeout(intervalId);
      window.removeEventListener("resize", update);
    };
  }, [rgbLoaded]);

  const handleReset = () => {
    const inst = pzRef.current;
    if (!inst) return;
    inst.zoomAbs(0, 0, 1);
    inst.moveTo(0, 0);
  };
  const handleDoubleClick = handleReset;

  // The rgb.png inside the stage is what we measure to convert mouse →
  // (row, col). Use its natural size as the source coordinate space.
  const rgbImgRef = useRef<HTMLImageElement | null>(null);
  const dotsCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Pulse animation. Per-pixel SVG would tank rendering; a single canvas
  // redraw at ~30 Hz is cheap even at 9k circles. Phase is global, so the
  // dots breathe in unison — easy on the eye.
  useEffect(() => {
    let raf = 0;
    const draw = (t: number) => {
      const canvas = dotsCanvasRef.current;
      const img = rgbImgRef.current;
      if (!canvas || !img || !bundle || !img.naturalWidth) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const W = img.naturalWidth;
      const H = img.naturalHeight;
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        raf = requestAnimationFrame(draw);
        return;
      }
      ctx.clearRect(0, 0, W, H);

      // Pulse: r oscillates ~2 → 3.6 (image px) at ~1.4 Hz, alpha 0.6 → 1.0.
      // Bigger and more aggressive than the yellow pass — red has to cut
      // through varied RGB backgrounds, so we lean on contrast not subtlety.
      const phase = (t / 700) % (Math.PI * 2);
      const r = 2.6 + 1.0 * Math.sin(phase);
      const alpha = 0.8 + 0.2 * Math.sin(phase);

      // Red ring + pink-red bright core. One colour for all dots so they
      // read as "look here," not "what is this." Material identity is on
      // the underlying overlay PNG and in the side rail.
      ctx.strokeStyle = `rgba(220, 38, 38, ${alpha.toFixed(3)})`;       // red-600
      ctx.lineWidth = 0.7;
      ctx.fillStyle = `rgba(248, 113, 113, ${(alpha * 0.65).toFixed(3)})`; // red-400

      // Iterate matched-pixel rows/cols from the bundle's spectraByCoord
      // keys. widthKey decodes back to (row, col).
      const widthKey = bundle.width;
      const matchedSet = bundle.matchedSet;
      // For perf at heavy zooms we could clip to the visible rect, but
      // 9k strokes on a 1280×1216 canvas is already <2 ms on a laptop.
      matchedSet.forEach((key) => {
        const row = Math.floor(key / widthKey);
        const col = key - row * widthKey;
        ctx.beginPath();
        ctx.arc(col + 0.5, row + 0.5, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [bundle]);

  // Save current view as a PNG.
  const handleSaveView = async () => {
    const img = rgbImgRef.current;
    const canvas = dotsCanvasRef.current;
    if (!img || !canvas) return;
    // Build a flat composite at the rgb's natural size: rgb + overlay + dots.
    const W = img.naturalWidth;
    const H = img.naturalHeight;
    const out = document.createElement("canvas");
    out.width = W;
    out.height = H;
    const octx = out.getContext("2d");
    if (!octx) return;
    octx.drawImage(img, 0, 0, W, H);
    // Re-fetch the overlay as a blob so cross-origin canvas tainting
    // doesn't block toBlob() — same-origin fetch is fine.
    const overlayImg = new Image();
    overlayImg.crossOrigin = "anonymous";
    await new Promise<void>((resolve, reject) => {
      overlayImg.onload = () => resolve();
      overlayImg.onerror = () => reject(new Error("overlay load failed"));
      overlayImg.src = overlayUrl;
    });
    octx.drawImage(overlayImg, 0, 0, W, H);
    octx.drawImage(canvas, 0, 0, W, H);
    out.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `spectral_match_${actionId.replace(/^action_/, "").slice(0, 8)}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    }, "image/png");
  };

  // Export submission-ready zip (hyper bundle for splib match).
  const [exportBusy, setExportBusy] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const handleExport = async () => {
    setExportBusy(true);
    setExportError(null);
    try {
      const r = await fetch(
        `/api/actions/${encodeURIComponent(actionId)}/export`,
        { method: "POST" },
      );
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const j = await r.json();
          detail = j.detail ?? detail;
        } catch { /* not json */ }
        if (detail === "crs_missing") {
          throw new Error(
            "Scene has no CRS — submission rules require a georeferenced GeoTIFF.",
          );
        }
        throw new Error(`Export failed: ${detail}`);
      }
      const blob = await r.blob();
      const filename =
        r.headers.get("Content-Disposition")
          ?.match(/filename="([^"]+)"/)?.[1]
        ?? `hyper_${actionId.replace(/^action_/, "").slice(0, 8)}.zip`;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err) {
      setExportError(String((err as Error).message ?? err));
    } finally {
      setExportBusy(false);
    }
  };

  const rawPixelFromMouse = (e: React.MouseEvent<HTMLElement>) => {
    const img = rgbImgRef.current;
    if (!img || !img.naturalWidth || !img.naturalHeight) return null;
    // With the stage's aspect-ratio matched to the rgb's natural aspect,
    // there is no letter/pillar-boxing inside the img element — the
    // visible pixels fill its box exactly. `getBoundingClientRect()` is
    // then the painted rect; this is the same simple math ZoomablePanel
    // uses for AVIRIS-NG in the anomaly_scoring viewer.
    const rect = img.getBoundingClientRect();
    const sx = img.naturalWidth / rect.width;
    const sy = img.naturalHeight / rect.height;
    const col = Math.floor((e.clientX - rect.left) * sx);
    const row = Math.floor((e.clientY - rect.top) * sy);
    if (row < 0 || col < 0 || row >= img.naturalHeight || col >= img.naturalWidth) {
      return null;
    }
    return { row, col };
  };

  // Snap to the nearest matched pixel within a small radius (in source
  // pixels). Search is brute force over `matchedSet` keys — at ~9k entries
  // it's instant.
  const SNAP_RADIUS = 6;
  const snapToMatched = (raw: { row: number; col: number }) => {
    if (!bundle) return null;
    const widthKey = bundle.width;
    // Quick check: exact match?
    if (bundle.matchedSet.has(raw.row * widthKey + raw.col)) return raw;
    // Else scan a small square.
    let best: { row: number; col: number; d2: number } | null = null;
    const r2max = SNAP_RADIUS * SNAP_RADIUS;
    for (let dr = -SNAP_RADIUS; dr <= SNAP_RADIUS; dr++) {
      for (let dc = -SNAP_RADIUS; dc <= SNAP_RADIUS; dc++) {
        const d2 = dr * dr + dc * dc;
        if (d2 > r2max) continue;
        const r = raw.row + dr;
        const c = raw.col + dc;
        const key = r * widthKey + c;
        if (bundle.matchedSet.has(key)) {
          if (!best || d2 < best.d2) best = { row: r, col: c, d2 };
        }
      }
    }
    return best ? { row: best.row, col: best.col } : null;
  };

  const onStageMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const raw = rawPixelFromMouse(e);
    if (!raw) { setHoverPx(null); return; }
    setHoverPx(snapToMatched(raw));
  };

  const onStageMouseLeave = () => setHoverPx(null);

  const onStageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const raw = rawPixelFromMouse(e);
    if (!raw) return;
    const snapped = snapToMatched(raw);
    if (!snapped) return;
    setPinnedPx(snapped);
  };

  const activePx = pinnedPx ?? hoverPx;
  const isPinned = pinnedPx != null;

  // Build chart series for the active pixel.
  const chartData = (() => {
    if (!bundle || !activePx) return null;
    const widthKey = bundle.width;
    const key = activePx.row * widthKey + activePx.col;
    const spec = bundle.spectraByCoord.get(key);
    const val = bundle.validityByCoord.get(key);
    const matches = bundle.matchesByCoord.get(key) ?? [];
    if (!spec || !val) return null;

    // Drop sensor-flagged zero-wavelength bands AND sort ascending.
    const wlAll = bundle.wavelengths;
    const idxs = wlAll
      .map((w, i) => [w, i] as [number, number])
      .filter(([w]) => w > 0)
      .sort((a, b) => a[0] - b[0]);
    const wls = idxs.map(([w]) => w);
    const toSeries = (vals: ArrayLike<number>, validity: ArrayLike<number>) =>
      idxs.map(([, i]) => (validity[i] ? vals[i] : NaN));

    const top1 = matches[0];
    const top1Refl = top1
      ? bundle.libRefl[top1.library_ix]
      : null;
    const top1Valid = top1 ? bundle.libValid[top1.library_ix] : null;

    // L2-normalise pixel and library on their joint-valid band set —
    // matches what SAM actually consumes (cos θ = u·l / |u||l|, where the
    // norm is over the intersection of valid bands). Without this the
    // chart shows raw reflectance which differs in amplitude even when
    // SAM angle is small.
    const pixelSeriesRaw = toSeries(spec, val);
    const libSeriesRaw =
      top1Refl && top1Valid ? toSeries(top1Refl, top1Valid) : null;
    const jointValid = libSeriesRaw
      ? idxs.map(([, i], j) =>
          val[i] && top1Valid![i] && Number.isFinite(pixelSeriesRaw[j]) &&
          Number.isFinite(libSeriesRaw[j]),
        )
      : idxs.map(([, i], j) =>
          val[i] && Number.isFinite(pixelSeriesRaw[j]),
        );
    const pixelNorm = Math.sqrt(
      pixelSeriesRaw.reduce(
        (s, v, j) => (jointValid[j] ? s + v * v : s),
        0,
      ),
    ) || 1;
    const libNorm = libSeriesRaw
      ? Math.sqrt(
          libSeriesRaw.reduce(
            (s, v, j) => (jointValid[j] ? s + v * v : s),
            0,
          ),
        ) || 1
      : 1;
    const pixelSeries = pixelSeriesRaw.map((v) => v / pixelNorm);
    const libSeries = libSeriesRaw
      ? libSeriesRaw.map((v) => v / libNorm)
      : null;

    return {
      wavelengths: wls,
      series: [
        {
          label: "pixel (L2)",
          color: isPinned ? "#0ea5e9" : "rgba(14,165,233,0.7)",
          values: pixelSeries,
        },
        ...(top1 && libSeries
          ? [{
              label: `top-1: ${top1.name} (L2)`,
              color: isPinned ? "#db2777" : "rgba(219,39,119,0.7)",
              values: libSeries,
              dashed: true,
            }]
          : []),
      ],
      matches,
    };
  })();

  // Render snap-ring at the active pixel by overlaying an SVG that rides
  // the same transform as the rgb image.
  const ringPx = activePx;

  return (
    <div className="anomaly-viewer__modal-backdrop" onClick={onClose}>
      <div
        className="anomaly-viewer__modal spectral-match-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <header className="anomaly-viewer__modal-header">
          <h3>Spectral library match · USGS splib07</h3>
          <button
            type="button"
            className="anomaly-viewer__modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="spectral-match-modal__layout">
          <section
            ref={containerRef}
            className="spectral-match-modal__map"
            onDoubleClick={handleDoubleClick}
            onMouseMove={onStageMouseMove}
            onMouseLeave={onStageMouseLeave}
            onClick={onStageClick}
          >
            <div className="spectral-match-modal__toolbar">
              <button
                type="button"
                className="spectral-match-modal__tool-btn"
                onClick={handleReset}
                title="Reset zoom + pan"
              >
                Reset
              </button>
              <button
                type="button"
                className="spectral-match-modal__tool-btn"
                onClick={handleSaveView}
                title="Save the current view as a PNG"
              >
                Save view
              </button>
              <button
                type="button"
                className="spectral-match-modal__tool-btn spectral-match-modal__tool-btn--primary"
                onClick={handleExport}
                disabled={exportBusy}
                title="Download submission bundle (hyper GeoTIFF + SHP + CSV + manifest)"
              >
                {exportBusy ? "Exporting…" : "Export bundle"}
              </button>
              {exportError && (
                <span
                  className="spectral-match-modal__tool-error"
                  title={exportError}
                >
                  {exportError.length > 60
                    ? exportError.slice(0, 60) + "…"
                    : exportError}
                </span>
              )}
            </div>
            <div
              ref={stageRef}
              className="spectral-match-modal__stage"
              style={
                rgbImgRef.current?.naturalWidth && rgbImgRef.current?.naturalHeight
                  ? {
                      aspectRatio: `${rgbImgRef.current.naturalWidth} / ${rgbImgRef.current.naturalHeight}`,
                    }
                  : undefined
              }
            >
              <img
                ref={rgbImgRef}
                src={rgbUrl}
                alt="Scene RGB"
                className="spectral-match-modal__layer spectral-match-modal__layer--rgb"
                draggable={false}
                onLoad={() => setRgbLoaded(true)}
              />
              {/* Follower layers: position + size match the panzoom'd
                  rgb img via layerBox (updated on every transform tick).
                  This is the same pattern ZoomablePanel uses for GT dots. */}
              {layerBox && (
                <img
                  src={overlayUrl}
                  alt="Match angle heatmap"
                  className="spectral-match-modal__layer spectral-match-modal__layer--overlay"
                  draggable={false}
                  style={{
                    position: "absolute",
                    left: layerBox.left,
                    top: layerBox.top,
                    width: layerBox.width,
                    height: layerBox.height,
                    pointerEvents: "none",
                  }}
                />
              )}
              {layerBox && (
                <canvas
                  ref={dotsCanvasRef}
                  className="spectral-match-modal__layer spectral-match-modal__layer--dots"
                  style={{
                    position: "absolute",
                    left: layerBox.left,
                    top: layerBox.top,
                    width: layerBox.width,
                    height: layerBox.height,
                    pointerEvents: "none",
                  }}
                />
              )}
              {ringPx && rgbImgRef.current && layerBox && (
                <svg
                  className="spectral-match-modal__layer spectral-match-modal__ring-svg"
                  viewBox={`0 0 ${rgbImgRef.current.naturalWidth} ${rgbImgRef.current.naturalHeight}`}
                  preserveAspectRatio="none"
                  style={{
                    position: "absolute",
                    left: layerBox.left,
                    top: layerBox.top,
                    width: layerBox.width,
                    height: layerBox.height,
                    pointerEvents: "none",
                  }}
                >
                  <circle
                    cx={ringPx.col + 0.5}
                    cy={ringPx.row + 0.5}
                    r={isPinned ? 4 : 3}
                    fill="none"
                    stroke={isPinned ? "#fde68a" : "#fbbf24"}
                    strokeWidth={isPinned ? 1.5 : 1}
                  />
                </svg>
              )}
            </div>
            <p className="spectral-match-modal__hint">
              Scroll = zoom · drag = pan · dbl-click = reset · hover = preview · click = pin
            </p>
            {!bundle && !bundleError && (
              <span className="spectral-match-modal__busy">loading…</span>
            )}
            {bundleError && (
              <span
                className="spectral-match-modal__busy"
                style={{
                  color: "#fca5a5",
                  fontSize: "0.7rem",
                  maxWidth: "60%",
                  whiteSpace: "normal",
                  textAlign: "right",
                }}
                title={bundleError}
              >
                {bundleError.length > 200 ? bundleError.slice(0, 200) + "…" : bundleError}
              </span>
            )}
          </section>

          <aside className="spectral-match-modal__rail">
            <section className="spectral-match-modal__panel">
              <h5>Spectrum</h5>
              {!activePx && (
                <p className="muted">Hover or click a matched pixel.</p>
              )}
              {activePx && !chartData && (
                <p className="muted">Loading…</p>
              )}
              {activePx && chartData && (
                <>
                  <SpectrumChart
                    wavelengths={chartData.wavelengths}
                    series={chartData.series}
                    height={220}
                    yLabel="L2-normalised reflectance"
                  />
                  <p className="muted small">
                    row {activePx.row} · col {activePx.col}
                    {isPinned ? " · pinned" : " · preview"}
                  </p>
                </>
              )}
            </section>

            <section className="spectral-match-modal__panel">
              <h5>At pixel · top {s.top_k ?? "?"}</h5>
              {!activePx && <p className="muted">No probe yet.</p>}
              {activePx && chartData && chartData.matches.length === 0 && (
                <p className="muted">No match at this pixel.</p>
              )}
              {activePx && chartData && chartData.matches.length > 0 && (
                <ol className="spectral-match-modal__match-list">
                  {chartData.matches.map((m) => (
                    <li key={m.rank}>
                      <div className="spectral-match-modal__match-row">
                        <strong>{m.name}</strong>
                        <span className="mono small">{m.angle_deg.toFixed(2)}°</span>
                      </div>
                      <div className="muted small">
                        {m.chapter}
                        {m.asd_subtype ? ` · ${m.asd_subtype}` : ""} ·{" "}
                        <span className="mono">{m.n_bands_used} bands</span>
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </section>

            <section className="spectral-match-modal__panel">
              <h5>Top materials</h5>
              {histogram == null && <p className="muted">Loading histogram…</p>}
              {histogram && histogram.length === 0 && (
                <p className="muted">No matches in histogram.</p>
              )}
              {histogram && histogram.length > 0 && (
                <ol className="spectral-match-modal__hist-list">
                  {histogram.slice(0, 12).map((row) => (
                    <li key={row.library_ix}>
                      <span className="mono small">{row.count}</span>{" "}
                      <strong>{row.name}</strong>{" "}
                      <span className="muted small">({row.chapter})</span>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}

interface MatchRow {
  row: number;
  col: number;
  rank: number;
  library_ix: number;
  material_id: string;
  name: string;
  chapter: string;
  asd_subtype: string | null;
  angle_deg: number;
  n_bands_used: number;
}
