// Scenes — Library (Step 5c).
//
// Lists onboarded scenes via GET /api/scenes. Library is shared (CC-3) — any
// authenticated user can list. The Ingest button is a placeholder until the
// onboarding flow lands in Step 7.
//
// Storyboard-spec § 5 (Scenes destination): sortable, filterable table; each
// row is a Scene; clicking a row opens its Detail page (also a later step).
// Sequence diagram for the underlying call: final design/diagrams/scenes-list.drawio

import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { deleteScene, listScenes, parseSceneInUse } from "../api/scenes";
import { IngestPanel } from "../components/IngestPanel";
import { OnboardingDrawer } from "../components/OnboardingDrawer";
import type { Scene, ScenesPage, SensorType } from "../types";

const PAGE_SIZE = 50;

interface SensorOption {
  value: SensorType | "all";
  label: string;
}

const SENSOR_OPTIONS: SensorOption[] = [
  { value: "all", label: "All sensors" },
  { value: "prisma", label: "PRISMA" },
  { value: "landsat9", label: "Landsat 9" },
  { value: "enmap", label: "EnMAP" },
  { value: "aviris_ng", label: "AVIRIS-NG" },
  { value: "hotsat1", label: "HotSAT-1" },
];

const SENSOR_LABELS: Record<string, string> = {
  prisma: "PRISMA",
  landsat9: "Landsat 9",
  enmap: "EnMAP",
  aviris_ng: "AVIRIS-NG",
  hotsat1: "HotSAT-1",
};

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

function formatPercent(v: number | null): string {
  if (v === null || v === undefined) return "—";
  return `${v.toFixed(1)}%`;
}

export function ScenesPage() {
  const navigate = useNavigate();
  const [filter, setFilter] = useState<SensorType | "all">("all");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<ScenesPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ingestOpen, setIngestOpen] = useState(false);
  // Bumped after each successful onboard to re-trigger the table fetch.
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listScenes({
      limit: PAGE_SIZE,
      offset,
      sensor_type: filter === "all" ? undefined : filter,
    })
      .then((result) => {
        if (cancelled) return;
        setPage(result);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setError(err.detail ?? `Error: HTTP ${err.status}`);
        } else {
          setError("Could not reach the server.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [filter, offset, reloadToken]);

  const onIngestComplete = useCallback(() => {
    setReloadToken((t) => t + 1);
  }, []);

  const onDeleteScene = useCallback(
    async (scene: Scene) => {
      if (
        !window.confirm(
          `Delete scene "${scene.name}"? This removes its raw files, vendable, annotations, and any saved visualizations sourced from it.`,
        )
      ) {
        return;
      }
      try {
        await deleteScene(scene.id);
        setReloadToken((t) => t + 1);
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          const detail = parseSceneInUse(err.detail);
          if (detail) {
            const list = detail.blocking_project_ids.join(", ");
            window.alert(
              `Cannot delete: still referenced by ${detail.blocking_project_ids.length} project(s). Delete those first:\n${list}`,
            );
            return;
          }
        }
        window.alert(
          err instanceof ApiError
            ? `Delete failed: ${err.detail ?? err.status}`
            : "Delete failed.",
        );
      }
    },
    [],
  );

  const total = page?.total ?? 0;
  const items = page?.items ?? [];
  const rangeStart = total === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + items.length, total);
  const canPrev = offset > 0;
  const canNext = offset + PAGE_SIZE < total;

  const onChangeFilter = (v: SensorType | "all") => {
    setFilter(v);
    setOffset(0);
  };

  return (
    <div className="scenes">
      <header className="scenes__header">
        <div>
          <h2 className="scenes__title">Scenes — Library</h2>
          <p className="scenes__subtitle">
            {loading && page === null
              ? "Loading…"
              : total === 0
                ? "No scenes onboarded yet."
                : `${total.toLocaleString()} scene${total === 1 ? "" : "s"} onboarded.`}
          </p>
        </div>
        <button
          type="button"
          className="scenes__ingest"
          onClick={() => setIngestOpen(true)}
          disabled={ingestOpen}
        >
          Ingest scene
        </button>
      </header>

      {ingestOpen && (
        <IngestPanel
          onClose={() => setIngestOpen(false)}
          onComplete={onIngestComplete}
        />
      )}

      <OnboardingDrawer
        refreshSignal={reloadToken}
        onJobFinished={() => setReloadToken((t) => t + 1)}
      />

      <div className="scenes__filters" role="tablist" aria-label="Filter by sensor">
        {SENSOR_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            role="tab"
            aria-selected={filter === opt.value}
            className={
              "scenes__filter" + (filter === opt.value ? " is-active" : "")
            }
            onClick={() => onChangeFilter(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {error && (
        <p className="form__error" role="alert">
          {error}
        </p>
      )}

      {!error && items.length === 0 && !loading ? (
        <div className="scenes__empty">
          <p>
            {filter === "all"
              ? "Nothing here yet — once a scene is onboarded it'll show up in this list."
              : `No ${SENSOR_LABELS[filter] ?? filter} scenes match.`}
          </p>
          <p className="scenes__empty-step">
            Click <strong>Ingest scene</strong> above to add a PRISMA, Landsat&nbsp;9,
            EnMAP, AVIRIS-NG, or HotSAT-1 scene.
          </p>
        </div>
      ) : (
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th aria-label="Thumbnail" />
                <th>Name</th>
                <th>Sensor</th>
                <th>Acquired</th>
                <th>Cloud cover</th>
                <th>Bands</th>
                <th>Annotations</th>
                <th>Onboarded</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {items.map((s: Scene) => (
                <tr
                  key={s.id}
                  className="data-table__row--clickable"
                  onClick={() => navigate(`/scenes/${encodeURIComponent(s.id)}`)}
                  role="link"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      navigate(`/scenes/${encodeURIComponent(s.id)}`);
                    }
                  }}
                >
                  <td className="data-table__thumb">
                    {s.thumbnail_path ? (
                      <img
                        src={`/api/scenes/${encodeURIComponent(s.id)}/thumbnail`}
                        alt=""
                        loading="lazy"
                      />
                    ) : (
                      <span className="data-table__thumb-placeholder" aria-hidden="true" />
                    )}
                  </td>
                  <td className="data-table__primary">{s.name}</td>
                  <td>
                    <span className="sensor-pill">
                      {SENSOR_LABELS[s.sensor_type] ?? s.sensor_type}
                    </span>
                  </td>
                  <td>{formatDate(s.acquisition_at)}</td>
                  <td>{formatPercent(s.cloud_cover_pct)}</td>
                  <td>{s.band_count}</td>
                  <td>
                    <span
                      className={
                        "annot-dot" + (s.has_annotations ? " is-on" : "")
                      }
                      aria-label={
                        s.has_annotations ? "Has annotations" : "No annotations"
                      }
                    />
                  </td>
                  <td>{formatDate(s.created_at)}</td>
                  <td
                    className="data-table__row-actions"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      type="button"
                      className="anomaly-viewer__tool-btn"
                      title="Delete this scene from the Library"
                      onClick={(e) => {
                        e.stopPropagation();
                        void onDeleteScene(s);
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {total > 0 && (
        <div className="pagination">
          <span className="pagination__range">
            {rangeStart.toLocaleString()}–{rangeEnd.toLocaleString()} of{" "}
            {total.toLocaleString()}
          </span>
          <div className="pagination__buttons">
            <button
              type="button"
              className="pagination__btn"
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              disabled={!canPrev || loading}
            >
              Previous
            </button>
            <button
              type="button"
              className="pagination__btn"
              onClick={() => setOffset(offset + PAGE_SIZE)}
              disabled={!canNext || loading}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
