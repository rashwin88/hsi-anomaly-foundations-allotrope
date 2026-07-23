// Projects landing — Step 10e.
//
// Table of all Projects across the Library. Reuses the same data-table
// shape as Scenes (same `.data-table` styles, clickable rows, keyboard
// handlers). "+ New project" CTA opens NewProjectDialog with a scene
// picker; on success the dialog closes and the table refreshes.
//
// Sequence diagram for the underlying call: projects-list.drawio

import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { deleteProject, listProjects } from "../api/projects";
import { NewProjectDialog } from "../components/NewProjectDialog";
import { useToast } from "../components/Toast";
import type { Project, ProjectsPage as ProjectsPageType } from "../types";

const PAGE_SIZE = 50;

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

export function ProjectsPage() {
  const toast = useToast();
  const navigate = useNavigate();
  const [page, setPage] = useState<ProjectsPageType | null>(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  // Bumped after each successful create / delete to refetch.
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listProjects({ limit: PAGE_SIZE, offset })
      .then((res) => {
        if (cancelled) return;
        setPage(res);
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
  }, [offset, reloadToken]);

  const onCreated = useCallback(
    (project: Project) => {
      setDialogOpen(false);
      setReloadToken((t) => t + 1);
      toast.push({
        title: "Project created",
        message: project.name,
        variant: "success",
        durationMs: 4000,
      });
    },
    [toast],
  );

  const onDelete = async (project: Project) => {
    if (
      !confirm(
        `Delete project "${project.name}"? This will cascade through any Actions, Visualizations, Notes, and Exports under it.`,
      )
    ) {
      return;
    }
    try {
      await deleteProject(project.id);
      setReloadToken((t) => t + 1);
      toast.push({
        title: "Project deleted",
        message: project.name,
        variant: "success",
        durationMs: 4000,
      });
    } catch (err) {
      const detail =
        err instanceof ApiError ? (err.detail ?? `HTTP ${err.status}`) : "delete failed";
      toast.push({
        title: "Delete failed",
        message: detail,
        variant: "error",
      });
    }
  };

  const total = page?.total ?? 0;
  const items = page?.items ?? [];
  const rangeStart = total === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + items.length, total);
  const canPrev = offset > 0;
  const canNext = offset + PAGE_SIZE < total;

  return (
    <div className="scenes">
      <header className="scenes__header">
        <div>
          <h2 className="scenes__title">Projects</h2>
          <p className="scenes__subtitle">
            {loading && page === null
              ? "Loading…"
              : total === 0
                ? "No projects yet."
                : `${total.toLocaleString()} project${total === 1 ? "" : "s"}.`}
          </p>
        </div>
        <button
          type="button"
          className="scenes__ingest"
          onClick={() => setDialogOpen(true)}
          disabled={dialogOpen}
        >
          + New project
        </button>
      </header>

      {error && <p className="form__error" role="alert">{error}</p>}

      {!error && items.length === 0 && !loading ? (
        <div className="scenes__empty">
          <p>
            Projects are workspaces bound to a Scene. Create one to start an
            investigation.
          </p>
          <p className="scenes__empty-step">
            Click <strong>+ New project</strong> above (or use the
            <strong> Create project</strong> button on a Scene Detail page).
          </p>
        </div>
      ) : (
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Scene</th>
                <th>Sensor</th>
                <th>Description</th>
                <th>Updated</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {items.map((p) => {
                const goWorkspace = () =>
                  navigate(`/projects/${encodeURIComponent(p.id)}`);
                return (
                <tr
                  key={p.id}
                  className="data-table__row--clickable"
                  onClick={goWorkspace}
                  role="link"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      goWorkspace();
                    }
                  }}
                >
                  <td className="data-table__primary">{p.name}</td>
                  <td className="mono projects-table__scene">{p.scene_name}</td>
                  <td>
                    <span className="sensor-pill">
                      {SENSOR_LABELS[p.scene_sensor_type] ?? p.scene_sensor_type}
                    </span>
                  </td>
                  <td className="projects-table__desc">
                    {p.description ?? <span className="form__optional">—</span>}
                  </td>
                  <td>{formatDate(p.updated_at)}</td>
                  <td className="projects-table__actions">
                    <button
                      type="button"
                      className="projects-table__delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        void onDelete(p);
                      }}
                      title="Delete project"
                      aria-label={`Delete ${p.name}`}
                    >
                      ×
                    </button>
                  </td>
                </tr>
                );
              })}
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

      {dialogOpen && (
        <NewProjectDialog
          onClose={() => setDialogOpen(false)}
          onCreated={onCreated}
        />
      )}
    </div>
  );
}
