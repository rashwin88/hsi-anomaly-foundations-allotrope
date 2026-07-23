// Project workspace Visualizations tab (Step 15, scoped 2026-05-11).
//
// Lists the curated Visualization rows for a Project, supports rename /
// delete, opens the saved PNG in a lightweight modal. New rows arrive
// here from "Save view" affordances on Action / Scene viewers — this
// pane itself does not create them.

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "../api/client";
import {
  deleteVisualization,
  listVisualizations,
  patchVisualization,
  visualizationImageUrl,
  type ProjectVisualization,
} from "../api/projectVisualizations";

interface Props {
  projectId: string;
  // Bumped by the parent (or an event bus) when a new visualization is
  // saved from a viewer elsewhere on the page so this tab re-fetches.
  refreshToken?: number;
}

export function VisualizationsPane({ projectId, refreshToken = 0 }: Props) {
  const [items, setItems] = useState<ProjectVisualization[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0);
  const [openId, setOpenId] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<{
    id: string;
    name: string;
    description: string;
  } | null>(null);

  useEffect(() => {
    const onSaved = () => setReloadTick((t) => t + 1);
    window.addEventListener("allotrope:viz-saved", onSaved);
    return () => window.removeEventListener("allotrope:viz-saved", onSaved);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    listVisualizations(projectId)
      .then((page) => {
        if (!cancelled) setItems(page.items);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.detail ?? `HTTP ${err.status}`
            : "Could not reach the server.",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, reloadTick, refreshToken]);

  const onDelete = useCallback(async (vizId: string) => {
    if (!window.confirm("Delete this visualization?")) return;
    try {
      await deleteVisualization(vizId);
      setReloadTick((t) => t + 1);
    } catch (err) {
      window.alert(
        err instanceof ApiError ? `Delete failed: ${err.status}` : "Delete failed",
      );
    }
  }, []);

  const onSaveRename = useCallback(async () => {
    if (!renaming) return;
    try {
      await patchVisualization(renaming.id, {
        name: renaming.name.trim() || undefined,
        description: renaming.description,
      });
      setRenaming(null);
      setReloadTick((t) => t + 1);
    } catch (err) {
      window.alert(
        err instanceof ApiError ? `Save failed: ${err.status}` : "Save failed",
      );
    }
  }, [renaming]);

  const openItem = items?.find((it) => it.id === openId) ?? null;

  return (
    <section className="workspace__card">
      <div className="workspace__card-header">
        <h3 className="workspace__card-title">Visualizations</h3>
        <span className="workspace__count">{items?.length ?? 0}</span>
      </div>

      {error && <p className="form__error" role="alert">{error}</p>}

      {items === null && !error && (
        <div className="workspace__card-body workspace__card-body--loading">
          Loading…
        </div>
      )}

      {items !== null && items.length === 0 && (
        <div className="workspace__empty">
          <p>
            No visualizations saved yet. Open an Action's output viewer and
            use <strong>Save view</strong> to pin a frame here.
          </p>
        </div>
      )}

      {items !== null && items.length > 0 && (
        <ul className="viz-list">
          {items.map((v) => (
            <li key={v.id} className="viz-list__item" data-viz-id={v.id}>
              <button
                type="button"
                className="viz-list__thumb"
                onClick={() => setOpenId(v.id)}
                title="Open"
              >
                <img src={visualizationImageUrl(v.id)} alt={v.name} />
              </button>
              <div className="viz-list__meta">
                <div className="viz-list__name">{v.name}</div>
                {v.description && (
                  <div className="viz-list__desc">{v.description}</div>
                )}
                <div className="viz-list__sub small">
                  {v.source_kind === "scene" ? "scene" : "action output"} ·{" "}
                  {new Date(v.created_at).toLocaleString()}
                </div>
                <div className="viz-list__actions">
                  <button
                    type="button"
                    className="anomaly-viewer__tool-btn"
                    onClick={() =>
                      setRenaming({
                        id: v.id,
                        name: v.name,
                        description: v.description ?? "",
                      })
                    }
                  >
                    Rename
                  </button>
                  <button
                    type="button"
                    className="anomaly-viewer__tool-btn"
                    onClick={() => void onDelete(v.id)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      {openItem && (
        <div className="viz-modal__scrim" onClick={() => setOpenId(null)}>
          <div
            className="viz-modal__body"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="viz-modal__header">
              <h3>{openItem.name}</h3>
              <button
                type="button"
                className="anomaly-viewer__modal-close"
                onClick={() => setOpenId(null)}
                aria-label="Close"
              >
                ×
              </button>
            </header>
            <img
              className="viz-modal__img"
              src={visualizationImageUrl(openItem.id)}
              alt={openItem.name}
            />
            {openItem.description && (
              <p className="viz-modal__desc">{openItem.description}</p>
            )}
          </div>
        </div>
      )}

      {renaming && (
        <div className="viz-modal__scrim" onClick={() => setRenaming(null)}>
          <div
            className="viz-modal__body viz-modal__body--narrow"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="viz-modal__header">
              <h3>Rename visualization</h3>
            </header>
            <label className="form__label">
              Name
              <input
                className="form__input"
                value={renaming.name}
                onChange={(e) =>
                  setRenaming({ ...renaming, name: e.target.value })
                }
              />
            </label>
            <label className="form__label">
              Description
              <textarea
                className="form__input"
                rows={3}
                value={renaming.description}
                onChange={(e) =>
                  setRenaming({ ...renaming, description: e.target.value })
                }
              />
            </label>
            <div className="viz-modal__actions">
              <button
                type="button"
                className="anomaly-viewer__tool-btn"
                onClick={() => setRenaming(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="anomaly-viewer__tool-btn"
                data-active="true"
                onClick={() => void onSaveRename()}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
