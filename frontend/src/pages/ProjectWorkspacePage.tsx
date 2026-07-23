// Project workspace shell — Step 11.
//
// Two-column layout with top-level tabs. The left rail holds the
// scene context + action list and can be collapsed to an icon column.
// The main pane is tab-driven so the Action detail isn't permanently
// fighting Result / Visualizations / Notes for horizontal width.
//
// Layout
//   header   ← All projects · name · scene pill · Export (disabled)
//   tabs     Action · Result · Visualizations · Notes
//   left     scene-context · Actions list   (collapsible)
//   main     selected tab's content (Action detail by default)
//
// Sequence diagram: final design/diagrams/project-workspace-load.drawio

import { useCallback, useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { deleteAction, listActionTypes, listProjectActions } from "../api/actions";
import { ApiError } from "../api/client";
import { getProject } from "../api/projects";
import { getScene } from "../api/scenes";
import { ActionDetailPane } from "../components/ActionDetailPane";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { NewActionDialog } from "../components/NewActionDialog";
import { NotesPane } from "../components/NotesPane";
import { ResultPanelPane } from "../components/ResultPanelPane";
import { VisualizationsPane } from "../components/VisualizationsPane";
import type { Action, ActionTypeMeta, ProjectDetail, Scene } from "../types";

type WorkspaceTab = "action" | "result" | "visualizations" | "notes";

const WORKSPACE_TABS: Array<{
  id: WorkspaceTab;
  label: string;
  hint?: string;
}> = [
  { id: "action", label: "Action" },
  { id: "result", label: "Result", hint: "wired in Step 17" },
  { id: "visualizations", label: "Visualizations" },
  { id: "notes", label: "Notes" },
];

const SENSOR_LABELS: Record<string, string> = {
  prisma: "PRISMA",
  landsat9: "Landsat 9",
  enmap: "EnMAP",
  aviris_ng: "AVIRIS-NG",
  hotsat1: "HotSAT-1",
};

// Scroll a target element into view + apply a short highlight pulse.
// Used by the note-reference click handler so the user sees where the
// note was pointing after the workspace navigates.
function pulseElement(selector: string): void {
  const el = document.querySelector(selector);
  if (!(el instanceof HTMLElement)) return;
  try {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch {
    /* older browsers fall back to instant scroll */
  }
  el.classList.add("ref-pulse");
  window.setTimeout(() => el.classList.remove("ref-pulse"), 1800);
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

function formatBbox(s: Scene): string {
  const f = (n: number) => n.toFixed(3);
  return `${f(s.bbox_min_lon)}, ${f(s.bbox_min_lat)}  →  ${f(s.bbox_max_lon)}, ${f(s.bbox_max_lat)}`;
}

export function ProjectWorkspacePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedActionId = searchParams.get("action");

  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [catalog, setCatalog] = useState<ActionTypeMeta[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Top-level workspace tab. Drives what the main pane shows.
  // Persisted in the URL so deep-links survive a reload.
  const tabParam = searchParams.get("tab");
  const activeTab: WorkspaceTab =
    tabParam === "result" ||
    tabParam === "visualizations" ||
    tabParam === "notes"
      ? tabParam
      : "action";
  const setActiveTab = useCallback(
    (next: WorkspaceTab) => {
      const params = new URLSearchParams(searchParams);
      if (next === "action") {
        params.delete("tab");
      } else {
        params.set("tab", next);
      }
      setSearchParams(params, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  // Left rail collapse — UI state, not URL-persisted.
  const [leftCollapsed, setLeftCollapsed] = useState(false);

  // Catalog is small and stable across the session — fetch once on mount.
  useEffect(() => {
    let cancelled = false;
    listActionTypes()
      .then((items) => {
        if (!cancelled) setCatalog(items);
      })
      .catch(() => {
        // Catalog failure is non-fatal; the Action card just falls back to slug.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectAction = useCallback(
    (actionId: string | null) => {
      const next = new URLSearchParams(searchParams);
      if (actionId) {
        next.set("action", actionId);
      } else {
        next.delete("action");
      }
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  // Note-reference click → navigate to the referenced entity. The
  // NotesPane fires a custom event with {kind, wireId, projectId} so
  // it doesn't need to know the workspace's routing surface. We
  // translate the event into a tab change + URL update here, then
  // pulse the target element (highlight animation) so the user's eye
  // finds it after the navigation.
  useEffect(() => {
    const handler = (ev: Event) => {
      const detail = (ev as CustomEvent).detail as
        | {
            kind: "scene" | "action" | "output" | "viz";
            wireId: string;
            projectId: string;
          }
        | undefined;
      if (!detail) return;
      const { kind, wireId } = detail;
      switch (kind) {
        case "scene": {
          // Hop to the scene detail page. Use react-router so the SPA
          // doesn't full-reload.
          window.location.href = `/scenes/${encodeURIComponent(wireId)}`;
          return;
        }
        case "action":
        case "output": {
          // Output ids reference an action_output_<uuid>; the action
          // that produced it is what the workspace selects. The
          // mention picker stores either form; for the "output" kind,
          // the wire id we received is the action_<uuid> the mention
          // helper resolved at insert time.
          setActiveTab("action");
          selectAction(wireId);
          requestAnimationFrame(() => pulseElement(`[data-action-id="${wireId}"]`));
          return;
        }
        case "viz": {
          setActiveTab("visualizations");
          requestAnimationFrame(() =>
            pulseElement(`[data-viz-id="${wireId}"]`),
          );
          return;
        }
        default:
          return;
      }
    };
    window.addEventListener("allotrope:note-ref-click", handler);
    return () =>
      window.removeEventListener("allotrope:note-ref-click", handler);
  }, [setActiveTab, selectAction]);

  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    setProject(null);
    setScene(null);
    setError(null);

    getProject(projectId)
      .then((p) => {
        if (cancelled) return;
        setProject(p);
        // Scene fetch fires once we know the scene_id. Could be issued
        // in parallel if the workspace knew the scene_id ahead of time;
        // sequential keeps error handling simple (a missing project
        // short-circuits the scene fetch).
        return getScene(p.scene_id);
      })
      .then((s) => {
        if (!cancelled && s) setScene(s);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setError(
            err.status === 404
              ? `No project named "${projectId}".`
              : err.detail ?? `Error: HTTP ${err.status}`,
          );
        } else {
          setError("Could not reach the server.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  if (!projectId) return null;

  return (
    <div className="page workspace">
      <Link to="/projects" className="model-detail__back">← All projects</Link>

      {error && <div className="page__error">{error}</div>}
      {!project && !error && (
        <div className="page__empty">Loading project…</div>
      )}

      {project && (
        <>
          {/* ---- Header bar ---- */}
          <header className="workspace__header">
            <div className="workspace__title-block">
              <h1 className="workspace__title">{project.name}</h1>
              <div className="workspace__meta">
                {scene && (
                  <Link
                    to={`/scenes/${encodeURIComponent(scene.id)}`}
                    className="sensor-pill workspace__scene-pill"
                    title="Open scene detail"
                  >
                    {SENSOR_LABELS[scene.sensor_type] ?? scene.sensor_type} ·{" "}
                    {scene.name}
                  </Link>
                )}
                <span className="workspace__id mono small">{project.id}</span>
              </div>
              {project.description && (
                <p className="workspace__description">{project.description}</p>
              )}
            </div>
            <div className="workspace__header-actions">
              <button
                type="button"
                className="workspace__export"
                onClick={() => setActiveTab("result")}
                title="Open the Result tab to bundle this project into a downloadable zip."
              >
                Export ↓
              </button>
            </div>
          </header>

          {/* ---- Top-level tabs ---- */}
          <nav
            className="workspace__tabs"
            role="tablist"
            aria-label="Project workspace sections"
          >
            {WORKSPACE_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                className="workspace__tab"
                data-active={activeTab === tab.id ? "true" : "false"}
                onClick={() => setActiveTab(tab.id)}
                title={tab.hint}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* ---- Body: collapsible rail + tab-driven main pane ---- */}
          <div
            className="workspace__body"
            data-rail-collapsed={leftCollapsed ? "true" : "false"}
          >
            <aside className="workspace__rail workspace__rail--left">
              <button
                type="button"
                className="workspace__rail-collapse"
                onClick={() => setLeftCollapsed((v) => !v)}
                aria-pressed={leftCollapsed}
                aria-label={leftCollapsed ? "Expand rail" : "Collapse rail"}
                title={leftCollapsed ? "Expand rail" : "Collapse rail"}
              >
                {leftCollapsed ? "›" : "‹"}
              </button>
              {!leftCollapsed && (
                <>
                  <SceneContextCard scene={scene} />
                  <ActionsListPane
                    projectId={project.id}
                    scene={scene}
                    selectedActionId={selectedActionId}
                    onSelectAction={selectAction}
                  />
                </>
              )}
            </aside>

            <section className="workspace__main">
              {activeTab === "action" && (
                <ErrorBoundary
                  fallback={
                    <div className="workspace__card workspace__card--center">
                      <h3 className="workspace__card-title">Action detail</h3>
                      <p className="form__error">
                        This Action's output couldn't be rendered — its
                        diagnostics are likely from an older worker version.
                        Re-run the Action to refresh the rich charts.
                      </p>
                    </div>
                  }
                >
                  <ActionDetailPane
                    actionId={selectedActionId}
                    catalog={catalog}
                  />
                </ErrorBoundary>
              )}
              {activeTab === "result" && (
                <ResultPanelPane projectId={project.id} />
              )}
              {activeTab === "visualizations" && (
                <VisualizationsPane projectId={project.id} />
              )}
              {activeTab === "notes" && (
                <NotesPane projectId={project.id} scene={scene} />
              )}
            </section>
          </div>
        </>
      )}
    </div>
  );
}

// --- Scene-context pane ----------------------------------------------

function SceneContextCard({ scene }: { scene: Scene | null }) {
  if (!scene) {
    return (
      <section className="workspace__card">
        <h3 className="workspace__card-title">Scene</h3>
        <div className="workspace__card-body workspace__card-body--loading">
          Loading scene…
        </div>
      </section>
    );
  }
  return (
    <section className="workspace__card">
      <h3 className="workspace__card-title">Scene</h3>
      <div className="workspace__scene-card">
        {scene.thumbnail_path ? (
          <Link
            to={`/scenes/${encodeURIComponent(scene.id)}`}
            className="workspace__scene-thumb"
            title="Open scene detail"
          >
            <img
              src={`/api/scenes/${encodeURIComponent(scene.id)}/thumbnail`}
              alt=""
              loading="lazy"
            />
          </Link>
        ) : (
          <div className="workspace__scene-thumb workspace__scene-thumb--empty">
            no thumbnail
          </div>
        )}
        <dl className="workspace__scene-meta">
          <dt>name</dt>
          <dd>{scene.name}</dd>
          <dt>sensor</dt>
          <dd>{SENSOR_LABELS[scene.sensor_type] ?? scene.sensor_type}</dd>
          <dt>captured</dt>
          <dd>{formatDate(scene.acquisition_at)}</dd>
          <dt>bands</dt>
          <dd>{scene.band_count}</dd>
          <dt>bbox</dt>
          <dd className="mono small">{formatBbox(scene)}</dd>
        </dl>
        <Link
          to={`/scenes/${encodeURIComponent(scene.id)}`}
          className="workspace__scene-link"
        >
          Open scene detail →
        </Link>
      </div>
    </section>
  );
}

// --- Actions list pane (Step 12g) -------------------------------------
//
// Lists Actions for the project, ordered by created_at desc. Polls every
// 3 s while any row is in queued/running so the UI animates through the
// status transitions the worker is mirroring on `actions.status`. Click
// an Action row → URL ?action=action_<uuid> (the Action card / Output
// viewer wires off that param in Step 12h).

const STATUS_LABELS: Record<Action["status"], string> = {
  queued: "queued",
  running: "running",
  complete: "complete",
  failed: "failed",
  cancelled: "cancelled",
  needs_threshold: "needs threshold",
};

function ActionsListPane({
  projectId,
  scene,
  selectedActionId,
  onSelectAction,
}: {
  projectId: string;
  scene: Scene | null;
  selectedActionId: string | null;
  onSelectAction: (actionId: string | null) => void;
}) {
  const [actions, setActions] = useState<Action[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [reloadToken, setReloadToken] = useState(0);

  const refresh = useCallback(() => {
    listProjectActions(projectId, { limit: 100 })
      .then((page) => setActions(page.items))
      .catch((err) => {
        setError(
          err instanceof ApiError ? (err.detail ?? "fetch failed") : "fetch failed",
        );
      });
  }, [projectId]);

  useEffect(() => {
    refresh();
  }, [refresh, reloadToken]);

  // Poll while anything is in flight.
  useEffect(() => {
    if (!actions) return;
    const inFlight = actions.some(
      (a) => a.status === "queued" || a.status === "running",
    );
    if (!inFlight) return;
    const id = window.setInterval(() => {
      setReloadToken((t) => t + 1);
    }, 3000);
    return () => window.clearInterval(id);
  }, [actions]);

  const onCreated = useCallback(
    (action: Action) => {
      setDialogOpen(false);
      setReloadToken((t) => t + 1);
      // Select the freshly-created Action so the center pane shows it.
      onSelectAction(action.id);
    },
    [onSelectAction],
  );

  const count = actions?.length ?? 0;

  return (
    <section className="workspace__card">
      <div className="workspace__card-header">
        <h3 className="workspace__card-title">Actions</h3>
        <span className="workspace__count">{count}</span>
      </div>

      {error && <p className="form__error" role="alert">{error}</p>}

      {!actions && !error && (
        <p className="workspace__empty"><span>Loading…</span></p>
      )}

      {actions && actions.length === 0 && (
        <p className="workspace__empty">
          <span>
            No Actions yet. Start with a <strong>band_filter_apply</strong>{" "}
            on this scene; <strong>scene_segmentation</strong> consumes its
            output.
          </span>
        </p>
      )}

      {actions && actions.length > 0 && (
        <ul className="actions-list">
          {actions.map((a) => {
            const selected = a.id === selectedActionId;
            return (
              <li
                key={a.id}
                className="actions-list__row"
                data-action-id={a.id}
                data-status={a.status}
                data-selected={selected ? "true" : "false"}
                onClick={() => onSelectAction(a.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelectAction(a.id);
                  }
                }}
              >
                <div className="actions-list__type">
                  <span>{a.type}</span>
                  <button
                    type="button"
                    className="actions-list__delete"
                    title={
                      a.status === "running"
                        ? "Cancel a running Action before deleting"
                        : "Delete this Action and its artifacts"
                    }
                    disabled={a.status === "running"}
                    onClick={async (e) => {
                      e.stopPropagation();
                      if (
                        !window.confirm(
                          `Delete ${a.type} action? This removes its output artifacts (and any visualizations sourced from it).`,
                        )
                      )
                        return;
                      try {
                        await deleteAction(a.id);
                        if (selected) onSelectAction(null);
                        setReloadToken((t) => t + 1);
                      } catch (err) {
                        window.alert(
                          err instanceof ApiError
                            ? `Delete failed: ${err.detail ?? err.status}`
                            : "Delete failed.",
                        );
                      }
                    }}
                  >
                    ×
                  </button>
                </div>
                <div className="actions-list__meta">
                  <span
                    className="actions-list__status"
                    data-status={a.status}
                  >
                    {STATUS_LABELS[a.status]}
                  </span>
                  {/* Show a green "committed" pip on
                      anomaly_detection_prep actions that have locked
                      in a threshold — these are the ones downstream
                      chained actions will be able to consume as
                      inputs once the chaining UI is wired. */}
                  {a.type === "anomaly_detection_prep" &&
                    typeof (a.configuration as Record<string, unknown>)
                      ?.committed_threshold === "number" && (
                      <span
                        className="actions-list__committed-pip"
                        title="Committed threshold — downstream actions can chain on this"
                      >
                        committed
                      </span>
                    )}
                  <span className="actions-list__time">
                    {formatRelative(a.created_at)}
                  </span>
                </div>
                {a.failure_reason && (
                  <div className="actions-list__failure">
                    {firstLine(a.failure_reason)}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      <div className="actions-list__cta">
        <button
          type="button"
          className="form__submit"
          onClick={() => setDialogOpen(true)}
          disabled={!scene}
        >
          + New Action
        </button>
      </div>

      {dialogOpen && scene && (
        <NewActionDialog
          projectId={projectId}
          sceneId={scene.id}
          sceneSensorType={scene.sensor_type}
          sceneName={scene.name}
          onClose={() => setDialogOpen(false)}
          onCreated={onCreated}
        />
      )}
    </section>
  );
}

function formatRelative(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "—";
  const delta = (Date.now() - t) / 1000;
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

function firstLine(s: string): string {
  const head = s.split("\n", 1)[0];
  // Keep the action-list rows single-line. Full message lives in the
  // Action detail pane's failure-reason block.
  return head.length > 80 ? `${head.slice(0, 80)}…` : head;
}

// ActionDetailPane is now imported from ../components/ActionDetailPane (Step 12h).


