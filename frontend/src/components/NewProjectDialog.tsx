// New Project dialog.
//
// Modal-style overlay that creates a Project bound to a Scene. Two
// invocation modes:
//   - From Scene Detail's "Create project" CTA → scene is fixed
//     (sceneId + sceneLabel passed in; scene picker hidden).
//   - From Projects landing's "+ New Project" CTA → user picks a scene
//     via a thumbnail-card list with name search and sensor filter.
//
// On success, calls onCreated(project) with the freshly-created row.
//
// Sequence diagram: final design/diagrams/projects-create.drawio

import { type FormEvent, useEffect, useMemo, useState } from "react";

import { ApiError } from "../api/client";
import { createProject } from "../api/projects";
import { listScenes } from "../api/scenes";
import type { Project, Scene, SensorType } from "../types";

interface NewProjectDialogProps {
  /** When set, the scene is fixed and the picker is hidden. */
  fixedSceneId?: string;
  /** Display label for a fixed scene (e.g. the scene's name). */
  fixedSceneLabel?: string;

  onClose: () => void;
  onCreated: (project: Project) => void;
}

const PAGE_SIZE = 10;
const SEARCH_DEBOUNCE_MS = 250;

const SENSOR_FILTERS: { value: "" | SensorType; label: string }[] = [
  { value: "", label: "All sensors" },
  { value: "prisma", label: "PRISMA" },
  { value: "landsat9", label: "Landsat 9" },
  { value: "enmap", label: "EnMAP" },
  { value: "aviris_ng", label: "AVIRIS-NG" },
  { value: "hotsat1", label: "HotSAT-1" },
];

export function NewProjectDialog({
  fixedSceneId,
  fixedSceneLabel,
  onClose,
  onCreated,
}: NewProjectDialogProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedSceneId, setSelectedSceneId] = useState<string>(
    fixedSceneId ?? "",
  );

  // --- Scene picker state (only used when no fixedSceneId) ---
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [sensorFilter, setSensorFilter] = useState<"" | SensorType>("");
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [totalScenes, setTotalScenes] = useState(0);
  const [scenesLoading, setScenesLoading] = useState(false);
  const [scenesError, setScenesError] = useState<string | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debounce the search input (250ms) so each keystroke isn't a request.
  useEffect(() => {
    if (fixedSceneId) return;
    const t = window.setTimeout(() => setSearchQuery(searchInput), SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [searchInput, fixedSceneId]);

  // Fetch up to PAGE_SIZE matching scenes whenever the query / filter changes.
  useEffect(() => {
    if (fixedSceneId) return;
    let cancelled = false;
    setScenesLoading(true);
    setScenesError(null);
    listScenes({
      limit: PAGE_SIZE,
      q: searchQuery || undefined,
      sensor_type: sensorFilter || undefined,
    })
      .then((page) => {
        if (cancelled) return;
        setScenes(page.items);
        setTotalScenes(page.total);
        // Auto-select the first row only if nothing is already selected
        // and there's at least one match.
        setSelectedSceneId((current) => {
          if (current && page.items.some((s) => s.id === current)) return current;
          return page.items[0]?.id ?? "";
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setScenesError(
          err instanceof ApiError ? (err.detail ?? "fetch failed") : "fetch failed",
        );
      })
      .finally(() => {
        if (!cancelled) setScenesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [searchQuery, sensorFilter, fixedSceneId]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!selectedSceneId || !name.trim()) return;
    setError(null);
    setSubmitting(true);
    try {
      const project = await createProject({
        scene_id: selectedSceneId,
        name: name.trim(),
        description: description.trim() || undefined,
      });
      onCreated(project);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 404 && err.detail === "scene_not_found") {
          setError("That scene no longer exists.");
        } else if (err.status === 422) {
          setError(err.detail ?? "Invalid input.");
        } else {
          setError(err.detail ?? `Error: HTTP ${err.status}`);
        }
      } else {
        setError("Could not reach the server.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  const canSubmit = !submitting && !!selectedSceneId && name.trim().length > 0;

  const overflowCount = useMemo(
    () => Math.max(totalScenes - scenes.length, 0),
    [totalScenes, scenes.length],
  );

  return (
    <div
      className="dialog-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <section
        className="dialog panel dialog--wide"
        role="dialog"
        aria-modal="true"
        aria-label="New project"
      >
        <header className="dialog__header">
          <h2 className="panel__heading">New project</h2>
          <button
            type="button"
            className="dialog__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        <form onSubmit={onSubmit} className="form">
          {fixedSceneId && fixedSceneLabel && (
            <div className="form__field">
              <span className="form__label">Scene</span>
              <p className="form__readonly mono">{fixedSceneLabel}</p>
            </div>
          )}

          {!fixedSceneId && (
            <div className="form__field scene-picker">
              <span className="form__label">Choose a scene</span>

              <div className="scene-picker__controls">
                <input
                  type="search"
                  className="scene-picker__search"
                  placeholder="Search scenes by name…"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  autoComplete="off"
                />
                <div className="scene-picker__filters">
                  {SENSOR_FILTERS.map((f) => (
                    <button
                      type="button"
                      key={f.value || "all"}
                      className="scene-picker__filter"
                      data-active={sensorFilter === f.value ? "true" : "false"}
                      onClick={() => setSensorFilter(f.value)}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>

              <div
                className="scene-picker__list"
                role="listbox"
                aria-label="Scenes"
              >
                {scenesError && (
                  <p className="form__error" role="alert">{scenesError}</p>
                )}
                {!scenesError && scenes.length === 0 && !scenesLoading && (
                  <p className="scene-picker__empty">
                    {searchQuery || sensorFilter
                      ? "No scenes match the current filters."
                      : "No onboarded scenes yet — onboard one from the Library first."}
                  </p>
                )}
                {scenes.map((s) => (
                  <SceneCard
                    key={s.id}
                    scene={s}
                    selected={s.id === selectedSceneId}
                    onSelect={() => setSelectedSceneId(s.id)}
                  />
                ))}
                {scenesLoading && (
                  <p className="scene-picker__hint">Loading…</p>
                )}
              </div>

              <p className="scene-picker__footer">
                Showing {scenes.length} of {totalScenes}
                {overflowCount > 0 && ` · refine search to surface the rest`}
              </p>
            </div>
          )}

          <label className="form__field">
            <span className="form__label">Project name</span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              maxLength={200}
              placeholder="e.g. Coastal anomaly investigation"
            />
          </label>

          <label className="form__field">
            <span className="form__label">
              Description <span className="form__optional">(optional)</span>
            </span>
            <textarea
              className="form__textarea"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              maxLength={2000}
              rows={3}
              placeholder="Notes for yourself or the team."
            />
          </label>

          {error && (
            <p className="form__error" role="alert">{error}</p>
          )}

          <div className="dialog__actions">
            <button
              type="button"
              className="ingest__cancel"
              onClick={onClose}
              disabled={submitting}
            >
              Cancel
            </button>
            <button type="submit" className="form__submit" disabled={!canSubmit}>
              {submitting ? "Creating…" : "Create project"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

// --- SceneCard ----------------------------------------------------

interface SceneCardProps {
  scene: Scene;
  selected: boolean;
  onSelect: () => void;
}

function sensorLabel(sensor: string): string {
  if (sensor === "prisma") return "PRISMA";
  if (sensor === "landsat9") return "Landsat 9";
  if (sensor === "enmap") return "EnMAP";
  if (sensor === "aviris_ng") return "AVIRIS-NG";
  if (sensor === "hotsat1") return "HotSAT-1";
  return sensor;
}

function formatDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

function SceneCard({ scene, selected, onSelect }: SceneCardProps) {
  const thumb = scene.thumbnail_path
    ? `/api/scenes/${encodeURIComponent(scene.id)}/thumbnail`
    : null;
  const captured = formatDate((scene as Scene & { captured_at?: string | null }).captured_at);
  return (
    <button
      type="button"
      className="scene-card"
      data-selected={selected ? "true" : "false"}
      onClick={onSelect}
      role="option"
      aria-selected={selected}
    >
      <div className="scene-card__thumb">
        {thumb ? (
          <img src={thumb} alt="" loading="lazy" />
        ) : (
          <div className="scene-card__thumb-placeholder">no thumb</div>
        )}
      </div>
      <div className="scene-card__body">
        <div className="scene-card__name">{scene.name}</div>
        <div className="scene-card__meta">
          <span className="sensor-pill">{sensorLabel(scene.sensor_type)}</span>
          {captured && <span className="scene-card__date">{captured}</span>}
        </div>
      </div>
      <div className="scene-card__check" aria-hidden="true">
        {selected ? "●" : ""}
      </div>
    </button>
  );
}
