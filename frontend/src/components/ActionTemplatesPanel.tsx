// ActionTemplate management for the Models destination (Step 18).
//
// Lists system + user templates, lets a user create / rename / re-
// describe / re-configure / delete their own (system templates are
// read-only per backend 409). Grouped by action type so the picker
// experience mirrors what the NewActionDialog already does.
//
// Keep this deliberately low-ceremony: a textarea for the JSON
// configuration is enough until per-type form schemas land. Bad JSON
// shows the parse error inline.

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  createActionTemplate,
  deleteActionTemplate,
  listActionTemplates,
  patchActionTemplate,
  type ActionTemplate,
} from "../api/actionTemplates";
import { ApiError } from "../api/client";
import type { ActionTypeMeta } from "../types";

interface Props {
  catalog: ActionTypeMeta[] | null; // pulled by parent so dialogs / picker share state
}

interface EditState {
  id: string;
  name: string;
  description: string;
  configurationText: string;
}

function parseConfig(text: string): { value?: Record<string, unknown>; error?: string } {
  const trimmed = text.trim();
  if (trimmed.length === 0) return { value: {} };
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return { error: "Configuration must be a JSON object." };
    }
    return { value: parsed as Record<string, unknown> };
  } catch (err) {
    return { error: (err as Error).message };
  }
}

export function ActionTemplatesPanel({ catalog }: Props) {
  const [templates, setTemplates] = useState<ActionTemplate[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0);
  const [creating, setCreating] = useState(false);
  const [createType, setCreateType] = useState<string>("");
  const [createName, setCreateName] = useState("");
  const [createDescription, setCreateDescription] = useState("");
  const [createConfigText, setCreateConfigText] = useState("{}");
  const [createError, setCreateError] = useState<string | null>(null);
  const [editing, setEditing] = useState<EditState | null>(null);
  const [editError, setEditError] = useState<string | null>(null);

  // Default the new-template type to the first catalog entry once loaded.
  useEffect(() => {
    if (!createType && catalog && catalog.length > 0) {
      setCreateType(catalog[0].type);
    }
  }, [catalog, createType]);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    listActionTemplates()
      .then((page) => {
        if (!cancelled) setTemplates(page.items);
      })
      .catch((err) => {
        if (!cancelled)
          setError(
            err instanceof ApiError
              ? err.detail ?? `HTTP ${err.status}`
              : "Could not reach the server.",
          );
      });
    return () => {
      cancelled = true;
    };
  }, [reloadTick]);

  const byType = useMemo(() => {
    const out = new Map<string, ActionTemplate[]>();
    for (const t of templates ?? []) {
      const arr = out.get(t.type);
      if (arr) arr.push(t);
      else out.set(t.type, [t]);
    }
    return out;
  }, [templates]);

  const typeLabel = useCallback(
    (slug: string) => catalog?.find((m) => m.type === slug)?.label ?? slug,
    [catalog],
  );

  const onSubmitCreate = useCallback(async () => {
    setCreateError(null);
    const cfg = parseConfig(createConfigText);
    if (cfg.error) {
      setCreateError(`Configuration JSON: ${cfg.error}`);
      return;
    }
    if (!createType.trim() || !createName.trim()) {
      setCreateError("Type and name are required.");
      return;
    }
    try {
      await createActionTemplate({
        type: createType,
        name: createName.trim(),
        description: createDescription.trim() || undefined,
        configuration: cfg.value,
      });
      setCreating(false);
      setCreateName("");
      setCreateDescription("");
      setCreateConfigText("{}");
      setReloadTick((t) => t + 1);
    } catch (err) {
      setCreateError(
        err instanceof ApiError
          ? err.detail ?? `HTTP ${err.status}`
          : "Save failed.",
      );
    }
  }, [createType, createName, createDescription, createConfigText]);

  const onSubmitEdit = useCallback(async () => {
    if (!editing) return;
    setEditError(null);
    const cfg = parseConfig(editing.configurationText);
    if (cfg.error) {
      setEditError(`Configuration JSON: ${cfg.error}`);
      return;
    }
    try {
      await patchActionTemplate(editing.id, {
        name: editing.name.trim(),
        description: editing.description,
        configuration: cfg.value,
      });
      setEditing(null);
      setReloadTick((t) => t + 1);
    } catch (err) {
      setEditError(
        err instanceof ApiError
          ? err.detail ?? `HTTP ${err.status}`
          : "Save failed.",
      );
    }
  }, [editing]);

  const onDelete = useCallback(async (t: ActionTemplate) => {
    if (!window.confirm(`Delete template "${t.name}"?`)) return;
    try {
      await deleteActionTemplate(t.id);
      setReloadTick((tk) => tk + 1);
    } catch (err) {
      window.alert(
        err instanceof ApiError ? `Delete failed: ${err.status}` : "Delete failed",
      );
    }
  }, []);

  return (
    <section className="templates-panel">
      <header className="templates-panel__header">
        <div>
          <h2 className="templates-panel__title">Action templates</h2>
          <p className="templates-panel__sub">
            Reusable recipes for Action submission. System templates seed
            the picker per Action type; your saved templates appear below
            the system ones in <code>NewActionDialog</code>.
          </p>
        </div>
        {!creating && (
          <button
            type="button"
            className="anomaly-viewer__tool-btn"
            onClick={() => {
              setCreating(true);
              setCreateError(null);
            }}
          >
            + New template
          </button>
        )}
      </header>

      {error && <p className="form__error" role="alert">{error}</p>}

      {creating && (
        <div className="templates-panel__editor">
          <h4>New template</h4>
          <label className="form__label">
            Action type
            <select
              className="form__select"
              value={createType}
              onChange={(e) => setCreateType(e.target.value)}
            >
              {(catalog ?? []).map((m) => (
                <option key={m.type} value={m.type}>
                  {m.label} ({m.type})
                </option>
              ))}
            </select>
          </label>
          <label className="form__label">
            Name
            <input
              className="form__input"
              value={createName}
              onChange={(e) => setCreateName(e.target.value)}
              placeholder="e.g. PRISMA · band-filter for hazy scenes"
            />
          </label>
          <label className="form__label">
            Description
            <input
              className="form__input"
              value={createDescription}
              onChange={(e) => setCreateDescription(e.target.value)}
              placeholder="optional"
            />
          </label>
          <label className="form__label">
            Configuration (JSON)
            <textarea
              className="form__input templates-panel__code"
              rows={8}
              value={createConfigText}
              onChange={(e) => setCreateConfigText(e.target.value)}
            />
          </label>
          {createError && <p className="form__error" role="alert">{createError}</p>}
          <div className="templates-panel__actions">
            <button
              type="button"
              className="anomaly-viewer__tool-btn"
              onClick={() => setCreating(false)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="anomaly-viewer__tool-btn"
              data-active="true"
              onClick={() => void onSubmitCreate()}
            >
              Create
            </button>
          </div>
        </div>
      )}

      {templates === null && !error && (
        <p className="page__hint">Loading templates…</p>
      )}

      {templates !== null && templates.length === 0 && !creating && (
        <div className="workspace__empty">
          <p>
            No templates yet. Run{" "}
            <code>python -m allotrope.cli seed-action-templates</code> to
            seed the system defaults, or use <strong>New template</strong>
            to save your own.
          </p>
        </div>
      )}

      {templates !== null && byType.size > 0 && (
        <div className="templates-panel__groups">
          {Array.from(byType.entries()).map(([type, group]) => (
            <div key={type} className="templates-panel__group">
              <h3 className="templates-panel__group-title">
                {typeLabel(type)}{" "}
                <code className="small">{type}</code>
              </h3>
              <ul className="templates-panel__list">
                {group.map((t) =>
                  editing && editing.id === t.id ? (
                    <li key={t.id} className="templates-panel__row">
                      <div className="templates-panel__editor templates-panel__editor--inline">
                        <label className="form__label">
                          Name
                          <input
                            className="form__input"
                            value={editing.name}
                            onChange={(e) =>
                              setEditing({ ...editing, name: e.target.value })
                            }
                          />
                        </label>
                        <label className="form__label">
                          Description
                          <input
                            className="form__input"
                            value={editing.description}
                            onChange={(e) =>
                              setEditing({
                                ...editing,
                                description: e.target.value,
                              })
                            }
                          />
                        </label>
                        <label className="form__label">
                          Configuration (JSON)
                          <textarea
                            className="form__input templates-panel__code"
                            rows={8}
                            value={editing.configurationText}
                            onChange={(e) =>
                              setEditing({
                                ...editing,
                                configurationText: e.target.value,
                              })
                            }
                          />
                        </label>
                        {editError && (
                          <p className="form__error" role="alert">
                            {editError}
                          </p>
                        )}
                        <div className="templates-panel__actions">
                          <button
                            type="button"
                            className="anomaly-viewer__tool-btn"
                            onClick={() => setEditing(null)}
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            className="anomaly-viewer__tool-btn"
                            data-active="true"
                            onClick={() => void onSubmitEdit()}
                          >
                            Save
                          </button>
                        </div>
                      </div>
                    </li>
                  ) : (
                    <li key={t.id} className="templates-panel__row">
                      <div className="templates-panel__row-head">
                        <div>
                          <span className="templates-panel__row-name">
                            {t.name}
                          </span>
                          {t.is_system ? (
                            <span className="templates-panel__badge templates-panel__badge--system">
                              system
                            </span>
                          ) : (
                            <span className="templates-panel__badge templates-panel__badge--user">
                              user
                            </span>
                          )}
                        </div>
                        <div className="templates-panel__row-actions">
                          {!t.is_system && (
                            <>
                              <button
                                type="button"
                                className="anomaly-viewer__tool-btn"
                                onClick={() => {
                                  setEditing({
                                    id: t.id,
                                    name: t.name,
                                    description: t.description ?? "",
                                    configurationText: JSON.stringify(
                                      t.configuration,
                                      null,
                                      2,
                                    ),
                                  });
                                  setEditError(null);
                                }}
                              >
                                Edit
                              </button>
                              <button
                                type="button"
                                className="anomaly-viewer__tool-btn"
                                onClick={() => void onDelete(t)}
                              >
                                Delete
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                      {t.description && (
                        <p className="templates-panel__row-desc">
                          {t.description}
                        </p>
                      )}
                      <details className="templates-panel__row-config">
                        <summary>configuration</summary>
                        <pre>{JSON.stringify(t.configuration, null, 2)}</pre>
                      </details>
                    </li>
                  ),
                )}
              </ul>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
