// Project workspace Notes tab (Step 16).
//
// Markdown notes scoped to a project, with inline @-mention chips that
// resolve to typed NoteReferences server-side. The editor is a plain
// textarea — keep it boring; rich-text would balloon the surface area.
//
// Mention syntax: @[label](kind:wire_id)
//   where kind ∈ {action, output, viz, scene}. The chip-picker dropdown
//   inserts the canonical form. The renderer turns those tokens into
//   clickable chips; the api persists them as NoteReference rows.

import type { ReactElement, ReactNode } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { listProjectActions } from "../api/actions";
import { ApiError } from "../api/client";
import {
  createNote,
  deleteNote,
  listNotes,
  patchNote,
  type Note,
} from "../api/notes";
import { listVisualizations } from "../api/projectVisualizations";
import type { Action, Scene } from "../types";

interface MentionOption {
  label: string;
  kind: "action" | "viz" | "scene";
  wireId: string;
}

interface Props {
  projectId: string;
  scene: Scene | null;
}

const MENTION_RE = /@\[([^\]]+)\]\((action|output|viz|scene|project):([^)]+)\)/g;

function extractReferenceIds(content: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  let m: RegExpExecArray | null;
  MENTION_RE.lastIndex = 0;
  while ((m = MENTION_RE.exec(content)) !== null) {
    const wire = m[3];
    if (!seen.has(wire)) {
      seen.add(wire);
      out.push(wire);
    }
  }
  return out;
}

function renderNoteContent(
  content: string,
  projectId: string,
): ReactElement {
  // Render paragraphs separated by blank lines; within each paragraph,
  // turn mention tokens into clickable chips. Clicks dispatch a
  // ``allotrope:note-ref-click`` custom event that the workspace
  // listens for — that's the navigation/highlight integration point
  // so NotesPane doesn't have to know the full routing surface.
  const paragraphs = content.split(/\n{2,}/);
  return (
    <>
      {paragraphs.map((p, pi) => {
        const parts: ReactNode[] = [];
        let cursor = 0;
        MENTION_RE.lastIndex = 0;
        let m: RegExpExecArray | null;
        while ((m = MENTION_RE.exec(p)) !== null) {
          if (m.index > cursor) parts.push(p.slice(cursor, m.index));
          const label = m[1];
          const kind = m[2] as
            | "action"
            | "output"
            | "viz"
            | "scene";
          const wireId = m[3];
          parts.push(
            <button
              key={`${pi}-${m.index}`}
              type="button"
              className="note-chip note-chip--clickable"
              data-kind={kind}
              title={`Open ${kind} · ${wireId}`}
              onClick={() => {
                window.dispatchEvent(
                  new CustomEvent("allotrope:note-ref-click", {
                    detail: { kind, wireId, projectId },
                  }),
                );
              }}
            >
              @{label}
              <span className="note-chip__arrow" aria-hidden="true">↗</span>
            </button>,
          );
          cursor = m.index + m[0].length;
        }
        if (cursor < p.length) parts.push(p.slice(cursor));
        return (
          <p key={pi} className="note-render__p">
            {parts.length === 0 ? p : parts}
          </p>
        );
      })}
    </>
  );
}

export function NotesPane({ projectId, scene }: Props) {
  const [notes, setNotes] = useState<Note[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0);
  const [mentionOptions, setMentionOptions] = useState<MentionOption[]>([]);

  const [composing, setComposing] = useState(false);
  const [composeContent, setComposeContent] = useState("");

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");

  // Load notes.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    listNotes(projectId)
      .then((page) => {
        if (!cancelled) setNotes(page.items);
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
  }, [projectId, reloadTick]);

  // Build the mention picker once per project: scene + all actions + all
  // visualizations. We don't yet expose ActionOutputs separately; the
  // Action chip stands in (and the api accepts both).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [actionsRes, vizPage] = await Promise.all([
          listProjectActions(projectId, { limit: 200 }),
          listVisualizations(projectId, 200),
        ]);
        if (cancelled) return;
        const opts: MentionOption[] = [];
        if (scene) {
          opts.push({
            label: scene.name,
            kind: "scene",
            wireId: scene.id,
          });
        }
        for (const a of actionsRes.items as Action[]) {
          opts.push({
            label: `${a.type} · ${a.id.slice(0, 12)}`,
            kind: "action",
            wireId: a.id,
          });
        }
        for (const v of vizPage.items) {
          opts.push({ label: v.name, kind: "viz", wireId: v.id });
        }
        setMentionOptions(opts);
      } catch {
        // non-fatal — the picker just shows nothing.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, scene]);

  const insertMention = useCallback(
    (
      content: string,
      setContent: (c: string) => void,
      opt: MentionOption,
    ) => {
      const chip = `@[${opt.label}](${opt.kind}:${opt.wireId})`;
      // If the textarea has a selection, replace it; otherwise append.
      const sep = content.length > 0 && !content.endsWith(" ") ? " " : "";
      setContent(`${content}${sep}${chip} `);
    },
    [],
  );

  const onSubmitNew = useCallback(async () => {
    const content = composeContent.trim();
    if (!content) return;
    try {
      await createNote(projectId, {
        content,
        references: extractReferenceIds(content),
      });
      setComposeContent("");
      setComposing(false);
      setReloadTick((t) => t + 1);
    } catch (err) {
      window.alert(
        err instanceof ApiError
          ? `Save failed: ${err.detail ?? err.status}`
          : "Save failed",
      );
    }
  }, [composeContent, projectId]);

  const onSubmitEdit = useCallback(async () => {
    if (!editingId) return;
    try {
      await patchNote(editingId, {
        content: editContent,
        references: extractReferenceIds(editContent),
      });
      setEditingId(null);
      setReloadTick((t) => t + 1);
    } catch (err) {
      window.alert(
        err instanceof ApiError
          ? `Save failed: ${err.detail ?? err.status}`
          : "Save failed",
      );
    }
  }, [editingId, editContent]);

  const onDelete = useCallback(async (noteId: string) => {
    if (!window.confirm("Delete this note?")) return;
    try {
      await deleteNote(noteId);
      setReloadTick((t) => t + 1);
    } catch (err) {
      window.alert(
        err instanceof ApiError ? `Delete failed: ${err.status}` : "Delete failed",
      );
    }
  }, []);

  const sortedOptions = useMemo(
    () =>
      [...mentionOptions].sort((a, b) => {
        if (a.kind !== b.kind) {
          const order = ["scene", "action", "viz"] as const;
          return order.indexOf(a.kind) - order.indexOf(b.kind);
        }
        return a.label.localeCompare(b.label);
      }),
    [mentionOptions],
  );

  return (
    <section className="workspace__card">
      <div className="workspace__card-header">
        <h3 className="workspace__card-title">Notes</h3>
        <span className="workspace__count">{notes?.length ?? 0}</span>
        {!composing && (
          <button
            type="button"
            className="anomaly-viewer__tool-btn"
            onClick={() => {
              setComposing(true);
              setComposeContent("");
            }}
          >
            New note
          </button>
        )}
      </div>

      {error && <p className="form__error" role="alert">{error}</p>}

      {composing && (
        <NoteEditor
          content={composeContent}
          onChange={setComposeContent}
          onInsertMention={(opt) =>
            insertMention(composeContent, setComposeContent, opt)
          }
          options={sortedOptions}
          onCancel={() => {
            setComposing(false);
            setComposeContent("");
          }}
          onSave={onSubmitNew}
          saveLabel="Create note"
        />
      )}

      {notes === null && !error && (
        <div className="workspace__card-body workspace__card-body--loading">
          Loading…
        </div>
      )}

      {notes !== null && notes.length === 0 && !composing && (
        <div className="workspace__empty">
          <p>
            No notes yet. Use <strong>New note</strong> to record
            observations, link Actions, and pin reasoning.
          </p>
        </div>
      )}

      {notes !== null && notes.length > 0 && (
        <ul className="note-list">
          {notes.map((n) =>
            editingId === n.id ? (
              <li key={n.id} className="note-list__item">
                <NoteEditor
                  content={editContent}
                  onChange={setEditContent}
                  onInsertMention={(opt) =>
                    insertMention(editContent, setEditContent, opt)
                  }
                  options={sortedOptions}
                  onCancel={() => setEditingId(null)}
                  onSave={onSubmitEdit}
                  saveLabel="Save"
                />
              </li>
            ) : (
              <li key={n.id} className="note-list__item">
                <div className="note-render">{renderNoteContent(n.content, projectId)}</div>
                <div className="note-list__meta small">
                  <span>
                    {new Date(n.updated_at).toLocaleString()} ·{" "}
                    {n.references.length} reference
                    {n.references.length === 1 ? "" : "s"}
                  </span>
                  <span className="note-list__actions">
                    <button
                      type="button"
                      className="anomaly-viewer__tool-btn"
                      onClick={() => {
                        setEditingId(n.id);
                        setEditContent(n.content);
                      }}
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      className="anomaly-viewer__tool-btn"
                      onClick={() => void onDelete(n.id)}
                    >
                      Delete
                    </button>
                  </span>
                </div>
              </li>
            ),
          )}
        </ul>
      )}
    </section>
  );
}

function NoteEditor({
  content,
  onChange,
  onInsertMention,
  options,
  onCancel,
  onSave,
  saveLabel,
}: {
  content: string;
  onChange: (s: string) => void;
  onInsertMention: (opt: MentionOption) => void;
  options: MentionOption[];
  onCancel: () => void;
  onSave: () => void;
  saveLabel: string;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  return (
    <div className="note-editor">
      <textarea
        className="form__input note-editor__text"
        rows={5}
        value={content}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Markdown — paragraphs separated by blank lines. Use @-mentions to link Actions, Visualizations, or the Scene."
      />
      <div className="note-editor__bar">
        <div className="note-editor__mention">
          <button
            type="button"
            className="anomaly-viewer__tool-btn"
            onClick={() => setPickerOpen((v) => !v)}
            disabled={options.length === 0}
          >
            @ Insert reference ({options.length})
          </button>
          {pickerOpen && (
            <ul className="note-editor__menu">
              {options.map((o) => (
                <li key={`${o.kind}:${o.wireId}`}>
                  <button
                    type="button"
                    onClick={() => {
                      onInsertMention(o);
                      setPickerOpen(false);
                    }}
                  >
                    <span className="note-editor__menu-kind">{o.kind}</span>
                    <span>{o.label}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="note-editor__buttons">
          <button
            type="button"
            className="anomaly-viewer__tool-btn"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            className="anomaly-viewer__tool-btn"
            data-active="true"
            onClick={onSave}
            disabled={!content.trim()}
          >
            {saveLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
