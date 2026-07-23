// AnnotationsPanel — manages a scene's annotations from the left rail.
//
// Three states overlaid in one panel:
//   1. List of attached annotations (toggle visibility, opacity slider,
//      delete button).
//   2. "+ Attach annotation" affordance that flips the panel into a
//      compact form (name + file picker + submit).
//   3. While an attach job is in flight, show progress + auto-refresh
//      the list when it completes.
//
// Designed to live in the Scene Detail left rail. Pulls overlay state
// from the SceneVizController so toggling an overlay here reflects on
// the viewport in the centre column.
//
// Sequence diagram: final design/diagrams/annotation-attach.drawio (9e)

import { type FormEvent, useEffect, useRef, useState } from "react";

import {
  attachAnnotation,
  deleteAnnotation,
  getAnnotationTypeCatalog,
} from "../api/annotations";
import { ApiError } from "../api/client";
import { useJobStatus } from "../hooks/useJobStatus";
import type { AnnotationTypeCatalogItem } from "../types";
import type { SceneVizController } from "./SceneVisualizations";
import { useToast } from "./Toast";

interface Props {
  ctrl: SceneVizController;
}

export function AnnotationsPanel({ ctrl }: Props) {
  const [mode, setMode] = useState<"list" | "form">("list");
  const [pendingJobId, setPendingJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [catalog, setCatalog] = useState<AnnotationTypeCatalogItem[]>([]);
  // The kind that will be sent. Auto-set from file extension; user can
  // override via the Type select. `""` while no file picked / no
  // matching type — submit is gated on this being set.
  const [selectedKind, setSelectedKind] = useState<string>("");
  // True when the user has picked a kind manually (so we don't clobber
  // their choice on subsequent file changes).
  const [kindWasManuallyPicked, setKindWasManuallyPicked] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  useEffect(() => {
    let cancelled = false;
    getAnnotationTypeCatalog()
      .then((c) => {
        if (cancelled) return;
        setCatalog(c.items);
      })
      .catch(() => {
        // Non-fatal — the api re-validates on attach.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Union of every registered type's extensions, for the file input's
  // `accept=` attribute.
  const acceptString = catalog.length
    ? Array.from(
        new Set(catalog.flatMap((c) => c.accepted_extensions)),
      ).join(",")
    : ".tif,.tiff";

  // Which types match the current file's extension?
  const matchingKinds = file
    ? catalog.filter((c) =>
        c.accepted_extensions.some((ext) =>
          file.name.toLowerCase().endsWith(ext.toLowerCase()),
        ),
      )
    : [];

  // Auto-set selectedKind to the inferred kind when:
  //   - the file changes AND user hasn't manually picked, OR
  //   - the manual pick is no longer in the matching set
  useEffect(() => {
    if (matchingKinds.length === 0) {
      // No registered type accepts this file — clear the field; the
      // submit button is gated separately.
      if (!kindWasManuallyPicked) setSelectedKind("");
      return;
    }
    if (kindWasManuallyPicked && matchingKinds.some((m) => m.kind === selectedKind)) {
      // User's manual pick still matches the new file's extensions.
      return;
    }
    // Auto-pick the first matching kind. (Single-match → unique;
    // multi-match → first one alphabetically per the api's catalog.)
    setSelectedKind(matchingKinds[0].kind);
    setKindWasManuallyPicked(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file?.name, catalog.length]);

  const { job, error: pollError } = useJobStatus(pendingJobId);

  // When the attach job completes, refresh the annotation list and
  // close the form.
  const completedRef = useRef<string | null>(null);
  useEffect(() => {
    if (!job || !pendingJobId) return;
    if (job.status === "complete" && completedRef.current !== pendingJobId) {
      completedRef.current = pendingJobId;
      void ctrl.reloadAnnotations();
      toast.push({
        title: "Annotation attached",
        message: name || "Ready to overlay.",
        variant: "success",
        durationMs: 5000,
      });
      // Reset form state.
      setMode("list");
      setPendingJobId(null);
      setName("");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } else if (job.status === "failed" && completedRef.current !== pendingJobId) {
      completedRef.current = pendingJobId;
      toast.push({
        title: "Annotation attach failed",
        message:
          job.failure_reason?.split("\n")[0] ??
          "The worker rejected the file.",
        variant: "error",
        durationMs: 8000,
      });
      setPendingJobId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status, pendingJobId]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file || !name.trim()) return;
    setSubmitError(null);
    setSubmitting(true);
    try {
      const res = await attachAnnotation(ctrl.sceneId, {
        name: name.trim(),
        file,
        // Always send the kind we resolved on the form. Cheap and
        // explicit — avoids the api having to infer.
        kind: selectedKind || undefined,
      });
      setPendingJobId(res.job_id);
    } catch (err) {
      if (err instanceof ApiError) {
        setSubmitError(err.detail ?? `Error: HTTP ${err.status}`);
      } else {
        setSubmitError("Could not reach the server.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  const onDelete = async (annotationId: string, label: string) => {
    if (!confirm(`Delete annotation "${label}"? This removes the file.`)) return;
    try {
      await deleteAnnotation(ctrl.sceneId, annotationId);
      await ctrl.reloadAnnotations();
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

  const inFlight = pendingJobId !== null && job?.status !== "complete" && job?.status !== "failed";

  return (
    <section className="panel viz-panel annot-panel">
      <header className="annot-panel__header">
        <h3 className="panel__heading">Annotations</h3>
        {mode === "list" && !inFlight && ctrl.overlays.length > 0 && (
          <button
            type="button"
            className="annot-panel__attach-btn"
            onClick={() => setMode("form")}
          >
            + Attach
          </button>
        )}
      </header>

      {/* In-flight job progress */}
      {inFlight && (
        <div className="annot-panel__progress">
          <p className="annot-panel__job-id mono">{pendingJobId}</p>
          <p className="annot-panel__status">
            <span className={`status-pill status-pill--${job?.status ?? "queued"}`}>
              {job?.status ?? "submitting"}
            </span>
          </p>
          {pollError && (
            <p className="ingest__poll-error">Polling: {pollError}</p>
          )}
        </div>
      )}

      {/* Attach form */}
      {mode === "form" && !inFlight && (
        <form onSubmit={onSubmit} className="annot-form">
          <label className="form__field">
            <span className="form__label">Name</span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              maxLength={200}
              placeholder="e.g. GT-1"
            />
          </label>
          <label className="form__field">
            <span className="form__label">
              File
              {catalog.length > 0 && (
                <span className="form__optional">
                  {" "}({Array.from(
                    new Set(catalog.flatMap((c) => c.accepted_extensions)),
                  ).join(", ")})
                </span>
              )}
            </span>
            <input
              ref={fileInputRef}
              type="file"
              accept={acceptString}
              onChange={(e) =>
                setFile(e.target.files && e.target.files.length > 0 ? e.target.files[0] : null)
              }
              required
            />
            {file && matchingKinds.length === 0 && (
              <span className="form__error annot-form__inferred">
                No registered type accepts this extension.
              </span>
            )}
          </label>
          {/* Type field is ALWAYS shown so the user can see + override
              what they're attaching. Auto-fills from the picked file's
              extension; manual changes stick (until the file changes
              to one whose extensions don't include the picked kind). */}
          <label className="form__field">
            <span className="form__label">
              Type
              {!kindWasManuallyPicked && file && matchingKinds.length === 1 && (
                <span className="form__optional"> (auto-detected)</span>
              )}
              {!kindWasManuallyPicked && file && matchingKinds.length > 1 && (
                <span className="form__optional"> (multiple match — pick one)</span>
              )}
            </span>
            <select
              value={selectedKind}
              onChange={(e) => {
                setSelectedKind(e.target.value);
                setKindWasManuallyPicked(true);
              }}
              disabled={catalog.length === 0}
            >
              {!selectedKind && (
                <option value="" disabled>
                  {file ? "no matching type" : "(pick a file first)"}
                </option>
              )}
              {(file ? matchingKinds : catalog).map((c) => (
                <option key={c.kind} value={c.kind}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          {submitError && (
            <p className="form__error" role="alert">{submitError}</p>
          )}
          <div className="annot-form__actions">
            <button
              type="button"
              className="ingest__cancel"
              onClick={() => {
                setMode("list");
                setSubmitError(null);
              }}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="form__submit"
              disabled={
                submitting ||
                !file ||
                !name.trim() ||
                !selectedKind ||
                (catalog.length > 0 && matchingKinds.length === 0)
              }
            >
              {submitting ? "Uploading…" : "Attach"}
            </button>
          </div>
        </form>
      )}

      {/* List */}
      {mode === "list" && !inFlight && ctrl.overlays.length === 0 && (
        <>
          <p className="scene-detail__hint">
            No annotations attached.
          </p>
          <button
            type="button"
            className="annot-panel__attach-btn annot-panel__attach-btn--first"
            onClick={() => setMode("form")}
          >
            + Attach annotation
          </button>
        </>
      )}

      {mode === "list" && !inFlight && ctrl.overlays.length > 0 && (
        <ul className="annot-list">
          {ctrl.overlays.map((o) => (
            <li key={o.annotation.id} className="annot-row">
              <button
                type="button"
                className={
                  "annot-row__toggle" + (o.visible ? " is-on" : "")
                }
                onClick={() => ctrl.toggleOverlay(o.annotation.id)}
                aria-pressed={o.visible}
                title={o.visible ? "Hide overlay" : "Show overlay"}
              >
                {o.visible ? "●" : "○"}
              </button>
              <div className="annot-row__body">
                <div className="annot-row__name" title={o.annotation.name}>
                  {o.annotation.name}
                </div>
                <div className="annot-row__sliders">
                  <label className="annot-row__slider-label" title="Opacity">
                    <span aria-hidden="true">α</span>
                    <input
                      className="annot-row__opacity"
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={o.opacity}
                      onChange={(e) =>
                        ctrl.setOverlayOpacity(
                          o.annotation.id,
                          Number(e.target.value),
                        )
                      }
                      aria-label="Overlay opacity"
                      disabled={!o.visible}
                    />
                  </label>
                  <label className="annot-row__slider-label" title="Dot size">
                    <span aria-hidden="true">●</span>
                    <input
                      className="annot-row__radius"
                      type="range"
                      min={2}
                      max={60}
                      step={1}
                      value={o.radius ?? 14}
                      onChange={(e) =>
                        ctrl.setOverlayRadius(
                          o.annotation.id,
                          Number(e.target.value),
                        )
                      }
                      aria-label="Overlay dot radius"
                      disabled={!o.visible}
                    />
                  </label>
                </div>
              </div>
              <button
                type="button"
                className="annot-row__delete"
                onClick={() => onDelete(o.annotation.id, o.annotation.name)}
                title="Delete annotation"
                aria-label={`Delete ${o.annotation.name}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
