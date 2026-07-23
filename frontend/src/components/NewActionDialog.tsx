// New Action dialog (Step 12f).
//
// Two-step modal:
//   1. Pick an action type (cards filtered by scene sensor_type)
//   2. Confirm inputs + submit (configuration = catalog defaults +
//      per-input override)
//
// The catalog payload from /action-types drives both steps — the
// description on the type card AND the input pickers in step 2 come
// from META verbatim. No type-specific UI logic lives here.
//
// Deferred to a later step:
//   - Template picker (we always submit with action_template_id=null
//     for now — the api accepts this; default configuration body comes
//     from META.default_config_per_sensor)
//   - Custom configuration editor (parameter form fields)
//   - Annotation-ref inputs (not used by the v1 action types)
//
// Sequence diagrams: action-submit · action-types-catalog · action-list

import { type FormEvent, useEffect, useMemo, useState } from "react";

import { ApiError } from "../api/client";
import {
  createAction,
  getActionOutput,
  getActionSummaryJson,
  listActionTypes,
  listProjectActions,
} from "../api/actions";
import { listAnnotations } from "../api/annotations";
import { listModels } from "../api/models";
import type {
  Action,
  ActionInputSpec,
  ActionTypeMeta,
  Annotation,
  CreateActionPayload,
  ModelSummary,
} from "../types";

interface NewActionDialogProps {
  projectId: string;            // wire format: project_<uuid>
  sceneId: string;              // wire format: scene_<uuid>
  sceneSensorType: string;      // "prisma" | "enmap" | "landsat9"
  sceneName: string;
  onClose: () => void;
  onCreated: (action: Action) => void;
}

export function NewActionDialog({
  projectId,
  sceneId,
  sceneSensorType,
  sceneName,
  onClose,
  onCreated,
}: NewActionDialogProps) {
  const [catalog, setCatalog] = useState<ActionTypeMeta[] | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<ActionTypeMeta | null>(null);
  const [inputRefs, setInputRefs] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // anomaly_scoring-specific state. Empty when not on that type.
  // model_overrides[codename] = { scoring_method?, patch_size?, stride?,
  // batch_size?, sam_l1_alpha? }. Only set fields are sent on submit.
  const [adModelCodenames, setAdModelCodenames] = useState<string[]>([]);
  const [adModelOverrides, setAdModelOverrides] = useState<
    Record<string, Partial<ModelOverrideKnobs>>
  >({});

  // scene_segmentation-specific state. Per-field overrides for bands +
  // thresholds, plus the list of classes to mask. Only fields the user
  // explicitly changed from default are emitted on submit; the backend
  // fills in the rest from the action_type META.
  const [segOverrides, setSegOverrides] = useState<SegmentationOverrides>({});

  // anomaly_detection_prep-specific state. Empty until the user picks
  // an upstream anomaly_scoring output; that triggers a fetch of its
  // summary.json to discover which algorithms ran, after which the
  // form reveals one weight input per algorithm.
  const [adpAlgorithms, setAdpAlgorithms] = useState<string[] | null>(null);
  const [adpWeights, setAdpWeights] = useState<Record<string, number>>({});
  const [adpUpstreamError, setAdpUpstreamError] = useState<string | null>(null);
  const [adpLoadingUpstream, setAdpLoadingUpstream] = useState(false);

  // Catalog fetch.
  useEffect(() => {
    let cancelled = false;
    listActionTypes()
      .then((items) => {
        if (cancelled) return;
        setCatalog(items);
      })
      .catch((err) => {
        if (cancelled) return;
        setCatalogError(
          err instanceof ApiError ? (err.detail ?? "fetch failed") : "fetch failed",
        );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // For anomaly_detection_prep: when the user picks an upstream
  // anomaly_scoring output, fetch its summary.json so we can list its
  // model_codenames and seed equal-weight inputs.
  useEffect(() => {
    if (selectedType?.type !== "anomaly_detection_prep") {
      setAdpAlgorithms(null);
      setAdpWeights({});
      setAdpUpstreamError(null);
      setAdpLoadingUpstream(false);
      return;
    }
    const upstreamId = inputRefs["input_anomaly_scoring_output_id"];
    if (!upstreamId) {
      setAdpAlgorithms(null);
      setAdpWeights({});
      setAdpUpstreamError(null);
      return;
    }
    let cancelled = false;
    setAdpLoadingUpstream(true);
    setAdpUpstreamError(null);
    (async () => {
      try {
        const out = await getActionOutput(upstreamId);
        const summary = await getActionSummaryJson(out.action_id);
        if (cancelled) return;
        const algos =
          (Array.isArray(summary.model_codenames)
            ? (summary.model_codenames as unknown[]).filter(
                (c): c is string => typeof c === "string",
              )
            : []) ?? [];
        if (algos.length === 0) {
          setAdpUpstreamError(
            "Upstream anomaly_scoring output declares no algorithms.",
          );
          setAdpAlgorithms([]);
          setAdpWeights({});
          return;
        }
        setAdpAlgorithms(algos);
        // Seed equal weights — 1.0 per algorithm.
        const seed: Record<string, number> = {};
        for (const a of algos) seed[a] = 1.0;
        setAdpWeights(seed);
      } catch (err) {
        if (cancelled) return;
        setAdpAlgorithms([]);
        setAdpWeights({});
        setAdpUpstreamError(
          err instanceof ApiError
            ? (err.detail ?? `HTTP ${err.status}`)
            : "could not resolve upstream output",
        );
      } finally {
        if (!cancelled) setAdpLoadingUpstream(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedType?.type, inputRefs]);

  // Available types = those accepting this scene's sensor.
  const availableTypes = useMemo<ActionTypeMeta[]>(() => {
    return (catalog ?? []).filter((t) =>
      t.accepted_sensor_types.includes(sceneSensorType),
    );
  }, [catalog, sceneSensorType]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!selectedType) return;
    setSubmitError(null);
    setSubmitting(true);

    // Configuration = sensor default + scene/output input overrides.
    const baseCfg =
      (selectedType.default_config_per_sensor[sceneSensorType] as
        | Record<string, unknown>
        | undefined) ?? {};
    const cfg: Record<string, unknown> = { ...baseCfg };

    for (const input of selectedType.inputs) {
      if (input.ref_kind === "scene") {
        cfg[input.key] = sceneId;
      } else if (input.ref_kind === "action_output") {
        const v = inputRefs[input.key];
        if (input.required && !v) {
          setSubmitError(`Pick a value for "${input.label}"`);
          setSubmitting(false);
          return;
        }
        if (v) cfg[input.key] = v;
      } else if (input.ref_kind === "annotation") {
        const v = inputRefs[input.key];
        if (input.required && !v) {
          setSubmitError(`Pick a value for "${input.label}"`);
          setSubmitting(false);
          return;
        }
        if (v) cfg[input.key] = v;
      }
    }

    // anomaly_scoring-specific fields: model_codenames + scoring_overrides.
    if (selectedType.type === "anomaly_scoring") {
      if (adModelCodenames.length === 0) {
        setSubmitError("Pick at least one model to run.");
        setSubmitting(false);
        return;
      }
      cfg.model_codenames = adModelCodenames;
      // Wire shape: model_overrides[codename] = { scoring_method?,
      // patch_size?, stride?, batch_size?, sam_l1_alpha? }. Only fields
      // the user explicitly set are emitted; everything else falls
      // back to the model's capability default in the worker.
      const cleanedOverrides: Record<string, Record<string, unknown>> = {};
      for (const cname of adModelCodenames) {
        const knobs = adModelOverrides[cname];
        if (!knobs) continue;
        const out: Record<string, unknown> = {};
        if (knobs.scoring_method) out.scoring_method = knobs.scoring_method;
        if (typeof knobs.patch_size === "number") out.patch_size = knobs.patch_size;
        if (typeof knobs.stride === "number") out.stride = knobs.stride;
        if (typeof knobs.batch_size === "number") out.batch_size = knobs.batch_size;
        if (typeof knobs.sam_l1_alpha === "number")
          out.sam_l1_alpha = knobs.sam_l1_alpha;
        if (typeof knobs.erosion_kernel_size === "number")
          out.erosion_kernel_size = knobs.erosion_kernel_size;
        if (typeof knobs.keep_mask_erosion_kernel_size === "number")
          out.keep_mask_erosion_kernel_size = knobs.keep_mask_erosion_kernel_size;
        if (Object.keys(out).length > 0) cleanedOverrides[cname] = out;
      }
      cfg.model_overrides = cleanedOverrides;
    }

    // scene_segmentation-specific fields: user can override the four
    // class-mask thresholds and the list of classes_to_mask. Bands
    // (red_nm / green_nm / nir_nm / vnir_brightness_end_nm) stay at
    // their sensor defaults — exposing them in the UI is overkill for
    // v1 since the worker auto-snaps to the nearest band on the
    // common 10 nm grid anyway.
    if (selectedType.type === "scene_segmentation") {
      const t: Record<string, number> = {};
      if (typeof segOverrides.ndwi_water === "number")
        t.ndwi_water = segOverrides.ndwi_water;
      if (typeof segOverrides.brightness_cloud === "number")
        t.brightness_cloud = segOverrides.brightness_cloud;
      if (typeof segOverrides.brightness_shadow === "number")
        t.brightness_shadow = segOverrides.brightness_shadow;
      if (typeof segOverrides.ndvi_vegetation === "number")
        t.ndvi_vegetation = segOverrides.ndvi_vegetation;
      if (Object.keys(t).length > 0) {
        // Merge over whatever default `thresholds` block the base
        // config carried so the user's overrides win.
        const base =
          (cfg.thresholds as Record<string, number> | undefined) ?? {};
        cfg.thresholds = { ...base, ...t };
      }
      if (segOverrides.classes_to_mask) {
        cfg.classes_to_mask = segOverrides.classes_to_mask;
      }
    }

    // anomaly_detection_prep-specific fields: algorithm_weights. The
    // upstream picker is handled by the generic input-ref machinery
    // above; this block layers the per-algorithm weights on top.
    if (selectedType.type === "anomaly_detection_prep") {
      if (!adpAlgorithms || adpAlgorithms.length === 0) {
        setSubmitError(
          "Pick an upstream anomaly_scoring output before submitting.",
        );
        setSubmitting(false);
        return;
      }
      // Build the wire dict from the user-supplied numbers. Reject
      // an all-zero configuration so the worker doesn't waste cycles
      // on a guaranteed-degenerate composite.
      const clean: Record<string, number> = {};
      let nonZero = 0;
      for (const algo of adpAlgorithms) {
        const w = Number(adpWeights[algo] ?? 1.0);
        if (!Number.isFinite(w) || w < 0) {
          setSubmitError(
            `Weight for ${algo} must be a non-negative number.`,
          );
          setSubmitting(false);
          return;
        }
        clean[algo] = w;
        if (w > 0) nonZero += 1;
      }
      if (nonZero === 0) {
        setSubmitError(
          "At least one algorithm must have a positive weight.",
        );
        setSubmitting(false);
        return;
      }
      cfg.algorithm_weights = clean;
    }

    const payload: CreateActionPayload = {
      type: selectedType.type,
      configuration: cfg,
    };

    try {
      const action = await createAction(projectId, payload);
      onCreated(action);
    } catch (err) {
      if (err instanceof ApiError) {
        setSubmitError(err.detail ?? `Error: HTTP ${err.status}`);
      } else {
        setSubmitError("Could not reach the server.");
      }
      setSubmitting(false);
    }
  };

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
        aria-label="New action"
      >
        <header className="dialog__header">
          <h2 className="panel__heading">
            New action {selectedType ? `· ${selectedType.label}` : ""}
          </h2>
          <button
            type="button"
            className="dialog__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        {catalogError && (
          <p className="form__error" role="alert">{catalogError}</p>
        )}
        {!catalog && !catalogError && (
          <p className="scene-detail__hint">Loading action catalog…</p>
        )}

        {catalog && !selectedType && (
          <ActionTypePicker
            types={availableTypes}
            sensorType={sceneSensorType}
            sceneName={sceneName}
            onPick={setSelectedType}
            onCancel={onClose}
          />
        )}

        {catalog && selectedType && (
          <form onSubmit={onSubmit} className="form">
            <ActionTypeSummary
              meta={selectedType}
              onBack={() => {
                setSelectedType(null);
                setInputRefs({});
                setSubmitError(null);
              }}
            />

            <ActionInputsForm
              meta={selectedType}
              projectId={projectId}
              sceneId={sceneId}
              sceneName={sceneName}
              sceneSensorType={sceneSensorType}
              inputRefs={inputRefs}
              onChangeRef={(key, value) =>
                setInputRefs((prev) => ({ ...prev, [key]: value }))
              }
            />

            {selectedType.type === "anomaly_scoring" && (
              <AnomalyScoringExtras
                sceneSensorType={sceneSensorType}
                modelCodenames={adModelCodenames}
                onChangeModels={setAdModelCodenames}
                overrides={adModelOverrides}
                onChangeOverrides={setAdModelOverrides}
              />
            )}

            {selectedType.type === "scene_segmentation" && (
              <SceneSegmentationExtras
                overrides={segOverrides}
                onChange={setSegOverrides}
              />
            )}

            {selectedType.type === "anomaly_detection_prep" && (
              <AnomalyDetectionPrepExtras
                upstreamSelected={Boolean(
                  inputRefs["input_anomaly_scoring_output_id"],
                )}
                algorithms={adpAlgorithms}
                loading={adpLoadingUpstream}
                error={adpUpstreamError}
                weights={adpWeights}
                onChangeWeight={(algo, value) =>
                  setAdpWeights((prev) => ({ ...prev, [algo]: value }))
                }
              />
            )}

            {submitError && (
              <p className="form__error" role="alert">{submitError}</p>
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
              <button
                type="submit"
                className="form__submit"
                disabled={submitting}
              >
                {submitting ? "Submitting…" : "Submit action"}
              </button>
            </div>
          </form>
        )}
      </section>
    </div>
  );
}

// --- Step 1: type picker ----------------------------------------------

interface ActionTypePickerProps {
  types: ActionTypeMeta[];
  sensorType: string;
  sceneName: string;
  onPick: (t: ActionTypeMeta) => void;
  onCancel: () => void;
}

function ActionTypePicker({
  types,
  sensorType,
  sceneName,
  onPick,
  onCancel,
}: ActionTypePickerProps) {
  if (types.length === 0) {
    return (
      <div className="action-picker__empty">
        <p>
          No action types are available for sensor type
          <strong> {sensorType}</strong> yet. Scene: <em>{sceneName}</em>.
        </p>
        <p className="form__optional">
          New types land via the action_types registry — see Step 12+.
        </p>
        <div className="dialog__actions">
          <button type="button" className="ingest__cancel" onClick={onCancel}>
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="action-picker">
      <p className="form__label">Choose an action type</p>
      <div className="action-picker__grid">
        {types.map((t) => (
          <button
            key={t.type}
            type="button"
            className="action-type-card"
            onClick={() => onPick(t)}
          >
            <div className="action-type-card__head">
              <span className="action-type-card__label">{t.label}</span>
              <span className="action-type-card__slug">{t.type}</span>
            </div>
            <p className="action-type-card__short">{t.short_description}</p>
            <div className="action-type-card__pills">
              {t.accepted_sensor_types.map((s) => (
                <span key={s} className="sensor-pill">{s}</span>
              ))}
              <span className="action-type-card__io">
                {t.inputs.length} in · {t.outputs.length} out
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// --- Step 2a: type summary header --------------------------------------

function ActionTypeSummary({
  meta,
  onBack,
}: {
  meta: ActionTypeMeta;
  onBack: () => void;
}) {
  return (
    <section className="action-summary">
      <button type="button" className="action-summary__back" onClick={onBack}>
        ← Pick a different type
      </button>
      <p className="action-summary__description">{meta.description}</p>
      {meta.when_to_use && (
        <p className="action-summary__when">
          <strong>When to use.</strong> {meta.when_to_use}
        </p>
      )}
      <div className="action-summary__io">
        <div>
          <span className="action-summary__io-title">Inputs</span>
          <ul>
            {meta.inputs.map((i) => (
              <li key={i.key}>
                <strong>{i.label}</strong>{" "}
                <span className="form__optional">({i.ref_kind})</span>
                {i.description && <> — {i.description}</>}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <span className="action-summary__io-title">
            Outputs ({meta.outputs.length})
          </span>
          <ul>
            {meta.outputs.map((o) => (
              <li key={o.key}>
                <strong>{o.label}</strong>{" "}
                <span className="form__optional">({o.artifact_type})</span>
                {o.description && <> — {o.description}</>}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

// --- Step 2b: inputs form ---------------------------------------------

interface InputsFormProps {
  meta: ActionTypeMeta;
  projectId: string;
  sceneId: string;
  sceneName: string;
  sceneSensorType: string;
  inputRefs: Record<string, string>;
  onChangeRef: (key: string, value: string) => void;
}

// Action types are family-keyed: some upstream producers only run on
// HSI scenes (band_filter_apply, scene_segmentation), others only on
// thermal (cloud_mask). Skip rendering optional input pickers that
// target a producer the current scene's sensor family can't run.
const HSI_ONLY_PRODUCING_TYPES = new Set([
  "band_filter_apply",
  "scene_segmentation",
]);
const THERMAL_ONLY_PRODUCING_TYPES = new Set([
  "cloud_mask",
]);

function isThermalSensor(sensor: string): boolean {
  return sensor === "landsat9";
}

function inputApplicable(
  input: ActionInputSpec,
  sensorType: string,
): boolean {
  if (input.ref_kind !== "action_output") return true;
  const isThermal = isThermalSensor(sensorType);
  const targetsHsiOnly = input.producing_action_types.some((t) =>
    HSI_ONLY_PRODUCING_TYPES.has(t),
  );
  const targetsThermalOnly = input.producing_action_types.some((t) =>
    THERMAL_ONLY_PRODUCING_TYPES.has(t),
  );
  if (isThermal && targetsHsiOnly) return false;
  if (!isThermal && targetsThermalOnly) return false;
  return true;
}

function ActionInputsForm({
  meta,
  projectId,
  sceneId,
  sceneName,
  sceneSensorType,
  inputRefs,
  onChangeRef,
}: InputsFormProps) {
  const visible = meta.inputs.filter((i) => inputApplicable(i, sceneSensorType));
  const hidden = meta.inputs.filter(
    (i) => !inputApplicable(i, sceneSensorType),
  );
  return (
    <div className="action-inputs">
      {visible.map((input) => (
        <ActionInputField
          key={input.key}
          input={input}
          projectId={projectId}
          sceneId={sceneId}
          sceneName={sceneName}
          value={inputRefs[input.key] ?? ""}
          onChange={(v) => onChangeRef(input.key, v)}
        />
      ))}
      {hidden.length > 0 && (
        <p className="form__optional small">
          {hidden.map((i) => i.label).join(" · ")} not used on{" "}
          <strong>{sceneSensorType}</strong> scenes (those upstream actions
          are hyperspectral-only).
        </p>
      )}
      <p className="action-inputs__note">
        Configuration uses the system defaults for{" "}
        <strong>{meta.label}</strong>. A custom-parameter editor lands in a
        later step.
      </p>
    </div>
  );
}

interface InputFieldProps {
  input: ActionInputSpec;
  projectId: string;
  sceneId: string;
  sceneName: string;
  value: string;
  onChange: (v: string) => void;
}

function ActionInputField({
  input,
  projectId,
  sceneId,
  sceneName,
  value,
  onChange,
}: InputFieldProps) {
  if (input.ref_kind === "scene") {
    return (
      <div className="form__field">
        <span className="form__label">{input.label}</span>
        <p className="form__readonly">
          <strong>{sceneName}</strong>{" "}
          <span className="form__optional mono">{sceneId}</span>
        </p>
        {input.description && (
          <p className="form__optional">{input.description}</p>
        )}
      </div>
    );
  }
  if (input.ref_kind === "action_output") {
    return (
      <ActionOutputPickerField
        input={input}
        projectId={projectId}
        value={value}
        onChange={onChange}
      />
    );
  }
  // Annotation refs — list raster annotations attached to this scene.
  return (
    <AnnotationPickerField
      input={input}
      sceneId={sceneId}
      value={value}
      onChange={onChange}
    />
  );
}

function AnnotationPickerField({
  input,
  sceneId,
  value,
  onChange,
}: {
  input: ActionInputSpec;
  sceneId: string;
  value: string;
  onChange: (v: string) => void;
}) {
  const [items, setItems] = useState<Annotation[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listAnnotations(sceneId)
      .then((res) => {
        if (cancelled) return;
        setItems(res.items);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError ? (err.detail ?? "fetch failed") : "fetch failed",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [sceneId]);

  return (
    <div className="form__field">
      <span className="form__label">
        {input.label}{" "}
        {!input.required && (
          <span className="form__optional">(optional)</span>
        )}
      </span>
      {error && <p className="form__error" role="alert">{error}</p>}
      {!items && !error && (
        <p className="scene-detail__hint">Loading annotations…</p>
      )}
      {items && items.length === 0 && (
        <p className="form__optional small">
          No annotations attached to this scene. Attach one from the
          Scene Detail page to enable ROC scoring.
        </p>
      )}
      {items && items.length > 0 && (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          required={input.required}
        >
          <option value="">— none (skip ROC) —</option>
          {items.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </select>
      )}
      {input.description && (
        <p className="form__optional">{input.description}</p>
      )}
    </div>
  );
}

function ActionOutputPickerField({
  input,
  projectId,
  value,
  onChange,
}: {
  input: ActionInputSpec;
  projectId: string;
  value: string;
  onChange: (v: string) => void;
}) {
  const [options, setOptions] = useState<Action[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    // We need *complete* actions of the producing types.  The list
    // endpoint already filters by status + type, so issue one call
    // per producing type and concatenate.
    const types =
      input.producing_action_types.length > 0
        ? input.producing_action_types
        : [undefined];
    Promise.all(
      types.map((t) =>
        listProjectActions(projectId, {
          status: "complete",
          type: t,
          limit: 100,
        }),
      ),
    )
      .then((pages) => {
        if (cancelled) return;
        const merged: Action[] = [];
        const seen = new Set<string>();
        for (const p of pages) {
          for (const a of p.items) {
            if (!seen.has(a.id)) {
              seen.add(a.id);
              merged.push(a);
            }
          }
        }
        merged.sort((a, b) =>
          a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0,
        );
        setOptions(merged);
        if (merged.length > 0 && !value) {
          // Auto-select most recent — reasonable default for the v1 flow.
          // The Output id derivation: each complete Action has exactly
          // one output (UNIQUE constraint). We need the output_<uuid>;
          // fetch via getAction for the picked one would add a round
          // trip, so we expose a stable convention: client constructs
          // output_<uuid> equal to the action's referenced output once
          // it learns it. Since we can't compute this client-side, we
          // pre-fetch each action's detail.
          // (Simpler path: backend will validate the selected output by
          // resolving via the action_id, BUT the schema requires
          // output_<uuid>. So we pivot to fetching detail per option.)
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError ? (err.detail ?? "fetch failed") : "fetch failed",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, input.producing_action_types, value]);

  return (
    <ActionOutputDropdown
      input={input}
      options={options}
      error={error}
      value={value}
      onChange={onChange}
    />
  );
}

function ActionOutputDropdown({
  input,
  options,
  error,
  value,
  onChange,
}: {
  input: ActionInputSpec;
  options: Action[] | null;
  error: string | null;
  value: string;
  onChange: (v: string) => void;
}) {
  // Fetch each candidate Action's detail to resolve its output_<uuid>.
  const [outputIds, setOutputIds] = useState<Record<string, string>>({});
  useEffect(() => {
    if (!options || options.length === 0) return;
    let cancelled = false;
    import("../api/actions").then(async ({ getAction }) => {
      const result: Record<string, string> = {};
      for (const a of options) {
        try {
          const detail = await getAction(a.id);
          if (detail.output) result[a.id] = detail.output.id;
        } catch {
          /* skip — surfaces as "missing output" in dropdown */
        }
        if (cancelled) return;
      }
      if (!cancelled) {
        setOutputIds(result);
        if (!value && options.length > 0 && result[options[0].id]) {
          onChange(result[options[0].id]);
        }
      }
    });
    return () => {
      cancelled = true;
    };
    // Only refresh when the option set changes (action_id list).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(options?.map((o) => o.id) ?? [])]);

  return (
    <div className="form__field">
      <span className="form__label">{input.label}</span>
      {error && <p className="form__error" role="alert">{error}</p>}
      {!options && !error && (
        <p className="scene-detail__hint">Loading available outputs…</p>
      )}
      {options && options.length === 0 && (
        <p className="form__error">
          No completed{" "}
          <strong>{input.producing_action_types.join(", ") || "actions"}</strong>{" "}
          in this project yet. Run one first.
        </p>
      )}
      {options && options.length > 0 && (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          required={input.required}
        >
          {options.map((a) => {
            const oid = outputIds[a.id];
            return (
              <option key={a.id} value={oid ?? ""} disabled={!oid}>
                {oid ? "" : "(no output yet) "}
                {a.type} · {new Date(a.created_at).toLocaleString()} ·{" "}
                {a.id}
              </option>
            );
          })}
        </select>
      )}
      {input.description && (
        <p className="form__optional">{input.description}</p>
      )}
    </div>
  );
}

// --- AnomalyScoringExtras -----------------------------------------
//
// Type-specific UI: multi-select model picker filtered by the scene's
// sensor, plus per-picked-model knobs (scoring_method / patch_size /
// stride / batch_size / sam_l1_alpha) driven by each model's
// capability table from /api/models.

export interface ModelOverrideKnobs {
  scoring_method?: string;
  patch_size?: number;
  stride?: number;
  batch_size?: number;
  sam_l1_alpha?: number;
  /** SegFormer-MAE family only — odd integer in [1, 129]. Pixels
   *  within kernel_size//2 of any invalid pixel are excluded from
   *  reconstruction accumulation. */
  erosion_kernel_size?: number;
  /** ALL model families — odd integer in [1, 129]. Erodes the
   *  upstream keep_mask before it's applied to the score. Defends
   *  against the boundary-rim score artifact at cloud / water /
   *  segmentation edges. Default 1 = no erosion. */
  keep_mask_erosion_kernel_size?: number;
}

interface AnomalyExtrasProps {
  sceneSensorType: string;
  modelCodenames: string[];
  onChangeModels: (next: string[]) => void;
  overrides: Record<string, Partial<ModelOverrideKnobs>>;
  onChangeOverrides: (
    next: Record<string, Partial<ModelOverrideKnobs>>,
  ) => void;
}

function AnomalyScoringExtras({
  sceneSensorType,
  modelCodenames,
  onChangeModels,
  overrides,
  onChangeOverrides,
}: AnomalyExtrasProps) {
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listModels()
      .then((items) => {
        if (cancelled) return;
        setModels(items);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError ? (err.detail ?? "fetch failed") : "fetch failed",
        );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Compatible models for this scene's sensor — must declare a sensor
  // that matches, and must expose at least one scoring method (else
  // they're not anomaly-scoring capable).
  const compatible = useMemo<ModelSummary[]>(() => {
    if (!models) return [];
    const sensorMatch = (m: ModelSummary): boolean => {
      // Scene sensors: "prisma" / "enmap" / "aviris_ng" → hyperspectral
      // models (all produce a 165-band common-grid cube post-filter).
      // Scene sensors: "landsat9" / "hotsat1" → thermal models. (HotSAT
      // currently has no anomaly-scoring-compatible model wired in the
      // backend; the picker will correctly show an empty list with the
      // "no compatible models" message until a HotSAT shim is added.)
      const sceneFamily =
        sceneSensorType === "prisma" ||
        sceneSensorType === "enmap" ||
        sceneSensorType === "aviris_ng"
          ? "hyperspectral"
          : sceneSensorType === "landsat9" || sceneSensorType === "hotsat1"
            ? "thermal"
            : null;
      return m.sensor === sceneFamily;
    };
    return models.filter(
      (m) => sensorMatch(m) && (m.scoring_methods ?? []).length > 0,
    );
  }, [models, sceneSensorType]);

  const toggleModel = (codename: string) => {
    const has = modelCodenames.includes(codename);
    onChangeModels(
      has
        ? modelCodenames.filter((c) => c !== codename)
        : [...modelCodenames, codename],
    );
  };

  return (
    <div className="form__field anomaly-extras">
      <span className="form__label">Models</span>
      {error && <p className="form__error" role="alert">{error}</p>}
      {!models && !error && (
        <p className="scene-detail__hint">Loading model catalog…</p>
      )}
      {models && compatible.length === 0 && (
        <p className="form__error">
          No anomaly-scoring-capable models for sensor{" "}
          <strong>{sceneSensorType}</strong>.
        </p>
      )}
      {compatible.length > 0 && (
        <ul className="anomaly-extras__model-list">
          {compatible.map((m) => {
            const picked = modelCodenames.includes(m.codename.name);
            const knobs = overrides[m.codename.name] ?? {};
            return (
              <li
                key={m.codename.name}
                className="anomaly-extras__model-row"
                data-picked={picked ? "true" : "false"}
              >
                <label className="anomaly-extras__model-pick">
                  <input
                    type="checkbox"
                    checked={picked}
                    onChange={() => toggleModel(m.codename.name)}
                  />
                  <span className="anomaly-extras__codename">
                    {m.codename.name}{" "}
                    <span className="anomaly-extras__script">
                      {m.codename.script}
                    </span>
                  </span>
                  <span className="anomaly-extras__model-meta">
                    {m.label} · val {m.val_loss.toFixed(4)}
                  </span>
                </label>
                {picked && (
                  <ModelKnobs
                    model={m}
                    knobs={knobs}
                    onChange={(next) => {
                      const all = { ...overrides };
                      if (Object.keys(next).length === 0) {
                        delete all[m.codename.name];
                      } else {
                        all[m.codename.name] = next;
                      }
                      onChangeOverrides(all);
                    }}
                  />
                )}
              </li>
            );
          })}
        </ul>
      )}
      <p className="form__optional">
        Pick one to scout, more to compare side by side. Each model has its
        own scoring + patch / stride / batch knobs — leave them on default
        unless you have a reason to deviate.
      </p>
    </div>
  );
}

// --- ModelKnobs ----------------------------------------------------
//
// Per-model knob row: scoring + collapsible "Advanced" with patch_size,
// stride, batch_size, and (only for combined) sam_l1_alpha.

function ModelKnobs({
  model,
  knobs,
  onChange,
}: {
  model: ModelSummary;
  knobs: Partial<ModelOverrideKnobs>;
  onChange: (next: Partial<ModelOverrideKnobs>) => void;
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Classical detectors (MNF-RX, Thermal-GRX) are whole-cube ops with
  // no patch / stride / batch / sam_l1_alpha / erosion_kernel_size
  // semantics. The backend rejects those overrides explicitly. Gate
  // the entire foundation-knob block on this so the dialog never
  // emits a forbidden field.
  const isClassical = model.family === "classical";

  const set = (patch: Partial<ModelOverrideKnobs>) => {
    const merged: Partial<ModelOverrideKnobs> = { ...knobs, ...patch };
    // Drop fields explicitly set to undefined or matching default.
    const cleaned: Partial<ModelOverrideKnobs> = {};
    if (merged.scoring_method && merged.scoring_method !== model.default_scoring_method) {
      cleaned.scoring_method = merged.scoring_method;
    }
    if (
      !isClassical &&
      typeof merged.patch_size === "number" &&
      merged.patch_size !== model.default_patch_size
    ) {
      cleaned.patch_size = merged.patch_size;
    }
    if (
      !isClassical &&
      typeof merged.stride === "number" &&
      merged.stride !== model.default_stride
    ) {
      cleaned.stride = merged.stride;
    }
    if (
      !isClassical &&
      typeof merged.batch_size === "number" &&
      merged.batch_size !== model.default_batch_size
    ) {
      cleaned.batch_size = merged.batch_size;
    }
    if (
      !isClassical &&
      (merged.scoring_method === "combined" || knobs.scoring_method === "combined")
    ) {
      if (typeof merged.sam_l1_alpha === "number" && merged.sam_l1_alpha !== 0.5) {
        cleaned.sam_l1_alpha = merged.sam_l1_alpha;
      }
    }
    // erosion_kernel_size — drop when it matches the InferenceConfig
    // default (15) so we don't bloat the submitted config with no-ops.
    // Also drop for classical: backend rejects it (validity-mask
    // erosion is foundation-only). keep_mask_erosion_kernel_size
    // below is the universal one that applies to both families.
    if (
      !isClassical &&
      typeof merged.erosion_kernel_size === "number" &&
      merged.erosion_kernel_size !== 15
    ) {
      cleaned.erosion_kernel_size = merged.erosion_kernel_size;
    }
    // keep_mask_erosion_kernel_size — drop when it's 1 (the no-op
    // default for "don't erode the keep_mask before scoring").
    if (
      typeof merged.keep_mask_erosion_kernel_size === "number" &&
      merged.keep_mask_erosion_kernel_size !== 1
    ) {
      cleaned.keep_mask_erosion_kernel_size =
        merged.keep_mask_erosion_kernel_size;
    }
    onChange(cleaned);
  };

  const currentScoring =
    knobs.scoring_method ?? model.default_scoring_method ?? model.scoring_methods[0];
  const currentPatch = knobs.patch_size ?? model.default_patch_size ?? 128;
  const currentStride = knobs.stride ?? model.default_stride ?? 64;
  const currentBatch = knobs.batch_size ?? model.default_batch_size ?? 8;
  const currentAlpha = knobs.sam_l1_alpha ?? 0.5;
  const currentErosion = knobs.erosion_kernel_size ?? 15;
  const currentKeepErosion = knobs.keep_mask_erosion_kernel_size ?? 1;
  // SegFormer-MAE family uses TokenMasking.erode_mask under the hood;
  // autoencoder family + classical detectors ignore the field. We
  // surface the knob only where it actually does something.
  const usesErosion =
    typeof model.architecture === "string" &&
    model.architecture.includes("segformer_mae");

  const validPatch =
    (model.valid_patch_sizes ?? []).length > 0
      ? model.valid_patch_sizes
      : [64, 128, 256];

  return (
    <div className="anomaly-extras__knobs">
      <label className="anomaly-extras__scoring">
        <span>scoring</span>
        <select
          value={currentScoring ?? ""}
          onChange={(e) => set({ scoring_method: e.target.value })}
        >
          {model.scoring_methods.map((sm) => (
            <option key={sm} value={sm}>
              {sm}
              {sm === model.default_scoring_method ? " (default)" : ""}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        className="anomaly-extras__advanced-toggle"
        onClick={() => setAdvancedOpen((v) => !v)}
        aria-expanded={advancedOpen}
      >
        {advancedOpen ? "Hide" : "Advanced"}{" "}
        <span aria-hidden="true">{advancedOpen ? "▾" : "▸"}</span>
      </button>
      {advancedOpen && (
        <div className="anomaly-extras__advanced">
          {/* Foundation-only knobs. Classical detectors (MNF-RX,
              Thermal-GRX) are whole-cube ops with no patch / stride /
              batch / sam_l1_alpha / inferencer-side erosion. The
              backend rejects them explicitly, so they're hidden here
              for classical models. */}
          {!isClassical && (
            <>
              <label className="anomaly-extras__knob">
                <span>patch_size</span>
                <select
                  value={currentPatch}
                  onChange={(e) => set({ patch_size: Number(e.target.value) })}
                >
                  {validPatch.map((p) => (
                    <option key={p} value={p}>
                      {p}
                      {p === model.default_patch_size ? " (default)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="anomaly-extras__knob">
                <span>stride</span>
                <input
                  type="number"
                  min={8}
                  max={currentPatch}
                  step={8}
                  value={currentStride}
                  onChange={(e) => set({ stride: Number(e.target.value) })}
                />
              </label>
              <label className="anomaly-extras__knob">
                <span>batch_size</span>
                <input
                  type="number"
                  min={1}
                  max={64}
                  step={1}
                  value={currentBatch}
                  onChange={(e) => set({ batch_size: Number(e.target.value) })}
                />
              </label>
            </>
          )}
          {currentScoring === "combined" && (
            <label className="anomaly-extras__knob anomaly-extras__knob--wide">
              <span>
                sam_l1_alpha{" "}
                <span className="anomaly-extras__knob-val">
                  {currentAlpha.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={currentAlpha}
                onChange={(e) => set({ sam_l1_alpha: Number(e.target.value) })}
              />
              <span className="anomaly-extras__knob-help">
                α = 1 → pure L1 · α = 0 → pure SAM · 0.5 = balanced
              </span>
            </label>
          )}
          {usesErosion && (
            <label className="anomaly-extras__knob anomaly-extras__knob--wide">
              <span>
                erosion_kernel_size{" "}
                <span className="anomaly-extras__knob-val">
                  {currentErosion}
                </span>
              </span>
              <input
                type="range"
                min={1}
                max={31}
                step={2}
                value={currentErosion}
                onChange={(e) =>
                  set({ erosion_kernel_size: Number(e.target.value) })
                }
              />
              <span className="anomaly-extras__knob-help">
                Odd integer. Default 15 → 7-px buffer around every invalid
                pixel. Bigger = wider edge exclusion (cleaner score at the
                cost of fewer scored pixels near boundaries).
              </span>
            </label>
          )}
          {/* keep_mask erosion applies to BOTH foundation and classical
              families — strips off the rim of high scores along
              cloud/water/segmentation boundaries. */}
          <label className="anomaly-extras__knob anomaly-extras__knob--wide">
            <span>
              keep_mask_erosion_kernel_size{" "}
              <span className="anomaly-extras__knob-val">
                {currentKeepErosion}
              </span>
            </span>
            <input
              type="range"
              min={1}
              max={31}
              step={2}
              value={currentKeepErosion}
              onChange={(e) =>
                set({ keep_mask_erosion_kernel_size: Number(e.target.value) })
              }
            />
            <span className="anomaly-extras__knob-help">
              Odd integer. Default 1 = no erosion. Erodes the upstream
              keep_mask (scene_segmentation / cloud_mask) by kernel//2
              pixels before scoring. Try 7 if you see bright rings
              tracing the segmentation boundary in the score panel.
            </span>
          </label>
          {!isClassical && (
            <p className="anomaly-extras__knob-help">
              Defaults: patch={model.default_patch_size}, stride=
              {model.default_stride}, batch={model.default_batch_size}.
              Smaller stride = more overlap = slower but smoother. Smaller
              batch = less memory.
            </p>
          )}
          {isClassical && (
            <p className="anomaly-extras__knob-help">
              Closed-form classical detector — no patch / stride / batch
              knobs. Only the keep_mask erosion + scoring method apply.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// --- SceneSegmentationExtras ---------------------------------------
//
// Per-Action knobs for scene_segmentation: the four classification
// thresholds (NDWI water · brightness cloud · brightness shadow · NDVI
// vegetation) plus a multi-select for which classes go into the
// keep_mask. Defaults come from ClassMaskThresholds in the backend; we
// only emit fields the user actually moved.

export interface SegmentationOverrides {
  ndwi_water?: number;
  brightness_cloud?: number;
  brightness_shadow?: number;
  ndvi_vegetation?: number;
  classes_to_mask?: string[];
}

const SEG_CLASS_OPTIONS = ["water", "cloud", "shadow", "vegetation"] as const;

// Defaults mirror backend ClassMaskThresholds. Keep in sync if the
// Pydantic schema moves.
const SEG_DEFAULTS = {
  ndwi_water: 0.3,
  brightness_cloud: 0.4,
  brightness_shadow: 0.02,
  ndvi_vegetation: 0.4,
};

function SceneSegmentationExtras({
  overrides,
  onChange,
}: {
  overrides: SegmentationOverrides;
  onChange: (next: SegmentationOverrides) => void;
}) {
  const [open, setOpen] = useState(false);
  const set = <K extends keyof SegmentationOverrides>(
    key: K,
    value: SegmentationOverrides[K],
  ) => {
    const next: SegmentationOverrides = { ...overrides, [key]: value };
    onChange(next);
  };
  const reset = () => onChange({});
  const v = {
    ndwi_water: overrides.ndwi_water ?? SEG_DEFAULTS.ndwi_water,
    brightness_cloud: overrides.brightness_cloud ?? SEG_DEFAULTS.brightness_cloud,
    brightness_shadow: overrides.brightness_shadow ?? SEG_DEFAULTS.brightness_shadow,
    ndvi_vegetation: overrides.ndvi_vegetation ?? SEG_DEFAULTS.ndvi_vegetation,
  };
  const activeClasses =
    overrides.classes_to_mask ?? [...SEG_CLASS_OPTIONS];
  const toggleClass = (cls: string) => {
    const next = activeClasses.includes(cls)
      ? activeClasses.filter((c) => c !== cls)
      : [...activeClasses, cls];
    onChange({ ...overrides, classes_to_mask: next });
  };
  return (
    <div className="form__field anomaly-extras">
      <div className="anomaly-extras__header">
        <span className="form__label">Segmentation thresholds</span>
        <button
          type="button"
          className="anomaly-extras__advanced-toggle"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
        >
          {open ? "Hide" : "Customize"}{" "}
          <span aria-hidden="true">{open ? "▾" : "▸"}</span>
        </button>
      </div>
      {!open && (
        <p className="anomaly-extras__knob-help small">
          Using defaults: NDWI&gt;0.3 = water · brightness&gt;0.4 = cloud ·
          brightness&lt;0.02 = shadow · NDVI&gt;0.4 = vegetation. All four
          classes go into the keep-mask exclusion.
        </p>
      )}
      {open && (
        <div className="anomaly-extras__advanced">
          <label className="anomaly-extras__knob anomaly-extras__knob--wide">
            <span>
              NDWI &gt; <span className="anomaly-extras__knob-val">{v.ndwi_water.toFixed(2)}</span> → water
            </span>
            <input type="range" min={-1} max={1} step={0.05}
              value={v.ndwi_water}
              onChange={(e) => set("ndwi_water", Number(e.target.value))} />
            <span className="anomaly-extras__knob-help">Higher = stricter water threshold (less masked).</span>
          </label>
          <label className="anomaly-extras__knob anomaly-extras__knob--wide">
            <span>
              VNIR brightness &gt; <span className="anomaly-extras__knob-val">{v.brightness_cloud.toFixed(2)}</span> → cloud
            </span>
            <input type="range" min={0} max={2} step={0.05}
              value={v.brightness_cloud}
              onChange={(e) => set("brightness_cloud", Number(e.target.value))} />
            <span className="anomaly-extras__knob-help">Higher = only the brightest pixels flagged as clouds.</span>
          </label>
          <label className="anomaly-extras__knob anomaly-extras__knob--wide">
            <span>
              VNIR brightness &lt; <span className="anomaly-extras__knob-val">{v.brightness_shadow.toFixed(3)}</span> → shadow
            </span>
            <input type="range" min={0} max={0.1} step={0.005}
              value={v.brightness_shadow}
              onChange={(e) => set("brightness_shadow", Number(e.target.value))} />
            <span className="anomaly-extras__knob-help">Higher = more dim pixels flagged as shadow.</span>
          </label>
          <label className="anomaly-extras__knob anomaly-extras__knob--wide">
            <span>
              NDVI &gt; <span className="anomaly-extras__knob-val">{v.ndvi_vegetation.toFixed(2)}</span> → vegetation
            </span>
            <input type="range" min={-1} max={1} step={0.05}
              value={v.ndvi_vegetation}
              onChange={(e) => set("ndvi_vegetation", Number(e.target.value))} />
            <span className="anomaly-extras__knob-help">Higher = only densely vegetated pixels flagged.</span>
          </label>
          <div className="anomaly-extras__knob anomaly-extras__knob--wide">
            <span>Classes folded into the keep_mask exclusion</span>
            <div className="seg-extras__class-chips">
              {SEG_CLASS_OPTIONS.map((c) => (
                <label key={c} className="seg-extras__class-chip" data-on={activeClasses.includes(c) ? "true" : "false"}>
                  <input type="checkbox" checked={activeClasses.includes(c)} onChange={() => toggleClass(c)} />
                  <span>{c}</span>
                </label>
              ))}
            </div>
          </div>
          <button type="button" className="anomaly-extras__advanced-toggle" onClick={reset}>
            Reset to defaults
          </button>
        </div>
      )}
    </div>
  );
}


// =====================================================================
// AnomalyDetectionPrepExtras — per-algorithm weight inputs
// =====================================================================
//
// The user has picked an upstream anomaly_scoring output via the
// generic input picker above. As soon as that selection completes, the
// dialog fetches the upstream's summary.json and reveals one weight
// input per algorithm (model codename) that ran in that upstream
// action. Default weight is 1.0 per algorithm — equal weighting on the
// composite. Weight 0 zeros that algorithm out without re-running.
//
// All weights >= 0; at least one must be > 0 (the submit validator
// enforces both).

function AnomalyDetectionPrepExtras({
  upstreamSelected,
  algorithms,
  loading,
  error,
  weights,
  onChangeWeight,
}: {
  upstreamSelected: boolean;
  algorithms: string[] | null;
  loading: boolean;
  error: string | null;
  weights: Record<string, number>;
  onChangeWeight: (algo: string, value: number) => void;
}) {
  return (
    <div className="form__field anomaly-extras">
      <span className="form__label">Algorithm weights</span>
      {!upstreamSelected && (
        <p className="scene-detail__hint">
          Pick an upstream <code>anomaly_scoring</code> output above —
          this form will reveal one weight input per algorithm that ran.
        </p>
      )}
      {upstreamSelected && loading && (
        <p className="scene-detail__hint">Loading upstream algorithms…</p>
      )}
      {upstreamSelected && error && (
        <p className="form__error" role="alert">{error}</p>
      )}
      {upstreamSelected && !loading && algorithms && algorithms.length > 0 && (
        <>
          <p className="form__hint small">
            Each algorithm's score map is rescaled to [0, 1] and then
            weight-averaged into a single composite. Weight 0 drops that
            algorithm from the composite. Defaults to equal weights.
          </p>
          <ul className="anomaly-extras__model-list">
            {algorithms.map((algo) => {
              const value = weights[algo] ?? 1.0;
              return (
                <li key={algo} className="anomaly-extras__model-row" data-picked="true">
                  <div className="anomaly-extras__model-head">
                    <span className="anomaly-extras__model-name">{algo}</span>
                    <label className="anomaly-extras__knob">
                      <span>weight</span>
                      <input
                        type="number"
                        min={0}
                        step={0.1}
                        value={value}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          onChangeWeight(algo, Number.isFinite(v) ? v : 0);
                        }}
                      />
                    </label>
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}
