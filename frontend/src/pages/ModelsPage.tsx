// Models landing — read-only catalog of foundation models.
//
// Renders 7 cards in a responsive grid, sorted by val loss ascending
// (best first). Each card shows the codename + Devanagari, label,
// sensor pill, key metrics, normalization mode, and links into the
// per-model detail page.
//
// Sequence diagram: final design/diagrams/models-list.drawio

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listActionTypes } from "../api/actions";
import { ApiError } from "../api/client";
import { listModels } from "../api/models";
import { ActionTemplatesPanel } from "../components/ActionTemplatesPanel";
import type { ActionTypeMeta, ModelSummary } from "../types";

const SENSOR_LABEL: Record<string, string> = {
  thermal: "Thermal · Landsat 9",
  hyperspectral: "Hyperspectral · PRISMA / EnMAP",
};

function formatParams(p: number): string {
  if (p >= 1_000_000) return `${(p / 1_000_000).toFixed(2)}M`;
  if (p >= 1_000) return `${(p / 1_000).toFixed(0)}K`;
  return String(p);
}

function NormBadge({ mode }: { mode: string }) {
  const baked = mode === "baked_in";
  return (
    <span
      className="models-card__norm"
      data-baked={baked ? "true" : "false"}
    >
      {baked ? "normalize: baked-in" : "normalize: none"}
    </span>
  );
}

export function ModelsPage() {
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ActionTypeMeta[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    listModels()
      .then((rows) => {
        if (!cancelled) setModels(rows);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setError(err.detail ?? `Error: HTTP ${err.status}`);
        } else {
          setError("Could not reach the server.");
        }
      });
    listActionTypes()
      .then((types) => {
        if (!cancelled) setCatalog(types);
      })
      .catch(() => {
        // Non-fatal — the templates panel falls back to raw slugs.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Foundation models</h1>
          <p className="page__subtitle">
            Reconstruction-based detectors shipped with Allotrope. Each
            architecture carries an Indic codename — Sanskrit-rooted, tied
            to what the model actually does.
          </p>
        </div>
      </div>

      {error && (
        <div className="page__error" role="alert">
          {error}
        </div>
      )}

      {!models && !error && (
        <div className="page__empty">Loading catalog…</div>
      )}

      {models && (
        <div className="models-grid">
          {models.map((m, idx) => (
            <Link
              key={m.architecture}
              to={`/models/${encodeURIComponent(m.architecture)}`}
              className="models-card"
              data-rank={idx + 1}
              data-family={m.family}
            >
              {m.family === "classical" && (
                <span
                  className="models-card__ribbon"
                  aria-label="Classical statistical detector — no learned weights"
                >
                  Classical
                </span>
              )}
              <div className="models-card__rank">#{idx + 1}</div>
              <div className="models-card__codename">
                <span className="models-card__codename-name">
                  {m.codename.name}
                </span>
                {m.codename.script && (
                  <span className="models-card__codename-script">
                    {m.codename.script}
                  </span>
                )}
              </div>
              <div className="models-card__label">{m.label}</div>
              <div className="models-card__meaning">
                <em>{m.codename.meaning}.</em> {m.codename.why}
              </div>
              <div className="models-card__pills">
                <span className="pill pill--sensor">
                  {SENSOR_LABEL[m.sensor] ?? m.sensor}
                </span>
                <NormBadge mode={m.normalization_mode} />
              </div>
              <dl className="models-card__metrics">
                {m.family === "classical" ? (
                  <>
                    <div>
                      <dt>method</dt>
                      <dd>closed-form</dd>
                    </div>
                    <div>
                      <dt>weights</dt>
                      <dd>none</dd>
                    </div>
                    <div>
                      <dt>knobs</dt>
                      <dd>—</dd>
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <dt>val loss</dt>
                      <dd>{m.val_loss.toFixed(4)}</dd>
                    </div>
                    <div>
                      <dt>params</dt>
                      <dd>{formatParams(m.params)}</dd>
                    </div>
                    <div>
                      <dt>version</dt>
                      <dd>
                        {m.version} <span className="muted">· ep {m.epoch}</span>
                      </dd>
                    </div>
                  </>
                )}
              </dl>
              <div className="models-card__cta">View architecture →</div>
            </Link>
          ))}
        </div>
      )}

      <ActionTemplatesPanel catalog={catalog} />
    </div>
  );
}
