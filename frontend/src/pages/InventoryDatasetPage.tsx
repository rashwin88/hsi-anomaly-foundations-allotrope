// Inventory — one dataset (a single scene delivery).
//
// Route: /inventory/<ds_<uuid>>. Read-only: metadata, footprint outline,
// files with their sha256, where it came from, tags, and the audit trail.
// No download button by design — the S3 location is shown as text.
//
// Backend: GET /inventory-api/datasets/{id} and /datasets/{id}/events.

import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  formatBytes,
  getInventoryDataset,
  isInventoryUnavailable,
  listInventoryEvents,
} from "../api/inventory";
import type { GeoJsonGeometry, InventoryDatasetDetail, InventoryEvent } from "../types";
import { formatInventoryDate, statusPillClass } from "./InventoryPage";

type Ring = [number, number][];

function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Outer rings of a Polygon or MultiPolygon; anything else draws nothing. */
function outerRings(g: GeoJsonGeometry): Ring[] {
  if (g.type === "Polygon") return [(g.coordinates as Ring[])[0]];
  if (g.type === "MultiPolygon") return (g.coordinates as Ring[][]).map((p) => p[0]);
  return [];
}

/** Footprint outline scaled into a fixed box, with its lon/lat extent as labels.
 *  No basemap — the shape and the numbers are what an inventory view needs. */
function FootprintOutline({ geometry }: { geometry: GeoJsonGeometry }) {
  const rings = outerRings(geometry);
  const pts = rings.flat();
  if (pts.length === 0) {
    return <p className="muted small">{geometry.type} footprint (not drawn).</p>;
  }
  const lons = pts.map((p) => p[0]);
  const lats = pts.map((p) => p[1]);
  const [minLon, maxLon, minLat, maxLat] = [
    Math.min(...lons), Math.max(...lons), Math.min(...lats), Math.max(...lats),
  ];
  const W = 320;
  const H = 220;
  const pad = 16;
  // Equal scale on both axes (degrees, near-enough at scene size) so the
  // shape isn't stretched.
  const span = Math.max(maxLon - minLon, maxLat - minLat) || 1e-6;
  const k = Math.min((W - 2 * pad) / span, (H - 2 * pad) / span);
  const ox = (W - (maxLon - minLon) * k) / 2;
  const oy = (H - (maxLat - minLat) * k) / 2;
  const toXY = ([lon, lat]: [number, number]) =>
    `${(ox + (lon - minLon) * k).toFixed(1)},${(oy + (maxLat - lat) * k).toFixed(1)}`;
  const f = (n: number) => n.toFixed(4);

  return (
    <figure className="inventory-detail__footprint">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Scene footprint outline">
        <rect x="0.5" y="0.5" width={W - 1} height={H - 1} className="footprint__frame" />
        {rings.map((ring, i) => (
          <polygon key={i} points={ring.map(toXY).join(" ")} className="footprint__shape" />
        ))}
      </svg>
      <figcaption className="muted small mono">
        lon {f(minLon)} … {f(maxLon)} · lat {f(minLat)} … {f(maxLat)}
      </figcaption>
    </figure>
  );
}

function describeEvent(e: InventoryEvent): string {
  const p = e.payload as Record<string, unknown>;
  switch (e.kind) {
    case "dataset.registered":
      return "Registered";
    case "file.declared":
      return `Declared ${String(p.path)} (${formatBytes(Number(p.size_bytes ?? 0))})`;
    case "file.uploaded":
      return `Uploaded and verified ${String(p.path)}`;
    case "file.copied":
      return `Copied ${String(p.path)} server-side from ${String(p.source)}`;
    case "file.rejected":
      return `Rejected ${String(p.path)} — checksum or size mismatch`;
    case "dataset.finalized":
      return `Finalised — ${String(p.files)} files, ${formatBytes(Number(p.total_bytes ?? 0))}`;
    case "dataset.downloaded":
      return "Download links issued";
    case "tag.added":
      return `Tagged ${String(p.tag)}`;
    case "tag.removed":
      return `Untagged ${String(p.tag)}`;
    default:
      return e.kind;
  }
}

export function InventoryDatasetPage() {
  const { datasetId = "" } = useParams<{ datasetId: string }>();
  const [d, setD] = useState<InventoryDatasetDetail | null>(null);
  const [events, setEvents] = useState<InventoryEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    Promise.all([getInventoryDataset(datasetId), listInventoryEvents(datasetId)])
      .then(([detail, evs]) => {
        if (cancelled) return;
        setD(detail);
        setEvents(evs);
      })
      .catch((err) => {
        if (cancelled) return;
        if (isInventoryUnavailable(err)) setError("The inventory service is not reachable.");
        else if (err instanceof ApiError && err.detail === "dataset_not_found")
          setError("No dataset with this id.");
        else if (err instanceof ApiError) setError(err.detail ?? `Error: HTTP ${err.status}`);
        else setError("Could not reach the server.");
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const back = (
    <Link to="/inventory" className="scene-detail__back">
      ← Inventory
    </Link>
  );

  if (error) {
    return (
      <div className="page inventory-detail">
        {back}
        <p className="page__error" role="alert">
          {error}
        </p>
      </div>
    );
  }
  if (!d) {
    return (
      <div className="page inventory-detail">
        {back}
        <p className="muted">Loading…</p>
      </div>
    );
  }

  const metaEntries = Object.entries(d.metadata);

  return (
    <div className="page inventory-detail">
      {back}
      <header className="page__header">
        <div>
          <h2 className="page__title mono inventory-detail__title">{d.sensor_scene_id}</h2>
          <p className="page__subtitle">
            <span className="sensor-pill">{d.sensor_type}</span>{" "}
            <span className="muted">{d.processing_level}</span>{" "}
            <span className={statusPillClass(d.status)}>{d.status}</span>
          </p>
        </div>
      </header>

      <div className="inventory-detail__grid">
        <section className="panel inventory-detail__panel">
          <h3 className="panel__heading">Dataset</h3>
          <dl className="panel__list">
            <div className="panel__row">
              <dt>Id</dt>
              <dd className="mono">{d.id}</dd>
            </div>
            <div className="panel__row">
              <dt>Acquired</dt>
              <dd>{formatDateTime(d.acquisition_at)}</dd>
            </div>
            <div className="panel__row">
              <dt>Size</dt>
              <dd>
                {formatBytes(d.total_bytes)} in {d.files.length} file{d.files.length === 1 ? "" : "s"}
              </dd>
            </div>
            <div className="panel__row">
              <dt>S3 location</dt>
              <dd className="mono">{d.s3_uri}</dd>
            </div>
            <div className="panel__row">
              <dt>Registered</dt>
              <dd>
                {formatDateTime(d.created_at)} by {d.created_by}
              </dd>
            </div>
            <div className="panel__row">
              <dt>Finalised</dt>
              <dd>{formatDateTime(d.finalized_at)}</dd>
            </div>
            <div className="panel__row">
              <dt>Tags</dt>
              <dd>
                {d.tags.length === 0 ? (
                  <span className="muted">—</span>
                ) : (
                  d.tags.map((t) => (
                    <Link
                      key={t}
                      to={`/inventory?tag=${encodeURIComponent(t)}`}
                      className="inventory__tag"
                    >
                      {t}
                    </Link>
                  ))
                )}
              </dd>
            </div>
          </dl>
        </section>

        <section className="panel inventory-detail__panel">
          <h3 className="panel__heading">Footprint</h3>
          {d.footprint ? (
            <FootprintOutline geometry={d.footprint} />
          ) : (
            <p className="muted small">No footprint recorded.</p>
          )}
        </section>

        {metaEntries.length > 0 && (
          <section className="panel inventory-detail__panel">
            <h3 className="panel__heading">Sensor metadata</h3>
            <dl className="panel__list">
              {metaEntries.map(([k, v]) => (
                <div key={k} className="panel__row">
                  <dt>{k}</dt>
                  <dd className="mono">{typeof v === "string" ? v : JSON.stringify(v)}</dd>
                </div>
              ))}
            </dl>
          </section>
        )}

        <section className="panel inventory-detail__panel">
          <h3 className="panel__heading">Sources</h3>
          {d.sources.length === 0 ? (
            <p className="muted small">No origin recorded.</p>
          ) : (
            <ul className="inventory-detail__list">
              {d.sources.map((s, i) => (
                <li key={i}>
                  <span className="mono">{s.source_uri}</span>
                  <span className="muted small">
                    {" "}
                    {s.relative_path ? `→ ${s.relative_path} · ` : "whole delivery · "}
                    {s.fetched_by}, {formatInventoryDate(s.fetched_at)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <h3 className="inventory-detail__section-title">Files</h3>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Path</th>
              <th>Size</th>
              <th>SHA-256</th>
              <th>State</th>
            </tr>
          </thead>
          <tbody>
            {d.files.map((f) => (
              <tr key={f.relative_path}>
                <td className="data-table__primary mono">{f.relative_path}</td>
                <td>{formatBytes(f.size_bytes)}</td>
                <td className="mono inventory-detail__hash" title={f.sha256}>
                  {f.sha256.slice(0, 16)}…
                </td>
                <td>
                  <span className={statusPillClass(f.uploaded ? "available" : "pending")}>
                    {f.uploaded ? "verified" : "uploading"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="inventory-detail__section-title">Activity</h3>
      <ol className="inventory-detail__timeline">
        {events.map((e, i) => (
          <li key={i}>
            <span className="muted small">{formatDateTime(e.created_at)}</span>
            <span>{describeEvent(e)}</span>
            <span className="muted small">{e.actor}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
