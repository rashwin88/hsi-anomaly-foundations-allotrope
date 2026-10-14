// Inventory — the raw-dataset catalogue as a whole.
//
// Read-only view over the separate inventory service (backend/allotrope_inventory)
// via /inventory-api/*. Summary tiles across the top (GET /stats), then search +
// filters, then a paginated table (GET /datasets). Datasets are ingested with
// the CLI, not here.
//
// Filters live in the URL query string, so a filtered view is linkable and
// survives the round trip to a dataset's detail page and back.

import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  formatBytes,
  getInventoryStats,
  isInventoryUnavailable,
  listInventoryDatasets,
  listInventorySensors,
  listInventoryTags,
} from "../api/inventory";
import type {
  InventoryDatasetsPage,
  InventoryKind,
  InventorySensor,
  InventoryStats,
  InventoryStatus,
  InventoryTag,
} from "../types";

const PAGE_SIZE = 50;
const STATUSES: InventoryStatus[] = ["available", "pending", "failed", "retired"];
const KINDS: InventoryKind[] = ["scene", "collection", "archive"];
// Filter keys mirrored in the URL. `tag` repeats.
const FILTER_KEYS = ["q", "sensor", "level", "status", "kind", "tag", "from", "to", "bbox"];

export function statusPillClass(status: string): string {
  const tone =
    status === "available"
      ? "complete"
      : status === "pending"
        ? "running"
        : status === "retired"
          ? "retired"
          : "failed";
  return `status-pill status-pill--${tone}`;
}

export function formatInventoryDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.detail === "bad_bbox") return "Bounding box must be min_lon,min_lat,max_lon,max_lat.";
    return err.detail ?? `Error: HTTP ${err.status}`;
  }
  return "Could not reach the server.";
}

export function InventoryPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const q = params.get("q") ?? "";
  const sensor = params.get("sensor") ?? "";
  const level = params.get("level") ?? "";
  const status = (params.get("status") ?? "") as InventoryStatus | "";
  const kind = (params.get("kind") ?? "") as InventoryKind | "";
  const tags = params.getAll("tag");
  const from = params.get("from") ?? "";
  const to = params.get("to") ?? "";
  const bbox = params.get("bbox") ?? "";
  const offset = Number(params.get("offset") ?? "0") || 0;

  const [stats, setStats] = useState<InventoryStats | null>(null);
  const [sensors, setSensors] = useState<InventorySensor[]>([]);
  const [allTags, setAllTags] = useState<InventoryTag[]>([]);
  const [page, setPage] = useState<InventoryDatasetsPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  // Text inputs are edited locally and pushed to the URL after a pause, so
  // each keystroke doesn't fire a query.
  const [qDraft, setQDraft] = useState(q);
  const [bboxDraft, setBboxDraft] = useState(bbox);
  const [levelDraft, setLevelDraft] = useState(level);

  const setFilter = (key: string, value: string | string[]) => {
    const next = new URLSearchParams(params);
    next.delete(key);
    for (const v of Array.isArray(value) ? value : [value]) {
      if (v !== "") next.append(key, v);
    }
    next.delete("offset");
    setParams(next, { replace: true });
  };

  useEffect(() => {
    const t = window.setTimeout(() => {
      if (qDraft !== q) setFilter("q", qDraft);
    }, 300);
    return () => window.clearTimeout(t);
  }, [qDraft]);

  // Reference data: stats tiles, sensor list, tag chips.
  useEffect(() => {
    let cancelled = false;
    Promise.all([getInventoryStats(), listInventorySensors(), listInventoryTags()])
      .then(([s, sens, tg]) => {
        if (cancelled) return;
        setStats(s);
        setSensors(sens);
        setAllTags(tg);
      })
      .catch((err) => {
        if (cancelled) return;
        if (isInventoryUnavailable(err)) setUnavailable(true);
        else setError(describeError(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const tagKey = tags.join("\u0000");
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listInventoryDatasets({
      limit: PAGE_SIZE,
      offset,
      q,
      sensor_type: sensor || undefined,
      processing_level: level || undefined,
      status: status || undefined,
      kind: kind || undefined,
      tags,
      bbox,
      acquired_from: from || undefined,
      acquired_to: to || undefined,
    })
      .then((result) => {
        if (!cancelled) setPage(result);
      })
      .catch((err) => {
        if (cancelled) return;
        if (isInventoryUnavailable(err)) setUnavailable(true);
        else setError(describeError(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [q, sensor, level, status, kind, tagKey, from, to, bbox, offset]);

  const bySensor = useMemo(() => {
    const out = new Map<string, { datasets: number; bytes: number }>();
    for (const c of stats?.cells ?? []) {
      const cur = out.get(c.sensor_type) ?? { datasets: 0, bytes: 0 };
      out.set(c.sensor_type, { datasets: cur.datasets + c.datasets, bytes: cur.bytes + c.bytes });
    }
    return out;
  }, [stats]);

  const countFor = (st: InventoryStatus) =>
    (stats?.cells ?? []).filter((c) => c.status === st).reduce((n, c) => n + c.datasets, 0);

  const sensorName = (slug: string) => sensors.find((s) => s.slug === slug)?.name ?? slug;
  const hasFilters = FILTER_KEYS.some((k) => params.has(k));

  if (unavailable) {
    return (
      <div className="page inventory">
        <header className="page__header">
          <div>
            <h2 className="page__title">Inventory</h2>
            <p className="page__subtitle">Raw scene deliveries held in S3.</p>
          </div>
        </header>
        <div className="inventory__unavailable" role="status">
          <p>
            <strong>The inventory service is not reachable or not configured.</strong>
          </p>
          <p className="muted small">
            It runs as a separate container. Start it with{" "}
            <code className="mono">
              docker compose -f docker/docker-compose.yml --profile inventory up -d inventory
            </code>
            , and make sure <code className="mono">INVENTORY_UI_KEY</code> is set for the frontend.
          </p>
        </div>
      </div>
    );
  }

  const total = page?.total ?? 0;
  const items = page?.items ?? [];
  const rangeStart = total === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + items.length, total);

  const setOffset = (n: number) => {
    const next = new URLSearchParams(params);
    if (n > 0) next.set("offset", String(n));
    else next.delete("offset");
    setParams(next, { replace: true });
  };

  const clearFilters = () => {
    setQDraft("");
    setBboxDraft("");
    setLevelDraft("");
    setParams(new URLSearchParams(), { replace: true });
  };

  const toggleTag = (name: string) =>
    setFilter("tag", tags.includes(name) ? tags.filter((t) => t !== name) : [...tags, name]);

  return (
    <div className="page inventory">
      <header className="page__header">
        <div>
          <h2 className="page__title">Inventory</h2>
          <p className="page__subtitle">
            Raw scene deliveries held in S3, exactly as each provider shipped them. Read-only —
            datasets are ingested with the inventory CLI.
          </p>
        </div>
      </header>

      {stats && (
        <section className="inventory__stats" aria-label="Inventory summary">
          <div className="inventory__stat">
            <span className="inventory__stat-value">{stats.datasets.toLocaleString()}</span>
            <span className="inventory__stat-label">datasets</span>
          </div>
          <div className="inventory__stat">
            <span className="inventory__stat-value">{formatBytes(stats.bytes)}</span>
            <span className="inventory__stat-label">stored</span>
          </div>
          {STATUSES.map((st) => (
            <button
              key={st}
              type="button"
              className={"inventory__stat inventory__stat--button" + (status === st ? " is-active" : "")}
              onClick={() => setFilter("status", status === st ? "" : st)}
              title={`Show ${st} datasets`}
            >
              <span className="inventory__stat-value">{countFor(st).toLocaleString()}</span>
              <span className="inventory__stat-label">{st}</span>
            </button>
          ))}
          {[...bySensor.entries()].map(([slug, v]) => (
            <button
              key={slug}
              type="button"
              className={"inventory__stat inventory__stat--button" + (sensor === slug ? " is-active" : "")}
              onClick={() => setFilter("sensor", sensor === slug ? "" : slug)}
              title={`Show ${sensorName(slug)} datasets`}
            >
              <span className="inventory__stat-value">{v.datasets.toLocaleString()}</span>
              <span className="inventory__stat-label">
                {sensorName(slug)} · {formatBytes(v.bytes)}
              </span>
            </button>
          ))}
        </section>
      )}

      <section className="inventory__filters" aria-label="Search and filters">
        <label className="form__field inventory__filter--wide">
          <span className="form__label">Search scene id</span>
          <input
            type="text"
            value={qDraft}
            onChange={(e) => setQDraft(e.target.value)}
            placeholder="e.g. PRS_L2D_STD_2025"
          />
        </label>
        <label className="form__field">
          <span className="form__label">Sensor</span>
          <select value={sensor} onChange={(e) => setFilter("sensor", e.target.value)}>
            <option value="">All sensors</option>
            {sensors.map((s) => (
              <option key={s.slug} value={s.slug}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label className="form__field">
          <span className="form__label">Level</span>
          <input
            type="text"
            value={levelDraft}
            onChange={(e) => setLevelDraft(e.target.value)}
            onBlur={() => setFilter("level", levelDraft.trim())}
            onKeyDown={(e) => e.key === "Enter" && setFilter("level", levelDraft.trim())}
            placeholder="L2D"
          />
        </label>
        <label className="form__field">
          <span className="form__label">Status</span>
          <select value={status} onChange={(e) => setFilter("status", e.target.value)}>
            <option value="">Any status</option>
            {STATUSES.map((st) => (
              <option key={st} value={st}>
                {st}
              </option>
            ))}
          </select>
        </label>
        <label className="form__field">
          <span className="form__label">Kind</span>
          <select value={kind} onChange={(e) => setFilter("kind", e.target.value)}>
            <option value="">Any kind</option>
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <label className="form__field">
          <span className="form__label">Acquired from</span>
          <input type="date" value={from} onChange={(e) => setFilter("from", e.target.value)} />
        </label>
        <label className="form__field">
          <span className="form__label">Acquired before</span>
          <input type="date" value={to} onChange={(e) => setFilter("to", e.target.value)} />
        </label>
        <label className="form__field inventory__filter--wide">
          <span className="form__label">Footprint intersects (lon/lat box)</span>
          <input
            type="text"
            value={bboxDraft}
            onChange={(e) => setBboxDraft(e.target.value)}
            onBlur={() => setFilter("bbox", bboxDraft.trim())}
            onKeyDown={(e) => e.key === "Enter" && setFilter("bbox", bboxDraft.trim())}
            placeholder="min_lon,min_lat,max_lon,max_lat — e.g. 86,23,87,24"
          />
        </label>
        {hasFilters && (
          <button type="button" className="pagination__btn inventory__clear" onClick={clearFilters}>
            Clear filters
          </button>
        )}
      </section>

      {allTags.length > 0 && (
        <div className="scenes__filters inventory__tags" role="group" aria-label="Filter by tag">
          {allTags.map((t) => (
            <button
              key={t.name}
              type="button"
              aria-pressed={tags.includes(t.name)}
              className={"scenes__filter" + (tags.includes(t.name) ? " is-active" : "")}
              onClick={() => toggleTag(t.name)}
              title={t.description ?? undefined}
            >
              {t.name} <span className="muted">{t.dataset_count}</span>
            </button>
          ))}
        </div>
      )}

      {error && (
        <p className="form__error" role="alert">
          {error}
        </p>
      )}

      {!error && items.length === 0 && !loading ? (
        <div className="page__empty">
          {hasFilters
            ? "No datasets match these filters."
            : "The inventory is empty — ingest a delivery with the inventory CLI and it will show up here."}
        </div>
      ) : (
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Scene id</th>
                <th>Kind</th>
                <th>Sensor</th>
                <th>Level</th>
                <th>Acquired</th>
                <th>Status</th>
                <th>Tags</th>
                <th>Registered</th>
              </tr>
            </thead>
            <tbody>
              {items.map((d) => {
                const open = () => navigate(`/inventory/${encodeURIComponent(d.id)}`);
                return (
                  <tr
                    key={d.id}
                    className="data-table__row--clickable"
                    onClick={open}
                    role="link"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        open();
                      }
                    }}
                  >
                    <td className="data-table__primary mono">{d.sensor_scene_id}</td>
                    <td>
                      <span className={`inventory__kind inventory__kind--${d.kind}`}>{d.kind}</span>
                    </td>
                    <td>
                      <span className="sensor-pill">{sensorName(d.sensor_type)}</span>
                    </td>
                    <td>{d.processing_level}</td>
                    <td>{formatInventoryDate(d.acquisition_at)}</td>
                    <td>
                      <span className={statusPillClass(d.status)}>{d.status}</span>
                    </td>
                    <td className="inventory__tag-cell">
                      {d.tags.length === 0 ? (
                        <span className="muted">—</span>
                      ) : (
                        d.tags.map((t) => (
                          <span key={t} className="inventory__tag">
                            {t}
                          </span>
                        ))
                      )}
                    </td>
                    <td>{formatInventoryDate(d.created_at)}</td>
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
            {rangeStart.toLocaleString()}–{rangeEnd.toLocaleString()} of {total.toLocaleString()}
          </span>
          <div className="pagination__buttons">
            <button
              type="button"
              className="pagination__btn"
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              disabled={offset === 0 || loading}
            >
              Previous
            </button>
            <button
              type="button"
              className="pagination__btn"
              onClick={() => setOffset(offset + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= total || loading}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
