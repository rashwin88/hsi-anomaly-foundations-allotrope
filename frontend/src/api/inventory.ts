// Raw-dataset inventory — read-only client.
//
// Not the main api: these calls go to /inventory-api/*, which nginx proxies
// to the separate inventory service after checking the JWT cookie against
// /api/auth/me and adding the service's api key server-side (nginx.conf).
// GET only — the UI is a viewer; datasets are ingested with the CLI.
//
// Backend routes: backend/allotrope_inventory/api/{datasets,stats,tags,sensors}.py

import type {
  InventoryDatasetDetail,
  InventoryDatasetsPage,
  InventoryEvent,
  InventorySensor,
  InventoryStats,
  InventoryStatus,
  InventoryTag,
} from "../types";
import { ApiError } from "./client";

const INVENTORY_BASE = "/inventory-api";

/** True when the inventory can't serve this user: service down or not
 *  deployed (nginx 502/503/504), or nginx's api key missing/wrong (401 from
 *  the inventory — a logged-in user only reaches it past auth_request). */
export function isInventoryUnavailable(err: unknown): boolean {
  return err instanceof ApiError && [401, 502, 503, 504].includes(err.status);
}

async function getJson<T>(path: string, params?: URLSearchParams): Promise<T> {
  const query = params?.toString();
  const res = await fetch(`${INVENTORY_BASE}${path}${query ? `?${query}` : ""}`, {
    credentials: "include",
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: unknown };
      // Most details are code strings; a few are {code, …} objects.
      detail =
        typeof body.detail === "string"
          ? body.detail
          : (body.detail as { code?: string } | undefined)?.code;
    } catch {
      /* body not JSON — nginx error pages are HTML */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export interface ListInventoryOptions {
  limit?: number;
  offset?: number;
  sensor_type?: string;
  processing_level?: string;
  status?: InventoryStatus;
  /** Every tag must match. */
  tags?: string[];
  /** Substring of the provider's scene id. */
  q?: string;
  /** "min_lon,min_lat,max_lon,max_lat" */
  bbox?: string;
  /** ISO date or datetime. */
  acquired_from?: string;
  acquired_to?: string;
}

export async function listInventoryDatasets(
  opts: ListInventoryOptions = {},
): Promise<InventoryDatasetsPage> {
  const params = new URLSearchParams();
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  if (opts.offset !== undefined) params.set("offset", String(opts.offset));
  if (opts.sensor_type) params.set("sensor_type", opts.sensor_type);
  if (opts.processing_level) params.set("processing_level", opts.processing_level);
  if (opts.status) params.set("status", opts.status);
  for (const t of opts.tags ?? []) params.append("tag", t);
  if (opts.q && opts.q.trim() !== "") params.set("q", opts.q.trim());
  if (opts.bbox && opts.bbox.trim() !== "") params.set("bbox", opts.bbox.trim());
  if (opts.acquired_from) params.set("acquired_from", opts.acquired_from);
  if (opts.acquired_to) params.set("acquired_to", opts.acquired_to);
  return getJson<InventoryDatasetsPage>("/datasets", params);
}

export async function getInventoryDataset(id: string): Promise<InventoryDatasetDetail> {
  return getJson<InventoryDatasetDetail>(`/datasets/${encodeURIComponent(id)}`);
}

export async function listInventoryEvents(id: string): Promise<InventoryEvent[]> {
  return getJson<InventoryEvent[]>(`/datasets/${encodeURIComponent(id)}/events`);
}

export async function getInventoryStats(): Promise<InventoryStats> {
  return getJson<InventoryStats>("/stats");
}

export async function listInventorySensors(): Promise<InventorySensor[]> {
  return getJson<InventorySensor[]>("/sensors");
}

export async function listInventoryTags(): Promise<InventoryTag[]> {
  return getJson<InventoryTag[]>("/tags");
}

/** "1.4 GB" — binary units, one decimal from KB up. */
export function formatBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return i === 0 ? `${v} B` : `${v.toFixed(1)} ${units[i]}`;
}
