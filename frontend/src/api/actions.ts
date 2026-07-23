// Allotrope api — actions endpoints (Step 12c).
//
// Backend route: backend/allotrope/api/actions.py
// Sequence diagrams:
//   final design/diagrams/action-submit.drawio
//   final design/diagrams/action-list.drawio
//   final design/diagrams/action-detail.drawio
//   final design/diagrams/action-types-catalog.drawio

import type {
  Action,
  ActionDetail,
  ActionsPage,
  ActionTypeMeta,
  CreateActionPayload,
} from "../types";
import { fetchJson } from "./client";

export interface ListActionsOptions {
  limit?: number;
  offset?: number;
  status?: Action["status"];
  type?: string;
}

export async function listProjectActions(
  projectId: string,
  opts: ListActionsOptions = {},
): Promise<ActionsPage> {
  const params = new URLSearchParams();
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  if (opts.offset !== undefined) params.set("offset", String(opts.offset));
  if (opts.status) params.set("status", opts.status);
  if (opts.type) params.set("type", opts.type);
  const query = params.toString();
  return fetchJson<ActionsPage>(
    `/projects/${encodeURIComponent(projectId)}/actions${query ? `?${query}` : ""}`,
  );
}

export async function createAction(
  projectId: string,
  payload: CreateActionPayload,
): Promise<Action> {
  return fetchJson<Action>(
    `/projects/${encodeURIComponent(projectId)}/actions`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export async function getAction(actionId: string): Promise<ActionDetail> {
  return fetchJson<ActionDetail>(`/actions/${encodeURIComponent(actionId)}`);
}

export async function listActionTypes(): Promise<ActionTypeMeta[]> {
  const res = await fetchJson<{ items: ActionTypeMeta[] }>("/action-types");
  return res.items;
}

// --- Anomaly detection prep -----------------------------------------------
//
// The prep action sits in `needs_threshold` after the worker writes the
// composite score; the user explores thresholds live by POSTing to the
// preview endpoint and reading the returned mask URL + metrics. See
// ROADMAP step 14.5 for the design.

export interface ActionOutputSummary {
  id: string; // output_<uuid>
  action_id: string; // action_<uuid>
  artifact_path: string;
  summary: Record<string, unknown>;
  created_at: string;
}

export async function getActionOutput(
  outputId: string,
): Promise<ActionOutputSummary> {
  return fetchJson<ActionOutputSummary>(
    `/action-outputs/${encodeURIComponent(outputId)}`,
  );
}

export async function getActionSummaryJson(
  actionId: string,
): Promise<Record<string, unknown>> {
  // Hits /actions/{id}/files/summary.json — the file endpoint cached
  // forever per the action's write-once contract.
  const url = `/api/actions/${encodeURIComponent(actionId)}/files/summary.json`;
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) {
    const { ApiError } = await import("./client");
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as Record<string, unknown>;
}

export interface AnomalyDetectionPreviewMetrics {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  n_gt_positives: number;
}

export interface AnomalyDetectionPreviewResponse {
  threshold_absolute: number;
  threshold_percentile: number;
  dilation_kernel: number;
  n_anomalous: number;
  n_kept: number;
  metrics: AnomalyDetectionPreviewMetrics | null;
  mask_url: string;
}

export async function submitAnomalyDetectionPreview(
  actionId: string,
  params: {
    threshold: number;
    threshold_mode: "percentile" | "absolute";
    dilation_kernel: number;
  },
): Promise<AnomalyDetectionPreviewResponse> {
  return fetchJson<AnomalyDetectionPreviewResponse>(
    `/actions/${encodeURIComponent(actionId)}/anomaly_detection_preview`,
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  );
}

// Commit locks the user's chosen threshold + dilation onto the prep
// action: writes anomaly_mask.tif + metrics.json into the action
// output dir, marks output.summary.committed = true (so downstream
// pickers can filter), and flips action.status to "complete". Re-commit
// is allowed — overwrites the prior mask + metrics.
export interface AnomalyDetectionCommitResponse {
  threshold_absolute: number;
  threshold_percentile: number;
  dilation_kernel: number;
  n_anomalous: number;
  n_kept: number;
  metrics: AnomalyDetectionPreviewMetrics | null;
  mask_tif_path: string;
}

export async function submitAnomalyDetectionCommit(
  actionId: string,
  params: {
    threshold: number;
    threshold_mode: "percentile" | "absolute";
    dilation_kernel: number;
  },
): Promise<AnomalyDetectionCommitResponse> {
  return fetchJson<AnomalyDetectionCommitResponse>(
    `/actions/${encodeURIComponent(actionId)}/anomaly_detection_commit`,
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  );
}

export async function deleteAction(actionId: string): Promise<void> {
  const res = await fetch(`/api/actions/${encodeURIComponent(actionId)}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok && res.status !== 204) {
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      /* not json */
    }
    const { ApiError } = await import("./client");
    throw new ApiError(res.status, detail);
  }
}
