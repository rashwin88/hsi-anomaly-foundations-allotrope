// Result + Export client (Step 17).

import { fetchJson } from "./client";

export interface ResultActionLine {
  id: string;
  type: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  started_at: string | null;
  completed_at: string | null;
  failure_reason: string | null;
  output_id: string | null;
  summary: Record<string, unknown> | null;
}

export interface ResultPublic {
  project: {
    id: string;
    name: string;
    description: string | null;
    created_at: string;
    scene_id: string;
    scene_name: string;
    scene_sensor_type: string;
  };
  actions: ResultActionLine[];
  visualization_count: number;
  note_count: number;
  annotation_count: number;
  last_action_completed_at: string | null;
  generated_at: string;
}

export interface ExportPublic {
  id: string;
  project_id: string;
  bundle_path: string;
  download_url: string;
  snapshot_at: string;
  size_bytes: number;
  format: string;
  created_at: string;
}

export interface ExportList {
  items: ExportPublic[];
}

export interface ExportAccepted {
  job_id: string;
  project_id: string;
}

export async function getProjectResult(
  projectId: string,
): Promise<ResultPublic> {
  return fetchJson<ResultPublic>(
    `/projects/${encodeURIComponent(projectId)}/result`,
  );
}

export async function createProjectExport(
  projectId: string,
): Promise<ExportAccepted> {
  return fetchJson<ExportAccepted>(
    `/projects/${encodeURIComponent(projectId)}/exports`,
    { method: "POST", body: JSON.stringify({}) },
  );
}

export async function listProjectExports(
  projectId: string,
): Promise<ExportList> {
  return fetchJson<ExportList>(
    `/projects/${encodeURIComponent(projectId)}/exports`,
  );
}
