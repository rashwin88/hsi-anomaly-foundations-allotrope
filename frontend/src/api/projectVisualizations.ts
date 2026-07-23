// Project-level Visualization api (Step 15, scoped 2026-05-11).
//
// The save flow is multipart: a flattened viewer PNG + a JSON metadata
// blob with the source pointer and any view_state we want to pin.

import { ApiError } from "./client";

const API_BASE = "/api";

export interface ProjectVisualizationSource {
  kind: "scene" | "action_output";
  scene_id?: string;
  action_id?: string;
}

export interface ProjectVisualization {
  id: string;
  project_id: string;
  source_kind: "scene" | "action_output";
  source_scene_id: string | null;
  source_action_output_id: string | null;
  name: string;
  description: string | null;
  artifact_path: string;
  image_url: string;
  view_state: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ProjectVisualizationList {
  items: ProjectVisualization[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateVisualizationInput {
  projectId: string;
  source: ProjectVisualizationSource;
  name: string;
  description?: string;
  viewState?: Record<string, unknown>;
  imageBlob: Blob;
}

export async function createVisualization(
  input: CreateVisualizationInput,
): Promise<ProjectVisualization> {
  const form = new FormData();
  form.append(
    "meta",
    JSON.stringify({
      source: input.source,
      name: input.name,
      description: input.description,
      view_state: input.viewState ?? {},
    }),
  );
  form.append("image", input.imageBlob, "view.png");
  const res = await fetch(
    `${API_BASE}/projects/${encodeURIComponent(input.projectId)}/visualizations`,
    { method: "POST", credentials: "include", body: form },
  );
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      // not json
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as ProjectVisualization;
}

export async function listVisualizations(
  projectId: string,
  limit = 50,
  offset = 0,
): Promise<ProjectVisualizationList> {
  const res = await fetch(
    `${API_BASE}/projects/${encodeURIComponent(projectId)}/visualizations?limit=${limit}&offset=${offset}`,
    { credentials: "include" },
  );
  if (!res.ok) throw new ApiError(res.status);
  return (await res.json()) as ProjectVisualizationList;
}

export async function patchVisualization(
  vizId: string,
  body: { name?: string; description?: string },
): Promise<ProjectVisualization> {
  const res = await fetch(
    `${API_BASE}/visualizations/${encodeURIComponent(vizId)}`,
    {
      method: "PATCH",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) throw new ApiError(res.status);
  return (await res.json()) as ProjectVisualization;
}

export async function deleteVisualization(vizId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/visualizations/${encodeURIComponent(vizId)}`,
    { method: "DELETE", credentials: "include" },
  );
  if (!res.ok && res.status !== 204) throw new ApiError(res.status);
}

export function visualizationImageUrl(vizId: string): string {
  return `${API_BASE}/visualizations/${encodeURIComponent(vizId)}/image`;
}
