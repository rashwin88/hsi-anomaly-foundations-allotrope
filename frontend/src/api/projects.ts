// Allotrope api — projects endpoints (Step 10).
//
// Backend route: backend/allotrope/api/projects.py
// Sequence diagrams:
//   final design/diagrams/projects-create.drawio
//   final design/diagrams/projects-list.drawio
//   final design/diagrams/projects-delete.drawio

import type {
  CreateProjectPayload,
  Project,
  ProjectDetail,
  ProjectsPage,
} from "../types";
import { ApiError, fetchJson } from "./client";

const API_BASE = "/api";

export interface ListProjectsOptions {
  limit?: number;
  offset?: number;
  scene_id?: string;
  user_id?: string;
}

export async function listProjects(
  opts: ListProjectsOptions = {},
): Promise<ProjectsPage> {
  const params = new URLSearchParams();
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  if (opts.offset !== undefined) params.set("offset", String(opts.offset));
  if (opts.scene_id) params.set("scene_id", opts.scene_id);
  if (opts.user_id) params.set("user_id", opts.user_id);
  const query = params.toString();
  return fetchJson<ProjectsPage>(`/projects${query ? `?${query}` : ""}`);
}

export async function getProject(projectId: string): Promise<ProjectDetail> {
  return fetchJson<ProjectDetail>(`/projects/${encodeURIComponent(projectId)}`);
}

export async function createProject(
  payload: CreateProjectPayload,
): Promise<Project> {
  return fetchJson<Project>("/projects", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deleteProject(projectId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/projects/${encodeURIComponent(projectId)}`,
    {
      method: "DELETE",
      credentials: "include",
    },
  );
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      /* */
    }
    throw new ApiError(res.status, detail);
  }
}
