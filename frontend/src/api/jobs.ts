// Allotrope api — jobs endpoints.
//
// Backend route: backend/allotrope/api/jobs.py
// Sequence diagram (showing where polling fits in): final design/diagrams/scene-onboard.drawio

import type { Job } from "../types";
import { fetchJson } from "./client";

/** Read a single job by wire-format id (e.g. "job_3f29c4a8-..."). */
export async function getJob(jobId: string): Promise<Job> {
  return fetchJson<Job>(`/jobs/${encodeURIComponent(jobId)}`);
}

export interface ListJobsOptions {
  limit?: number;
  offset?: number;
  status?: Job["status"];
  type?: string;
  project_id?: string;
}

export interface JobsPage {
  items: Job[];
  total: number;
  limit: number;
  offset: number;
}

export async function listJobs(opts: ListJobsOptions = {}): Promise<JobsPage> {
  const params = new URLSearchParams();
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  if (opts.offset !== undefined) params.set("offset", String(opts.offset));
  if (opts.status) params.set("status", opts.status);
  if (opts.type) params.set("type", opts.type);
  if (opts.project_id) params.set("project_id", opts.project_id);
  const q = params.toString();
  return fetchJson<JobsPage>(`/jobs${q ? `?${q}` : ""}`);
}
