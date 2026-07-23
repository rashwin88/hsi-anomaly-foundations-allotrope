// ActionTemplate CRUD client (Step 18).

import { ApiError, fetchJson } from "./client";

export interface ActionTemplate {
  id: string;
  type: string;
  name: string;
  description: string | null;
  configuration: Record<string, unknown>;
  is_system: boolean;
  created_at: string;
  updated_at: string;
}

export interface ActionTemplateList {
  items: ActionTemplate[];
}

export async function listActionTemplates(
  type?: string,
): Promise<ActionTemplateList> {
  const q = type ? `?type=${encodeURIComponent(type)}` : "";
  return fetchJson<ActionTemplateList>(`/action-templates${q}`);
}

export async function createActionTemplate(body: {
  type: string;
  name: string;
  description?: string;
  configuration?: Record<string, unknown>;
}): Promise<ActionTemplate> {
  return fetchJson<ActionTemplate>("/action-templates", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function patchActionTemplate(
  templateId: string,
  body: {
    name?: string;
    description?: string;
    configuration?: Record<string, unknown>;
  },
): Promise<ActionTemplate> {
  return fetchJson<ActionTemplate>(
    `/action-templates/${encodeURIComponent(templateId)}`,
    { method: "PATCH", body: JSON.stringify(body) },
  );
}

export async function deleteActionTemplate(templateId: string): Promise<void> {
  const res = await fetch(
    `/api/action-templates/${encodeURIComponent(templateId)}`,
    { method: "DELETE", credentials: "include" },
  );
  if (!res.ok && res.status !== 204) throw new ApiError(res.status);
}
