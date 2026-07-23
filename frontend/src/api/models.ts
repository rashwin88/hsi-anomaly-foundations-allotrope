// Allotrope api — foundation model catalog endpoints.
//
// Backend route: backend/allotrope/api/models.py
// Sequence diagrams:
//   final design/diagrams/models-list.drawio
//   final design/diagrams/models-detail.drawio

import type { ModelDetail, ModelSummary } from "../types";
import { fetchJson } from "./client";

export async function listModels(): Promise<ModelSummary[]> {
  return fetchJson<ModelSummary[]>("/models");
}

export async function getModel(architecture: string): Promise<ModelDetail> {
  return fetchJson<ModelDetail>(
    `/models/${encodeURIComponent(architecture)}`,
  );
}
