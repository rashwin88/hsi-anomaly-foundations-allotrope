// Allotrope api — annotation endpoints (Step 9).
//
// Backend: backend/allotrope/api/annotations.py
// Sequence diagram: final design/diagrams/annotation-attach.drawio (9e)

import type {
  Annotation,
  AnnotationAttachAccepted,
  AnnotationTypeCatalog,
  AnnotationsList,
} from "../types";
import { ApiError, fetchJson } from "./client";

const API_BASE = "/api";

/** Read the supported-annotation-types catalogue. Drives the form's
 *  type picker + per-type accept= for the file input. */
export async function getAnnotationTypeCatalog(): Promise<AnnotationTypeCatalog> {
  return fetchJson<AnnotationTypeCatalog>("/annotation-types");
}

/** List annotations attached to a scene. */
export async function listAnnotations(
  sceneId: string,
): Promise<AnnotationsList> {
  return fetchJson<AnnotationsList>(
    `/scenes/${encodeURIComponent(sceneId)}/annotations`,
  );
}

/** Attach an annotation file. Multipart upload; the api stages and
 *  enqueues an annotation_attach job. Caller polls the returned job id.
 *  `kind` is OPTIONAL — when omitted, the api infers from the filename
 *  extension. Only override when the extension is ambiguous across
 *  multiple registered types. */
export async function attachAnnotation(
  sceneId: string,
  args: {
    name: string;
    description?: string;
    file: File;
    kind?: string;
  },
): Promise<AnnotationAttachAccepted> {
  const form = new FormData();
  form.append("name", args.name);
  if (args.description) form.append("description", args.description);
  if (args.kind) form.append("annotation_type", args.kind);
  form.append("file", args.file, args.file.name);

  const res = await fetch(
    `${API_BASE}/scenes/${encodeURIComponent(sceneId)}/annotations`,
    {
      method: "POST",
      credentials: "include",
      body: form,
    },
  );
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      /* empty body or non-JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as AnnotationAttachAccepted;
}

/** Synchronous delete. 204 No Content on success. */
export async function deleteAnnotation(
  sceneId: string,
  annotationId: string,
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/scenes/${encodeURIComponent(sceneId)}/annotations/${encodeURIComponent(annotationId)}`,
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

/** URL for the worker-rendered RGBA overlay PNG. */
export function annotationOverlayUrl(
  sceneId: string,
  annotationId: string,
): string {
  return `${API_BASE}/scenes/${encodeURIComponent(sceneId)}/annotations/${encodeURIComponent(annotationId)}/overlay`;
}

/** Re-export so consumers don't need to import the type from `../types`
 *  alongside this module. */
export type { Annotation };
