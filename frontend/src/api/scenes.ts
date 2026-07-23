// Allotrope api — scenes endpoints.
//
// Backend route: backend/allotrope/api/scenes.py
// Sequence diagrams:
//   final design/diagrams/scenes-list.drawio    (GET /scenes)
//   final design/diagrams/scene-onboard.drawio  (POST /scenes/onboard)

import type {
  HistogramJson,
  OnboardAccepted,
  Scene,
  ScenesPage,
  SensorType,
  SpectrumResponse,
  VisualizationList,
} from "../types";
import { ApiError, fetchJson } from "./client";

export interface ListScenesOptions {
  limit?: number;
  offset?: number;
  sensor_type?: SensorType;
  /** Case-insensitive substring match against scene name. */
  q?: string;
}

const API_BASE = "/api";

/** Read a single scene by wire-format id (e.g. "scene_3f29c4a8-..."). */
export async function getScene(sceneId: string): Promise<Scene> {
  return fetchJson<Scene>(`/scenes/${encodeURIComponent(sceneId)}`);
}

/** List the pre-rendered visualizations the worker wrote at onboard time. */
export async function listVisualizations(
  sceneId: string,
): Promise<VisualizationList> {
  return fetchJson<VisualizationList>(
    `/scenes/${encodeURIComponent(sceneId)}/visualizations`,
  );
}

/** Pixel-value distribution + summary stats for a scene. */
export async function getHistogram(sceneId: string): Promise<HistogramJson> {
  return fetchJson<HistogramJson>(
    `/scenes/${encodeURIComponent(sceneId)}/histogram`,
  );
}

/** Reflectance spectrum at one (row, col). Hyperspectral only. */
export async function getSpectrum(
  sceneId: string,
  row: number,
  col: number,
): Promise<SpectrumResponse> {
  return fetchJson<SpectrumResponse>(
    `/scenes/${encodeURIComponent(sceneId)}/spectrum?row=${row}&col=${col}`,
  );
}

export async function listScenes(
  opts: ListScenesOptions = {},
): Promise<ScenesPage> {
  const params = new URLSearchParams();
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  if (opts.offset !== undefined) params.set("offset", String(opts.offset));
  if (opts.sensor_type) params.set("sensor_type", opts.sensor_type);
  if (opts.q !== undefined && opts.q.trim() !== "") params.set("q", opts.q.trim());
  const query = params.toString();
  const res = await fetch(`${API_BASE}/scenes${query ? `?${query}` : ""}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      /* body not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as ScenesPage;
}

/**
 * Upload files for onboarding via multipart/form-data.
 *
 * Each file's `webkitRelativePath` (when present) is used as the multipart
 * filename so the api can reconstruct folder structure on disk. For
 * single-file sensors (PRISMA, Landsat 9) `webkitRelativePath` is empty
 * and we fall back to `file.name`.
 *
 * Note: NOT using fetchJson here because the body is FormData, not JSON.
 * Setting Content-Type explicitly would break the boundary header the
 * browser auto-generates — we let the browser handle it.
 */
export interface UploadProgress {
  /** Bytes successfully sent so far. */
  loaded: number;
  /** Total bytes the browser knows about. 0 until first progress event. */
  total: number;
  /** Convenience: `loaded / total` clipped to [0,1]; 0 if total unknown. */
  fraction: number;
}

/**
 * Like `onboardScene`, but driven by XMLHttpRequest so the caller can
 * observe upload progress. `fetch` doesn't expose a request-side
 * progress event; we'd have to manually chunk to ReadableStream, which
 * doesn't work in Safari for multipart uploads. XHR is the boring,
 * cross-browser answer.
 *
 * The `onProgress` callback fires synchronously as the browser ships
 * bytes. The returned promise resolves with the api's JSON response
 * once the server has accepted the staged upload, or rejects with an
 * `ApiError` (network failure → status 0, detail "network_error").
 */
export async function onboardScene(
  sensorType: SensorType,
  files: File[],
  onProgress?: (p: UploadProgress) => void,
  /** Fires once the browser has finished pushing every byte but before
   * the server's 202 lands. Use it to flip the UI from "Uploading XX%"
   * to "Staging files…" — the api spends a few seconds streaming the
   * multipart spool to disk before it enqueues the job. */
  onUploadComplete?: () => void,
): Promise<OnboardAccepted> {
  const form = new FormData();
  form.append("sensor_type", sensorType);
  for (const file of files) {
    const filename = (file as File & { webkitRelativePath?: string })
      .webkitRelativePath || file.name;
    form.append("files", file, filename);
  }

  return new Promise<OnboardAccepted>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/scenes/onboard`, true);
    xhr.withCredentials = true;
    // Don't set Content-Type — the browser auto-generates the multipart
    // boundary header for us.

    if (onProgress) {
      xhr.upload.onprogress = (event: ProgressEvent) => {
        const total = event.lengthComputable ? event.total : 0;
        const loaded = event.loaded;
        const fraction = total > 0 ? Math.min(1, loaded / total) : 0;
        onProgress({ loaded, total, fraction });
      };
      xhr.upload.onload = () => {
        // Fire one last "100%" event when the upload completes, even if
        // the final progress event was a few bytes short.
        const total = xhr.upload as unknown as { total?: number };
        onProgress({
          loaded: total.total ?? 0,
          total: total.total ?? 0,
          fraction: 1,
        });
      };
    }
    // Independently of onProgress, surface "upload bytes done" to the
    // caller so it can switch from "Uploading…" to "Staging…" while the
    // api spools the multipart body to disk. Wired on upload.onload via
    // a second listener so it fires even when onProgress isn't supplied.
    xhr.upload.addEventListener("load", () => {
      onUploadComplete?.();
    });

    xhr.onload = () => {
      const status = xhr.status;
      if (status >= 200 && status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as OnboardAccepted);
        } catch (parseErr) {
          reject(new ApiError(status, (parseErr as Error).message));
        }
        return;
      }
      let detail: string | undefined;
      try {
        const body = JSON.parse(xhr.responseText) as { detail?: string };
        detail = body.detail;
      } catch {
        /* not json */
      }
      reject(new ApiError(status, detail));
    };

    xhr.onerror = () => reject(new ApiError(0, "network_error"));
    xhr.onabort = () => reject(new ApiError(0, "upload_aborted"));

    xhr.send(form);
  });
}

export interface SceneInUseDetail {
  code: "scene_in_use";
  blocking_project_ids: string[];
}

const API_BASE_ROOT = "/api";

/**
 * Delete a Scene. Throws ApiError on failure. The api returns 409 with
 * a structured detail `{code, blocking_project_ids}` when one or more
 * Projects still reference the Scene; use `parseSceneInUse` to recover
 * the typed shape from ApiError.detail.
 */
export async function deleteScene(sceneId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_ROOT}/scenes/${encodeURIComponent(sceneId)}`,
    { method: "DELETE", credentials: "include" },
  );
  if (!res.ok && res.status !== 204) {
    let detail: unknown;
    try {
      const body = (await res.json()) as { detail?: unknown };
      detail = body.detail;
    } catch {
      /* not json */
    }
    const detailStr =
      typeof detail === "string" ? detail : JSON.stringify(detail ?? "");
    throw new ApiError(res.status, detailStr);
  }
}

/** Recover the structured 409 detail from an ApiError.detail string. */
export function parseSceneInUse(
  detail: string | undefined,
): SceneInUseDetail | null {
  if (!detail) return null;
  try {
    const parsed = JSON.parse(detail) as Partial<SceneInUseDetail>;
    if (
      parsed &&
      parsed.code === "scene_in_use" &&
      Array.isArray(parsed.blocking_project_ids)
    ) {
      return {
        code: "scene_in_use",
        blocking_project_ids: parsed.blocking_project_ids as string[],
      };
    }
  } catch {
    /* not JSON */
  }
  return null;
}
