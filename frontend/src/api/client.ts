// Allotrope api — fetch wrapper.
//
// Single-origin discipline: all requests go through `/api/*` which nginx
// proxies to the api container. The browser sees one origin (the frontend's),
// so cookies set by the api are sent on subsequent requests automatically —
// no CORS, no SameSite=Lax relaxation needed.

const API_BASE = "/api";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail?: string,
  ) {
    super(`API ${status}${detail ? `: ${detail}` : ""}`);
    this.name = "ApiError";
  }
}

/** Issue a JSON request to the api and parse the JSON response.
 *
 * - Sends cookies (credentials: "include") so the JWT cookie auto-attaches.
 * - Sets Content-Type: application/json by default.
 * - Throws ApiError on non-2xx responses, exposing the status and the
 *   `detail` field from the api's error body when present.
 * - Returns undefined for 204 No Content.
 */
export async function fetchJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      // body wasn't JSON — leave detail undefined
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}
