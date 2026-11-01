// Story GIF of a spectral_library_match action — GET /actions/{id}/story.gif.
//
// One looping GIF of how the scene was narrowed down to the selected anomalies, for presentations
// (docs/lld/story-gif.md). The selection has the class-driven export's meaning: classes (comma list
// of natural, manmade or material keys), confident_only, manmade_p. It is NOT a submission file:
// one frame is a heatmap, so it never goes into an export pack.

const API_BASE = "/api";

export interface StorySelection {
  classes?: string[];        // default ["manmade"] on the server
  confidentOnly?: boolean;
  manmadeP?: number;         // man-made cut on an object's mean P(man-made); server default 0.5
  size?: number;             // longer side of the map in px, 320-1600; default 960
}

/** The GIF's URL. Usable as an <img src> as well as a download. */
export function storyGifUrl(actionId: string, sel: StorySelection = {}): string {
  const q = new URLSearchParams();
  if (sel.classes && sel.classes.length) q.set("classes", sel.classes.join(","));
  if (sel.confidentOnly) q.set("confident_only", "true");
  if (sel.manmadeP != null) q.set("manmade_p", String(sel.manmadeP));
  if (sel.size != null) q.set("size", String(sel.size));
  const qs = q.toString();
  return `${API_BASE}/actions/${encodeURIComponent(actionId)}/story.gif${qs ? `?${qs}` : ""}`;
}

/** Fetch the GIF and save it under the server's file name. Throws Error(detail) on a non-2xx. */
export async function downloadStoryGif(actionId: string, sel: StorySelection = {}): Promise<void> {
  const r = await fetch(storyGifUrl(actionId, sel), { credentials: "include" });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(`Story GIF failed: ${detail}`);
  }
  const blob = await r.blob();
  const name = r.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1]
    ?? `allotrope_story_${actionId.replace(/^action_/, "").slice(0, 8)}.gif`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
