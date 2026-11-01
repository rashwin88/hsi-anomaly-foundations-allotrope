// The anomaly-points panel's drawing (spec addition D): only the committed anomaly pixels on a
// dimmed RGB, with a marker per object whose shape AND colour give the call.
//
// Two layers, both inside MatchMap's zoomed wrapper. The pixels are painted at image size (a
// canvas, like the classified overlay) so the footprint is exact when zoomed in. A single pixel
// is a fraction of a screen pixel at full-scene zoom, so each object also gets a marker at its
// centroid that keeps MARK_PX on screen at any zoom: a red square for man-made, a green circle
// for natural, each with a dark outline. Shape carries the call for colour-blind readers, colour
// for a glance (user, 2026-11-01: "the leaves should also be green"). Objects in the class
// selection get an ochre halo, so the selection stays distinct. Zoomed in past GLYPH_ZOOM the
// marker becomes a pin holding a building or a leaf. Markers are a few svg paths, not thousands
// of elements, because they are rebuilt on every zoom step; glyphs are drawn only in view.

import type { ReactNode } from "react";

import type { View } from "./MatchMap";
import { INK, NAMER_MANMADE, NATURAL_POINT, type NamerData, type NamerObject, POINT_COLOUR } from "./data";
import type { Call } from "./selection";

export const MARK_PX = 7;          // marker edge on screen, CSS px
const HALO_PX = 2;                 // ochre halo beyond the marker, CSS px
const GLYPH_PX = 16;
const GLYPH_ZOOM = 6;              // screen px per image px where glyphs take over
const PIXEL_ALPHA = 220;

const rgbOf = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const MM_RGB = rgbOf(NAMER_MANMADE);
const NAT_RGB = rgbOf(NATURAL_POINT);

// Committed anomaly pixels of the shown objects, coloured by call, with a dark one-pixel outline.
export function paintPoints(w: number, h: number, data: NamerData, calls: Map<number, Call>,
                            show: (o: NamerObject) => boolean): ImageData {
  const im = new ImageData(w, h);
  const d = im.data;
  const lit: number[] = [];
  for (const p of data.pixels) {
    if (p.row >= h || p.col >= w) continue;
    const o = data.byId.get(p.obj_id);
    if (!o || !show(o)) continue;
    const [r, g, b] = calls.get(o.obj_id)?.call === "manmade" ? MM_RGB : NAT_RGB;
    const i = (p.row * w + p.col) * 4;
    d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = PIXEL_ALPHA;
    lit.push(p.row * w + p.col);
  }
  for (const at of lit) {
    const row = Math.floor(at / w);
    const col = at - row * w;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const rr = row + dr;
        const cc = col + dc;
        if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
        const j = (rr * w + cc) * 4;
        if (d[j + 3] !== 0) continue;
        d[j] = 17; d[j + 1] = 17; d[j + 2] = 17; d[j + 3] = 190;
      }
    }
  }
  return im;
}

// 16 x 16 glyphs with different outlines: a pin holding a building, its tip at (8, 16) on the
// centroid (man-made), and a leaf centred on it (natural).
const PIN = "M8 16C8 16 2 10.6 2 6.6 2 3.3 4.7 1 8 1s6 2.3 6 5.6C14 10.6 8 16 8 16Z";
const BUILDING = "M5.5 9V4.5L8 3l2.5 1.5V9Z";
const LEAF = "M2 14C2 7 6 2 14 2 14 9 9 14 2 14Z";
const VEIN = "M2 14 10 6";

const square = (x: number, y: number, s: number) => `M${x - s / 2} ${y - s / 2}h${s}v${s}h${-s}Z`;
const circle = (x: number, y: number, s: number) =>
  `M${x - s / 2} ${y}a${s / 2} ${s / 2} 0 1 0 ${s} 0a${s / 2} ${s / 2} 0 1 0 ${-s} 0Z`;

export function pointMarkers(v: View, objects: NamerObject[], calls: Map<number, Call>,
                             inSelection: (o: NamerObject) => boolean): ReactNode {
  if (v.zoom >= GLYPH_ZOOM) {
    const k = (GLYPH_PX * v.px) / 16;
    const inView = objects.filter((o) => o.col >= v.box.c0 - 2 && o.col <= v.box.c1 + 2
      && o.row >= v.box.r0 - 2 && o.row <= v.box.r1 + 2);
    return inView.map((o) => {
      const mm = calls.get(o.obj_id)?.call === "manmade";
      const dy = mm ? 16 : 8;
      return (
        <g key={o.obj_id} data-kind={mm ? "pin" : "leaf"}
           transform={`translate(${o.col + 0.5 - 8 * k} ${o.row + 0.5 - dy * k}) scale(${k})`}>
          {inSelection(o) && (
            <path d={mm ? PIN : LEAF} fill="none" stroke={POINT_COLOUR} strokeWidth={4} strokeLinejoin="round" />
          )}
          <path d={mm ? PIN : LEAF} fill={mm ? NAMER_MANMADE : NATURAL_POINT} stroke={INK} strokeWidth={1.2}
                strokeLinejoin="round" />
          <path d={mm ? BUILDING : VEIN} fill={mm ? "#fffff8" : "none"} stroke={mm ? INK : "#fffff8"}
                strokeWidth={mm ? 0.8 : 1.2} />
        </g>
      );
    });
  }
  const s = MARK_PX * v.px;
  const hs = s + 2 * HALO_PX * v.px;
  let squares = "";
  let circles = "";
  let halos = "";
  for (const o of objects) {
    const x = o.col + 0.5;
    const y = o.row + 0.5;
    const mm = calls.get(o.obj_id)?.call === "manmade";
    if (mm) squares += square(x, y, s); else circles += circle(x, y, s);
    if (inSelection(o)) halos += mm ? square(x, y, hs) : circle(x, y, hs);
  }
  return (
    <>
      <path data-kind="halos" d={halos} fill={POINT_COLOUR} stroke="none" />
      <path data-kind="circles" d={circles} fill={NATURAL_POINT} stroke={INK} strokeWidth={v.px} />
      <path data-kind="squares" d={squares} fill={NAMER_MANMADE} stroke={INK} strokeWidth={v.px} />
    </>
  );
}
