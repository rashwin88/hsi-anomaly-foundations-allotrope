// One map of the spectral match viewer: scene RGB, a result overlay, markers drawn in image
// coordinates, the selection and hover rings, with panzoom, hover, click-to-pixel and an
// optional link that keeps two maps on the same view (the classified map and the anomaly-points
// map are small multiples: same grid, same scale, moved together).
//
// Every layer (rgb img, overlay img or canvas, marker svg) lives inside ONE wrapper div and
// panzoom transforms the wrapper, so the layers cannot drift apart. The earlier viewer
// panzoomed the rgb img and moved the other layers after it on "transform" events; when the
// panzoom instance was recreated nothing re-subscribed, and the dots stayed put while the
// image zoomed (user report, 2026-11-01). Markers that keep a minimum on-screen size are drawn
// in image units scaled by 1 / zoom, still inside the wrapper.
//
// Linking: panzoom emits "transform" a frame after a change (it applies transforms in
// requestAnimationFrame), so a follower's echo cannot be told apart by a flag set around the
// call. A map instead re-emits only a transform that differs from the one it was last told to
// apply; comparing with the link's latest would bounce a stale echo back to the leader mid-drag.
// Both stages must be the same size for one transform to mean the same view: the screen's grid
// gives them equal cells.
//
// The wrapper is sized in JS to the largest box of the image's aspect ratio that fits the
// stage, so an extreme aspect (AVIRIS-NG strips) cannot collapse it. Pointer -> pixel goes
// through the wrapper's getBoundingClientRect(), which already includes the transform.
// Assumes rgb.png is at the scene's full resolution, so image pixel == scene (row, col).

import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";

type Pz = {
  dispose: () => void;
  getTransform: () => { x: number; y: number; scale: number };
  moveTo: (x: number, y: number) => void;
  zoomAbs: (x: number, y: number, scale: number) => void;
  on: (evt: string, fn: () => void) => void;
};

export interface Box { r0: number; r1: number; c0: number; c1: number }
export interface Px { row: number; col: number }
export interface View {
  px: number;      // image units per screen px
  zoom: number;    // screen px per image px
  box: Box;        // the image area in view
}
interface Tf { x: number; y: number; scale: number }
const sameTf = (a: Tf | null, b: Tf) =>
  a != null && Math.abs(a.x - b.x) < 0.01 && Math.abs(a.y - b.y) < 0.01 && Math.abs(a.scale - b.scale) < 1e-6;

// Shared view of linked maps; `last` lets a map mounted later (the narrow-screen switch) start
// where the other one is.
export class PanLink {
  last: Tf | null = null;
  private subs = new Map<string, (t: Tf) => void>();
  join(id: string, fn: (t: Tf) => void) {
    this.subs.set(id, fn);
    return () => { this.subs.delete(id); };
  }
  emit(from: string, t: Tf) {
    this.last = t;
    for (const [id, fn] of this.subs) if (id !== from) fn(t);
  }
}

// A drag of more than this many CSS px is a pan, not a click.
const CLICK_SLOP_PX = 5;
// An object zoomed to fills about this fraction of the view's shorter side.
const ZOOM_FILL = 1 / 6;
const MAX_ZOOM = 30;

export function MatchMap({
  id = "map",
  rgbUrl,
  dimRgb,
  overlayUrl,
  overlayImage,
  markers,
  selected,
  hovered,
  marked,
  focus,
  caption,
  readout,
  status,
  link,
  onDims,
  onHover,
  onPick,
}: {
  id?: string;
  rgbUrl: string;
  dimRgb?: boolean;                    // dimmed, desaturated RGB under bright points
  overlayUrl?: string;                 // a ready-made transparent PNG (older actions)
  overlayImage?: ImageData | null;     // or one painted client-side at image size
  markers?: (v: View) => ReactNode;    // svg content in image coordinates
  selected?: Box | null;               // ringed
  hovered?: Box | null;                // thin ring
  marked?: Px | null;                  // the hovered / pinned pixel (older actions)
  focus?: { box: Box; seq: number } | null;   // zoom to box whenever seq changes
  caption?: ReactNode;                 // the map's meaning, in the margin below it
  readout?: ReactNode;                 // hover text, before the caption (SegmentationViewer)
  status?: ReactNode;
  link?: PanLink;
  onDims?: (w: number, h: number) => void;
  onHover?: (px: Px | null, zoom: number) => void;
  onPick?: (px: Px, zoom: number) => void;
}) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pzRef = useRef<Pz | null>(null);
  const downRef = useRef<{ x: number; y: number } | null>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [fit, setFit] = useState<{ w: number; h: number; ox: number; oy: number; sw: number; sh: number } | null>(null);
  const [tf, setTf] = useState<Tf>({ x: 0, y: 0, scale: 1 });

  const onImgLoad = () => {
    const img = imgRef.current;
    if (img?.naturalWidth) setNat({ w: img.naturalWidth, h: img.naturalHeight });
  };
  // A cached image can finish before React attaches onLoad.
  useEffect(() => {
    if (imgRef.current?.complete) onImgLoad();
  }, [rgbUrl]);
  useEffect(() => {
    if (nat) onDims?.(nat.w, nat.h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nat]);

  // Fit the wrapper into the stage, now and on every resize.
  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage || !nat) return;
    const measure = () => {
      const sw = stage.clientWidth;
      const sh = stage.clientHeight;
      const s = Math.min(sw / nat.w, sh / nat.h);
      const w = nat.w * s;
      const h = nat.h * s;
      setFit({ w, h, ox: (sw - w) / 2, oy: (sh - h) / 2, sw, sh });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(stage);
    return () => ro.disconnect();
  }, [nat]);

  // One panzoom on the wrapper, created once the wrapper has its size.
  const ready = fit != null;
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    let inst: Pz | null = null;
    let leave: (() => void) | null = null;
    void import("panzoom").then((mod) => {
      const target = wrapRef.current;
      if (cancelled || !target) return;
      const pz = mod.default(target, {
        // No bounds: panzoom drops the scale change of a zoomAbs whose result it had to
        // pull back inside the bounds, which broke zoom-to-object and Fit. Fit recovers.
        maxZoom: MAX_ZOOM, minZoom: 1, bounds: false,
        smoothScroll: false, zoomDoubleClickSpeed: 1,
      }) as unknown as Pz;
      inst = pz;
      let told: Tf | null = null;
      const apply = (t: Tf) => { told = t; pz.zoomAbs(0, 0, t.scale); pz.moveTo(t.x, t.y); };
      pz.on("transform", () => {
        const t = { ...pz.getTransform() };
        setTf(t);
        if (!sameTf(told, t)) link?.emit(id, t);
      });
      leave = link?.join(id, apply) ?? null;
      if (link?.last) apply(link.last);
      pzRef.current = pz;
    });
    return () => {
      cancelled = true;
      leave?.();
      inst?.dispose();             // also drops the transform listener
      pzRef.current = null;
    };
  }, [ready, link, id]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !overlayImage) return;
    canvas.width = overlayImage.width;
    canvas.height = overlayImage.height;
    canvas.getContext("2d")?.putImageData(overlayImage, 0, 0);
  }, [overlayImage]);

  // panzoom maps a wrapper-local point p to stage ox + t + scale * p, so to centre the box's
  // centre c: t = stage centre - offset - scale * c.
  useEffect(() => {
    const inst = pzRef.current;
    const stage = stageRef.current;
    if (!focus || !inst || !stage || !fit || !nat) return;
    const s0 = fit.w / nat.w;
    const { r0, r1, c0, c1 } = focus.box;
    const bw = (c1 - c0 + 1) * s0;
    const bh = (r1 - r0 + 1) * s0;
    const sw = stage.clientWidth;
    const sh = stage.clientHeight;
    const z = Math.min(MAX_ZOOM * 0.66, Math.max(2,
      Math.min(sw, sh) * ZOOM_FILL / Math.max(bw, bh, 1)));
    inst.zoomAbs(0, 0, z);
    inst.moveTo(sw / 2 - fit.ox - z * ((c0 + c1 + 1) / 2) * s0,
                sh / 2 - fit.oy - z * ((r0 + r1 + 1) / 2) * s0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.seq]);

  const reset = () => {
    pzRef.current?.zoomAbs(0, 0, 1);
    pzRef.current?.moveTo(0, 0);
  };

  const zoom = fit && nat ? (fit.w / nat.w) * tf.scale : 1;
  const pixelAt = (e: React.MouseEvent): Px | null => {
    const wrap = wrapRef.current;
    if (!wrap || !nat) return null;
    const r = wrap.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * nat.w);
    const row = Math.floor(((e.clientY - r.top) / r.height) * nat.h);
    return row >= 0 && col >= 0 && row < nat.h && col < nat.w ? { row, col } : null;
  };

  // Stroke widths in image units that come out a fixed number of CSS px at any zoom.
  const px = 1 / zoom;
  const view: View | null = fit && nat ? {
    px, zoom,
    box: { c0: (-fit.ox - tf.x) / zoom, c1: (fit.sw - fit.ox - tf.x) / zoom,
           r0: (-fit.oy - tf.y) / zoom, r1: (fit.sh - fit.oy - tf.y) / zoom },
  } : null;
  const ring = (b: Box, w: number, pad: number, colour: string) => (
    <rect x={b.c0 - pad * px} y={b.r0 - pad * px}
          width={b.c1 - b.c0 + 1 + 2 * pad * px} height={b.r1 - b.r0 + 1 + 2 * pad * px}
          fill="none" stroke={colour} strokeWidth={w * px} />
  );

  return (
    <figure className="smv-map">
      <div
        ref={stageRef}
        className="smv-map__stage"
        onMouseDownCapture={(e) => { downRef.current = { x: e.clientX, y: e.clientY }; }}
        onMouseMoveCapture={(e) => onHover?.(pixelAt(e), zoom)}
        onMouseLeave={() => onHover?.(null, zoom)}
        onClickCapture={(e) => {
          const d = downRef.current;
          if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) > CLICK_SLOP_PX) return;
          const p = pixelAt(e);
          if (p) onPick?.(p, zoom);
        }}
        onDoubleClickCapture={reset}
      >
        <div
          ref={wrapRef}
          className="smv-map__wrap"
          style={fit ? { left: fit.ox, top: fit.oy, width: fit.w, height: fit.h } : { visibility: "hidden" }}
        >
          <img ref={imgRef} src={rgbUrl} alt="Scene RGB"
               className={`smv-map__layer${dimRgb ? " smv-map__rgb--dim" : ""}`}
               draggable={false} onLoad={onImgLoad} />
          {overlayUrl && (
            <img src={overlayUrl} alt="" className="smv-map__layer" draggable={false} />
          )}
          {overlayImage && <canvas ref={canvasRef} className="smv-map__layer" />}
          {nat && view && (
            <svg className="smv-map__layer smv-map__vector" viewBox={`0 0 ${nat.w} ${nat.h}`}
                 preserveAspectRatio="none" aria-hidden="true">
              {markers?.(view)}
              {hovered && ring(hovered, 1, 3, "#111")}
              {selected && <>{ring(selected, 3.5, 3, "#111")}{ring(selected, 1.5, 3, "#d4a017")}</>}
              {marked && (
                <rect x={marked.col - px} y={marked.row - px} width={1 + 2 * px} height={1 + 2 * px}
                      fill="none" stroke="#ffffff" strokeWidth={1.5 * px} />
              )}
            </svg>
          )}
        </div>
      </div>
      <figcaption className="smv-map__caption">
        <span className="smv-map__meaning">
          {readout && <span className="smv-map__readout">{readout}</span>}
          {readout && caption ? " · " : null}{caption}
        </span>
        {status && <span className="smv-map__status">{status}</span>}
        <button type="button" className="smv-toggle" onClick={reset}
                title="Fit the whole scene (or double-click the map)">fit</button>
      </figcaption>
    </figure>
  );
}
