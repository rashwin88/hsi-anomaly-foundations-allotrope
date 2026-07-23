// Visualizations panel for the Scene Detail page (three-column layout).
//
// This file exports two pieces the page composes into separate grid
// cells:
//   <SceneVizCenter />  — filmstrip + active viewport (with panzoom)
//   <SceneVizRail   />  — histogram + spectrum / band browser
//
// The two halves share state via the lifted hook `useSceneViz()`. The
// page calls `useSceneViz(sceneId, isHyperspectral)` once and passes
// the returned controller to both components.
//
// Active modes:
//   color | nir | swir | ndvi    — pre-rendered PNG with pan/zoom.
//                                  color tile on hyperspectral scenes
//                                  also supports pixel-pick → spectrum.
//   band_mosaic                  — pre-rendered PNG (visual overview).
//   bands                        — interactive: pick any band, render
//                                  on-demand via the worker's bands api.
//
// Sequence diagram: final design/diagrams/scene-visualizations.drawio (8.5d)

import {
  type MouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { annotationOverlayUrl, listAnnotations } from "../api/annotations";
import { ApiError } from "../api/client";
import {
  getHistogram,
  getSpectrum,
  listVisualizations,
} from "../api/scenes";
import type {
  Annotation,
  HistogramJson,
  SpectrumResponse,
  VisualizationList,
} from "../types";

// ---------------------------------------------------------------------
// Synthetic 'bands' kind — not a worker-rendered PNG; the frontend
// adds it to the gallery for hyperspectral scenes so the user has an
// entry point into the band browser.
// ---------------------------------------------------------------------
const BANDS_KIND = "bands";
const BANDS_LABEL = "Band browser";

// ---------------------------------------------------------------------
// Shared controller — owns all state. Returned by useSceneViz.
// ---------------------------------------------------------------------

export interface OverlayState {
  annotation: Annotation;
  visible: boolean;
  /** 0–1 opacity slider value. */
  opacity: number;
  /** Dot radius in OUTPUT pixels. `null` → use api default (no query). */
  radius: number | null;
}

export interface SceneVizController {
  sceneId: string;
  isHyperspectral: boolean;
  list: VisualizationList | null;
  listError: string | null;
  activeKind: string | null;
  setActiveKind: (k: string) => void;
  pickedFrac: { x: number; y: number } | null;
  setPickedFrac: (p: { x: number; y: number } | null) => void;
  spectrum: SpectrumResponse | null;
  spectrumLoading: boolean;
  spectrumError: string | null;
  fetchSpectrum: (fracX: number, fracY: number) => void;
  selectedBandIndex: number | null;
  setSelectedBandIndex: (i: number | null) => void;

  // Annotations / overlays.
  overlays: OverlayState[];
  overlaysLoading: boolean;
  reloadAnnotations: () => Promise<void>;
  toggleOverlay: (annotationId: string) => void;
  setOverlayOpacity: (annotationId: string, opacity: number) => void;
  setOverlayRadius: (annotationId: string, radius: number | null) => void;
}

export function useSceneViz(
  sceneId: string,
  isHyperspectral: boolean,
): SceneVizController {
  const [list, setList] = useState<VisualizationList | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [activeKind, setActiveKind] = useState<string | null>(null);
  const [pickedFrac, setPickedFrac] = useState<{ x: number; y: number } | null>(null);
  const [spectrum, setSpectrum] = useState<SpectrumResponse | null>(null);
  const [spectrumLoading, setSpectrumLoading] = useState(false);
  const [spectrumError, setSpectrumError] = useState<string | null>(null);
  const [selectedBandIndex, setSelectedBandIndex] = useState<number | null>(null);

  // Overlays — fetched once at mount, refreshed after attach/delete.
  const [overlays, setOverlays] = useState<OverlayState[]>([]);
  const [overlaysLoading, setOverlaysLoading] = useState(true);

  const reloadAnnotations = async () => {
    setOverlaysLoading(true);
    try {
      const res = await listAnnotations(sceneId);
      setOverlays((prev) => {
        // Preserve user's visibility / opacity / radius choices for
        // annotations that survive the refresh; default new ones to
        // visible at default opacity and the api's default radius.
        const prevById = new Map(prev.map((o) => [o.annotation.id, o]));
        return res.items.map((a) => {
          const existing = prevById.get(a.id);
          return existing
            ? { ...existing, annotation: a }
            : { annotation: a, visible: true, opacity: 0.6, radius: null };
        });
      });
    } catch (err) {
      // Non-fatal — overlays are an enhancement, not a blocker. Log
      // for visibility but don't surface a UI error.
      // eslint-disable-next-line no-console
      console.warn("listAnnotations failed", err);
    } finally {
      setOverlaysLoading(false);
    }
  };

  useEffect(() => {
    void reloadAnnotations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneId]);

  const toggleOverlay = (annotationId: string) => {
    setOverlays((prev) =>
      prev.map((o) =>
        o.annotation.id === annotationId ? { ...o, visible: !o.visible } : o,
      ),
    );
  };

  const setOverlayOpacity = (annotationId: string, opacity: number) => {
    setOverlays((prev) =>
      prev.map((o) =>
        o.annotation.id === annotationId ? { ...o, opacity } : o,
      ),
    );
  };

  const setOverlayRadius = (annotationId: string, radius: number | null) => {
    setOverlays((prev) =>
      prev.map((o) =>
        o.annotation.id === annotationId ? { ...o, radius } : o,
      ),
    );
  };

  useEffect(() => {
    let cancelled = false;
    listVisualizations(sceneId)
      .then((res) => {
        if (cancelled) return;
        setList(res);
        if (res.items.length > 0) {
          setActiveKind(res.items[0].kind);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setListError(err.detail ?? `Error: HTTP ${err.status}`);
        } else {
          setListError("Could not reach the server.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [sceneId]);

  // Reset transient state when the active tile changes.
  useEffect(() => {
    setPickedFrac(null);
    setSpectrum(null);
    setSpectrumError(null);
  }, [activeKind, sceneId]);

  // Cache of scene cube dimensions so we don't probe (0,0) on every
  // pick. Filled by the first successful spectrum fetch (or the (0,0)
  // probe used as a fallback when the guess lands out of range).
  const dimsRef = useRef<{ height: number; width: number } | null>(null);

  const fetchSpectrum = (fracX: number, fracY: number) => {
    setSpectrumLoading(true);
    setSpectrumError(null);

    const fetchAt = (row: number, col: number) =>
      getSpectrum(sceneId, row, col);

    const refetchWithCorrectDims = (height: number, width: number) => {
      dimsRef.current = { height, width };
      const realRow = Math.min(Math.max(0, Math.round(fracY * height)), height - 1);
      const realCol = Math.min(Math.max(0, Math.round(fracX * width)), width - 1);
      return fetchAt(realRow, realCol);
    };

    // Fast path: we already know the cube dimensions from a prior pick.
    if (dimsRef.current) {
      const { height, width } = dimsRef.current;
      const row = Math.min(Math.max(0, Math.round(fracY * height)), height - 1);
      const col = Math.min(Math.max(0, Math.round(fracX * width)), width - 1);
      fetchAt(row, col)
        .then((resp) => setSpectrum(resp))
        .catch((err) => {
          if (err instanceof ApiError)
            setSpectrumError(err.detail ?? `Error: HTTP ${err.status}`);
          else setSpectrumError("Could not reach the server.");
        })
        .finally(() => setSpectrumLoading(false));
      return;
    }

    // Cold path: probe (0,0) — always in range — to learn the cube
    // shape, then refetch at the real pick coords.
    fetchAt(0, 0)
      .then((resp) => refetchWithCorrectDims(resp.height, resp.width))
      .then((resp) => setSpectrum(resp))
      .catch((err) => {
        if (err instanceof ApiError) {
          setSpectrumError(err.detail ?? `Error: HTTP ${err.status}`);
        } else {
          setSpectrumError("Could not reach the server.");
        }
      })
      .finally(() => setSpectrumLoading(false));
  };

  return {
    sceneId,
    isHyperspectral,
    list,
    listError,
    activeKind,
    setActiveKind,
    pickedFrac,
    setPickedFrac,
    spectrum,
    spectrumLoading,
    spectrumError,
    fetchSpectrum,
    selectedBandIndex,
    setSelectedBandIndex,
    overlays,
    overlaysLoading,
    reloadAnnotations,
    toggleOverlay,
    setOverlayOpacity,
    setOverlayRadius,
  };
}

// ---------------------------------------------------------------------
// Center column — filmstrip + active viewport.
// ---------------------------------------------------------------------

export function SceneVizCenter({ ctrl }: { ctrl: SceneVizController }) {
  if (ctrl.listError) {
    return (
      <div className="viz-panel">
        <p className="form__error" role="alert">{ctrl.listError}</p>
      </div>
    );
  }
  if (!ctrl.list) {
    return (
      <div className="viz-panel">
        <p className="scene-detail__hint">Loading visualizations…</p>
      </div>
    );
  }
  if (ctrl.list.items.length === 0) {
    return (
      <div className="viz-panel">
        <p className="scene-detail__hint">
          No visualizations rendered for this scene yet. Re-onboard to generate.
        </p>
      </div>
    );
  }

  // Synthetic Bands tile — only on hyperspectral scenes. Appended at
  // the end of the filmstrip.
  const filmstripItems = [
    ...ctrl.list.items.map((it) => ({
      kind: it.kind,
      label: it.label,
      image_url: it.image_url,
    })),
    ...(ctrl.isHyperspectral
      ? [{ kind: BANDS_KIND, label: BANDS_LABEL, image_url: null }]
      : []),
  ];

  return (
    <div className="viz-center">
      <Filmstrip
        items={filmstripItems}
        activeKind={ctrl.activeKind}
        onSelect={ctrl.setActiveKind}
      />
      <ActiveViewport ctrl={ctrl} />
    </div>
  );
}

interface FilmstripItem {
  kind: string;
  label: string;
  image_url: string | null;  // null for synthetic kinds (bands)
}

function Filmstrip({
  items,
  activeKind,
  onSelect,
}: {
  items: FilmstripItem[];
  activeKind: string | null;
  onSelect: (k: string) => void;
}) {
  return (
    <div className="viz-filmstrip" role="tablist" aria-label="Visualizations">
      {items.map((it) => (
        <button
          key={it.kind}
          type="button"
          role="tab"
          aria-selected={it.kind === activeKind}
          className={
            "viz-thumb" + (it.kind === activeKind ? " is-active" : "")
          }
          onClick={() => onSelect(it.kind)}
          title={it.label}
        >
          {it.image_url ? (
            <img src={`/api${it.image_url}`} alt={it.label} loading="lazy" />
          ) : (
            <span className="viz-thumb__icon" aria-hidden="true">⌘</span>
          )}
          <span className="viz-thumb__label">{it.label}</span>
        </button>
      ))}
    </div>
  );
}

// --- Active viewport ---------------------------------------------------

function ActiveViewport({ ctrl }: { ctrl: SceneVizController }) {
  if (ctrl.activeKind === BANDS_KIND) {
    return <ActiveBandViewport ctrl={ctrl} />;
  }
  // Static-PNG mode (color / nir / swir / ndvi / band_mosaic).
  const item = ctrl.list?.items.find((i) => i.kind === ctrl.activeKind);
  if (!item) return null;

  const supportsPicker = ctrl.isHyperspectral && item.kind === "color";

  return (
    <PanzoomViewport
      imageSrc={`/api${item.image_url}`}
      imageAlt={item.label}
      pickable={supportsPicker}
      pickedFrac={ctrl.pickedFrac}
      onPick={(fracX, fracY) => {
        ctrl.setPickedFrac({ x: fracX, y: fracY });
        ctrl.fetchSpectrum(fracX, fracY);
      }}
      hint={
        supportsPicker
          ? "Scroll to zoom · drag to pan · click a pixel for its spectrum."
          : "Scroll to zoom · drag to pan."
      }
      overlays={overlayLayersFor(ctrl)}
    />
  );
}

function overlayLayersFor(ctrl: SceneVizController): PanzoomOverlay[] {
  return ctrl.overlays
    .filter((o) => o.visible && o.annotation.has_overlay)
    .map((o) => {
      let src = annotationOverlayUrl(ctrl.sceneId, o.annotation.id);
      // Non-default radius → query param triggers on-demand re-render.
      // Stable URL per radius means the browser cache keeps each one.
      if (o.radius !== null) {
        src += `?radius=${encodeURIComponent(o.radius)}`;
      }
      return {
        id: `${o.annotation.id}@r${o.radius ?? "default"}`,
        src,
        opacity: o.opacity,
        label: o.annotation.name,
      };
    });
}

function ActiveBandViewport({ ctrl }: { ctrl: SceneVizController }) {
  return (
    <div className="viz-bands">
      <BandCarousel ctrl={ctrl} />
      {ctrl.selectedBandIndex === null ? (
        <div className="viz-panel viz-empty">
          <p className="scene-detail__hint">
            Pick a band from the carousel above to render it.
          </p>
        </div>
      ) : (
        <PanzoomViewport
          imageSrc={`/api/scenes/${encodeURIComponent(ctrl.sceneId)}/bands/${ctrl.selectedBandIndex}/image`}
          imageAlt={`Band ${ctrl.selectedBandIndex}`}
          pickable={false}
          pickedFrac={null}
          onPick={() => undefined}
          hint="Inferno-mapped reflectance · scroll to zoom · drag to pan."
          overlays={overlayLayersFor(ctrl)}
        />
      )}
    </div>
  );
}

// --- Reusable panzoom viewport ----------------------------------------

interface PanzoomOverlay {
  id: string;             // stable key; usually the annotation id
  src: string;            // RGBA PNG url
  opacity: number;        // 0–1
  label?: string;         // for accessibility / tooltip
}

interface PanzoomViewportProps {
  imageSrc: string;
  imageAlt: string;
  pickable: boolean;
  pickedFrac: { x: number; y: number } | null;
  onPick: (fracX: number, fracY: number) => void;
  hint: string;
  /** Layers stacked on top of the base image, sharing its panzoom
   *  transform. Sized to fill the image element so the natural
   *  coordinates line up automatically. */
  overlays?: PanzoomOverlay[];
}

function PanzoomViewport({
  imageSrc,
  imageAlt,
  pickable,
  pickedFrac,
  onPick,
  hint,
  overlays = [],
}: PanzoomViewportProps) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [zoomToken, setZoomToken] = useState(0);

  useEffect(() => {
    if (!wrapRef.current) return;
    let cancelled = false;
    let dispose: (() => void) | null = null;
    (async () => {
      const { default: panzoom } = await import("panzoom");
      if (cancelled || !wrapRef.current) return;
      const inst = panzoom(wrapRef.current, {
        maxZoom: 16,
        minZoom: 0.5,
        bounds: true,
        boundsPadding: 0.2,
        smoothScroll: false,
        zoomDoubleClickSpeed: 1,
      });
      dispose = () => inst.dispose();
    })();
    return () => {
      cancelled = true;
      dispose?.();
    };
  }, [imageSrc, zoomToken]);

  const onClick = (e: MouseEvent<HTMLImageElement>) => {
    if (!pickable || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    if (rect.width === 0) return;
    const fracX = (e.clientX - rect.left) / rect.width;
    const fracY = (e.clientY - rect.top) / rect.height;
    onPick(fracX, fracY);
  };

  return (
    <div className="viz-active">
      <div className="viz-active__viewport">
        <div ref={wrapRef} className="viz-active__panzoom">
          {/*
            Inner stage sized to the image. Overlays and the pick
            marker are positioned relative to this — NOT the outer
            panzoom wrapper, which fills the viewport (so wheel events
            anywhere over the viewport are captured by panzoom even
            for narrow scenes like AVIRIS-NG). Without this inner div
            the overlay/marker coords would resolve against the empty
            padding around a narrow image. */}
          <div className="viz-active__stage">
            <img
              ref={imgRef}
              src={imageSrc}
              alt={imageAlt}
              className={
                "viz-active__image" + (pickable ? " viz-active__image--pickable" : "")
              }
              onClick={onClick}
              draggable={false}
            />
            {/*
              Overlay layers — each fills the stage, so they share
              natural coordinates with the base image and ride the
              panzoom transform. RGBA PNGs from the api have
              transparent backgrounds so only the masked pixels
              colour through. */}
            {overlays.map((o) => (
              <img
                key={o.id}
                src={o.src}
                alt={o.label ?? "annotation overlay"}
                className="viz-active__overlay"
                style={{ opacity: o.opacity }}
                draggable={false}
                aria-hidden="true"
              />
            ))}
            {pickedFrac && (
              <span
                className="viz-active__pick-marker"
                style={{
                  left: `${pickedFrac.x * 100}%`,
                  top: `${pickedFrac.y * 100}%`,
                }}
                aria-hidden="true"
              />
            )}
          </div>
        </div>
        <button
          type="button"
          className="viz-active__reset"
          onClick={() => setZoomToken((t) => t + 1)}
          title="Reset zoom"
        >
          ⟲ Reset
        </button>
      </div>
      <div className="viz-active__hint">{hint}</div>
    </div>
  );
}

// ---------------------------------------------------------------------
// Right rail — histogram + (spectrum | band browser | nothing).
// ---------------------------------------------------------------------

export function SceneVizRail({ ctrl }: { ctrl: SceneVizController }) {
  return (
    <div className="viz-rail">
      <HistogramSection sceneId={ctrl.sceneId} />
      {ctrl.activeKind === BANDS_KIND && ctrl.isHyperspectral ? (
        <SelectedBandInfo ctrl={ctrl} />
      ) : (
        <SpectrumSection
          spectrum={ctrl.spectrum}
          loading={ctrl.spectrumLoading}
          error={ctrl.spectrumError}
          isHyperspectral={ctrl.isHyperspectral}
          activeKind={ctrl.activeKind}
        />
      )}
    </div>
  );
}

// --- Histogram --------------------------------------------------------

function HistogramSection({ sceneId }: { sceneId: string }) {
  const [hist, setHist] = useState<HistogramJson | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHistogram(sceneId)
      .then((h) => {
        if (cancelled) return;
        setHist(h);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? (err.detail ?? `HTTP ${err.status}`) : "fetch failed");
      });
    return () => {
      cancelled = true;
    };
  }, [sceneId]);

  return (
    <section className="panel viz-panel">
      <h3 className="panel__heading">Distribution</h3>
      {error && <p className="form__error" role="alert">{error}</p>}
      {!error && !hist && <p className="scene-detail__hint">Loading…</p>}
      {hist && <HistogramChart hist={hist} />}
      {hist && hist.stats && Object.keys(hist.stats).length > 0 && (
        <dl className="viz-stats">
          <Stat label="Mean" v={hist.stats.mean} />
          <Stat label="Std" v={hist.stats.std} />
          <Stat label="Min" v={hist.stats.min} />
          <Stat label="Max" v={hist.stats.max} />
          <Stat
            label="p2 / p50 / p98"
            v={`${hist.stats.p2.toFixed(2)} · ${hist.stats.p50.toFixed(2)} · ${hist.stats.p98.toFixed(2)}`}
          />
        </dl>
      )}
    </section>
  );
}

function Stat({ label, v }: { label: string; v: number | string }) {
  const text =
    typeof v === "number" ? (Number.isFinite(v) ? v.toFixed(3) : "—") : v;
  return (
    <div className="viz-stat">
      <dt>{label}</dt>
      <dd className="mono">{text}</dd>
    </div>
  );
}

function HistogramChart({ hist }: { hist: HistogramJson }) {
  const ref = useRef<HTMLDivElement | null>(null);

  const data = useMemo(() => {
    if (hist.bins.length < 2 || hist.counts.length === 0) {
      return null;
    }
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < hist.counts.length; i++) {
      xs.push((hist.bins[i] + hist.bins[i + 1]) / 2);
      ys.push(hist.counts[i]);
    }
    return [xs, ys] as [number[], number[]];
  }, [hist]);

  useEffect(() => {
    if (!ref.current || !data) return;
    const container = ref.current;
    let dispose: (() => void) | null = null;

    (async () => {
      const [{ default: uPlot }] = await Promise.all([
        import("uplot"),
        import("uplot/dist/uPlot.min.css"),
      ]);
      const width = container.clientWidth || 280;
      const height = 160;
      const opts: object = {
        width,
        height,
        scales: { x: { time: false }, y: { auto: true } },
        legend: { show: false },
        cursor: { drag: { x: true, y: false }, points: { show: true } },
        series: [
          {},
          {
            label: "count",
            stroke: "#1f5f3d",
            fill: "rgba(31, 95, 61, 0.25)",
            paths: (uPlot as unknown as { paths: { bars: (cfg: object) => unknown } }).paths.bars({
              size: [0.95, 100],
              align: 0,
            }),
          },
        ],
        axes: [{ stroke: "#888" }, { stroke: "#888" }],
      };
      const inst = new (uPlot as unknown as { new (o: object, d: unknown[], el: HTMLElement): { destroy(): void } })(opts, data, container);
      dispose = () => inst.destroy();
    })();

    return () => {
      dispose?.();
      while (container.firstChild) container.removeChild(container.firstChild);
    };
  }, [data]);

  if (!data) return <p className="scene-detail__hint">No data.</p>;
  return <div ref={ref} className="viz-chart" />;
}

// --- Spectrum ---------------------------------------------------------

interface SpectrumProps {
  spectrum: SpectrumResponse | null;
  loading: boolean;
  error: string | null;
  isHyperspectral: boolean;
  activeKind: string | null;
}

function SpectrumSection({ spectrum, loading, error, isHyperspectral, activeKind }: SpectrumProps) {
  if (!isHyperspectral) {
    return null;
  }
  return (
    <section className="panel viz-panel">
      <h3 className="panel__heading">
        Spectral signature
        {spectrum && (
          <span className="panel__heading-sub mono">
            {" "}row {spectrum.row}, col {spectrum.col}
          </span>
        )}
      </h3>
      {!spectrum && !loading && !error && activeKind === "color" && (
        <p className="scene-detail__hint">
          Click a pixel on the colour image to read its reflectance spectrum here.
        </p>
      )}
      {!spectrum && !loading && !error && activeKind !== "color" && (
        <p className="scene-detail__hint">
          Switch to the colour image and click a pixel to read its spectrum.
        </p>
      )}
      {loading && <p className="scene-detail__hint">Loading spectrum…</p>}
      {error && <p className="form__error" role="alert">{error}</p>}
      {!loading && !error && spectrum && <SpectrumChart spectrum={spectrum} />}
    </section>
  );
}

function SpectrumChart({ spectrum }: { spectrum: SpectrumResponse }) {
  const ref = useRef<HTMLDivElement | null>(null);

  const data = useMemo(() => {
    const sorted = [...spectrum.points].sort(
      (a, b) => a.wavelength_nm - b.wavelength_nm,
    );
    const xs = sorted.map((p) => p.wavelength_nm);
    // Hide invalid bands (bbl=0 atmospheric water-vapor regions,
    // edge-trimmed bands) and ignore-value sentinels (-9999 on
    // AVIRIS-NG). uPlot treats null as a gap so the curve breaks
    // cleanly across the absorption windows instead of spiking.
    const ys = sorted.map((p) => {
      if (!p.is_valid) return null;
      if (!Number.isFinite(p.reflectance)) return null;
      if (p.reflectance <= -1000) return null;
      return p.reflectance;
    });
    return [xs, ys] as [number[], (number | null)[]];
  }, [spectrum]);

  useEffect(() => {
    if (!ref.current || !data) return;
    const container = ref.current;
    let dispose: (() => void) | null = null;

    (async () => {
      const [{ default: uPlot }] = await Promise.all([
        import("uplot"),
        import("uplot/dist/uPlot.min.css"),
      ]);
      const width = container.clientWidth || 280;
      const height = 200;
      const opts = {
        width,
        height,
        scales: { x: { time: false }, y: { auto: true } },
        legend: { show: false },
        cursor: { points: { show: true } },
        series: [
          { label: "λ (nm)" },
          {
            label: "reflectance",
            stroke: "#1f5f3d",
            width: 1.5,
            points: { show: false },
            spanGaps: false,
          },
        ],
        axes: [
          { stroke: "#888", label: "λ (nm)" },
          { stroke: "#888", label: "ρ" },
        ],
      };
      const inst = new (uPlot as unknown as { new (o: object, d: unknown[], el: HTMLElement): { destroy(): void } })(opts, data, container);
      dispose = () => inst.destroy();
    })();

    return () => {
      dispose?.();
      while (container.firstChild) container.removeChild(container.firstChild);
    };
  }, [data]);

  return <div ref={ref} className="viz-chart" />;
}

// --- Band browser -----------------------------------------------------

interface BandInfo {
  index: number;
  wavelength_nm: number;
  spectral_family: string | null;
  is_valid: boolean;
}

interface BandList {
  sensor_type: string;
  band_count: number;
  bands: BandInfo[];
}

// =====================================================================
// Band data fetched once and shared between center carousel + rail info.
// Cached on the controller via a tiny module-local map keyed by scene id
// so we don't refetch when the user toggles between modes.
// =====================================================================

const _bandListCache = new Map<string, BandList>();

function useBandList(sceneId: string): {
  list: BandList | null;
  error: string | null;
} {
  const cached = _bandListCache.get(sceneId) ?? null;
  const [list, setList] = useState<BandList | null>(cached);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (_bandListCache.has(sceneId)) {
      setList(_bandListCache.get(sceneId) ?? null);
      return;
    }
    let cancelled = false;
    setError(null);
    fetch(`/api/scenes/${encodeURIComponent(sceneId)}/bands`, {
      credentials: "include",
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const json = (await r.json()) as BandList;
        if (cancelled) return;
        _bandListCache.set(sceneId, json);
        setList(json);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "fetch failed");
      });
    return () => {
      cancelled = true;
    };
  }, [sceneId]);

  return { list, error };
}

// =====================================================================
// Center-column carousel — horizontal scrollable card row + search.
// =====================================================================

function BandCarousel({ ctrl }: { ctrl: SceneVizController }) {
  const { list, error } = useBandList(ctrl.sceneId);
  const [familyFilter, setFamilyFilter] = useState<"all" | "VNIR" | "SWIR">("all");
  const [search, setSearch] = useState<string>("");
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  // Auto-pick a default band the first time the carousel loads.
  useEffect(() => {
    if (list && ctrl.selectedBandIndex === null && list.bands.length > 0) {
      // Prefer a visible-light band if there's one near 660 nm; falls
      // back to the middle of the list. Better default than band 0
      // (which on PRISMA is a SWIR2 band, less intuitive at first sight).
      let bestIdx = list.bands[Math.floor(list.bands.length / 2)].index;
      let bestDist = Infinity;
      for (const b of list.bands) {
        const d = Math.abs(b.wavelength_nm - 660);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = b.index;
        }
      }
      ctrl.setSelectedBandIndex(bestIdx);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list]);

  // Smooth-scroll the active band's card into view when it changes.
  useEffect(() => {
    if (ctrl.selectedBandIndex === null) return;
    const el = scrollerRef.current?.querySelector<HTMLElement>(
      `[data-band-index="${ctrl.selectedBandIndex}"]`,
    );
    if (el) {
      el.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    }
  }, [ctrl.selectedBandIndex]);

  if (error) {
    return (
      <div className="viz-panel">
        <p className="form__error" role="alert">{error}</p>
      </div>
    );
  }
  if (!list) {
    return (
      <div className="viz-panel">
        <p className="scene-detail__hint">Loading bands…</p>
      </div>
    );
  }

  // Filter: family + free-text. Search matches: band index (`100`,
  // `#100`), exact wavelength integer (`660`), or "near" patterns
  // (anything that parses as a number is treated as a wavelength
  // target and we keep bands within ±20 nm).
  const q = search.trim();
  const familyFiltered =
    familyFilter === "all"
      ? list.bands
      : list.bands.filter((b) => b.spectral_family === familyFilter);

  let filtered = familyFiltered;
  if (q) {
    const num = Number(q.replace(/^#/, ""));
    if (Number.isFinite(num)) {
      // Numeric query — match by index OR by wavelength within 20 nm.
      filtered = familyFiltered.filter(
        (b) =>
          b.index === num ||
          Math.abs(b.wavelength_nm - num) <= 20,
      );
    } else {
      const ql = q.toLowerCase();
      filtered = familyFiltered.filter((b) =>
        (b.spectral_family ?? "").toLowerCase().includes(ql),
      );
    }
  }

  return (
    <div className="band-carousel">
      <div className="band-carousel__controls">
        <input
          type="search"
          className="band-carousel__search"
          placeholder="Search by index or λ (nm)…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search bands"
        />
        <div className="band-filter">
          {(["all", "VNIR", "SWIR"] as const).map((f) => (
            <button
              key={f}
              type="button"
              className={"band-filter__btn" + (familyFilter === f ? " is-active" : "")}
              onClick={() => setFamilyFilter(f)}
            >
              {f === "all" ? "All" : f}
            </button>
          ))}
        </div>
        <span className="band-carousel__count">
          {filtered.length} / {list.band_count}
        </span>
      </div>

      {filtered.length === 0 ? (
        <p className="band-carousel__empty scene-detail__hint">
          No bands match.
        </p>
      ) : (
        <div ref={scrollerRef} className="band-carousel__strip" role="listbox">
          {filtered.map((b) => (
            <button
              key={b.index}
              type="button"
              role="option"
              aria-selected={b.index === ctrl.selectedBandIndex}
              data-band-index={b.index}
              className={
                "band-card" +
                (b.index === ctrl.selectedBandIndex ? " is-active" : "") +
                (b.is_valid ? "" : " is-invalid")
              }
              onClick={() => ctrl.setSelectedBandIndex(b.index)}
              title={
                `Band ${b.index} · ${b.wavelength_nm.toFixed(1)} nm` +
                (b.spectral_family ? ` · ${b.spectral_family}` : "") +
                (b.is_valid ? "" : " · invalid")
              }
            >
              <span className="band-card__index mono">#{b.index}</span>
              <span className="band-card__wl mono">
                {b.wavelength_nm.toFixed(0)}
                <span className="band-card__wl-unit"> nm</span>
              </span>
              {b.spectral_family && (
                <span
                  className={"band-card__family band-card__family--" + b.spectral_family}
                >
                  {b.spectral_family}
                </span>
              )}
              {!b.is_valid && (
                <span className="band-card__invalid" aria-label="invalid">⚠</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// =====================================================================
// Right rail — show selected band's metadata (replaces spectrum chart
// while Bands mode is active).
// =====================================================================

function SelectedBandInfo({ ctrl }: { ctrl: SceneVizController }) {
  const { list } = useBandList(ctrl.sceneId);
  const idx = ctrl.selectedBandIndex;
  const band = list?.bands.find((b) => b.index === idx) ?? null;

  return (
    <section className="panel viz-panel">
      <h3 className="panel__heading">Selected band</h3>
      {!band ? (
        <p className="scene-detail__hint">Pick a band from the carousel.</p>
      ) : (
        <dl className="scene-detail__meta-list">
          <div className="scene-detail__meta-row">
            <dt>Index</dt>
            <dd className="mono">#{band.index}</dd>
          </div>
          <div className="scene-detail__meta-row">
            <dt>Wavelength</dt>
            <dd className="mono">{band.wavelength_nm.toFixed(2)} nm</dd>
          </div>
          {band.spectral_family && (
            <div className="scene-detail__meta-row">
              <dt>Family</dt>
              <dd>
                <span
                  className={"band-card__family band-card__family--" + band.spectral_family}
                >
                  {band.spectral_family}
                </span>
              </dd>
            </div>
          )}
          <div className="scene-detail__meta-row">
            <dt>Validity</dt>
            <dd>
              {band.is_valid ? (
                <span style={{ color: "var(--ok)" }}>valid</span>
              ) : (
                <span style={{ color: "var(--warn)" }}>marked invalid</span>
              )}
            </dd>
          </div>
        </dl>
      )}
    </section>
  );
}
