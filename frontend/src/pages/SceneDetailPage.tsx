// Scene Detail page — three-column command-center layout.
//
// Route: /scenes/<scene_<uuid>>.
//
//   ┌──────────────────────────────────────────────────────────┐
//   │ header  ← Library                              [Create]  │
//   ├────────────┬─────────────────────────────┬───────────────┤
//   │ metadata   │ filmstrip                   │ histogram     │
//   │ rail       ├─────────────────────────────┤ + spectrum or │
//   │ (compact   │ active viewport (panzoom)   │   band browser│
//   │  panels)   │                             │               │
//   └────────────┴─────────────────────────────┴───────────────┘
//
// Sequence diagram for the read paths: scene-detail.drawio (Step 8c)
// + scene-visualizations.drawio (Step 8.5d).

import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { deleteScene, getScene, parseSceneInUse } from "../api/scenes";
import { AnnotationsPanel } from "../components/AnnotationsPanel";
import { NewProjectDialog } from "../components/NewProjectDialog";
import {
  SceneVizCenter,
  SceneVizRail,
  useSceneViz,
} from "../components/SceneVisualizations";
import { useToast } from "../components/Toast";
import type { Scene } from "../types";

const SENSOR_LABELS: Record<string, string> = {
  prisma: "PRISMA",
  landsat9: "Landsat 9",
  enmap: "EnMAP",
  aviris_ng: "AVIRIS-NG",
  hotsat1: "HotSAT-1",
};

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatPercent(v: number | null): string {
  if (v === null || v === undefined) return "—";
  return `${v.toFixed(2)}%`;
}

function formatBbox(s: Scene): string {
  const f = (n: number) => n.toFixed(4);
  return `[${f(s.bbox_min_lon)}, ${f(s.bbox_min_lat)}, ${f(s.bbox_max_lon)}, ${f(s.bbox_max_lat)}]`;
}

export function SceneDetailPage() {
  const { sceneId } = useParams<{ sceneId: string }>();

  const [scene, setScene] = useState<Scene | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sceneId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getScene(sceneId)
      .then((s) => {
        if (cancelled) return;
        setScene(s);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          if (err.status === 404) setError("Scene not found.");
          else if (err.status === 422) setError("Malformed scene id.");
          else setError(err.detail ?? `Error: HTTP ${err.status}`);
        } else {
          setError("Could not reach the server.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sceneId]);

  if (loading) {
    return (
      <div className="scene-detail">
        <p className="scene-detail__loading">Loading scene…</p>
      </div>
    );
  }

  if (error || !scene) {
    return (
      <div className="scene-detail">
        <header className="scene-detail__header">
          <Link to="/scenes" className="scene-detail__back">
            ← Library
          </Link>
        </header>
        <p className="form__error" role="alert">
          {error ?? "Scene unavailable."}
        </p>
      </div>
    );
  }

  return (
    <SceneDetailLayout scene={scene} />
  );
}

// =====================================================================
// Layout — pulled into its own component so the visualization hook
// (which depends on scene properties) only runs once we have a scene.
// =====================================================================

function SceneDetailLayout({ scene }: { scene: Scene }) {
  const navigate = useNavigate();
  const toast = useToast();
  const isHyperspectral =
    scene.sensor_type === "prisma" ||
    scene.sensor_type === "enmap" ||
    scene.sensor_type === "aviris_ng";
  const ctrl = useSceneViz(scene.id, isHyperspectral);
  const [newProjectOpen, setNewProjectOpen] = useState(false);

  return (
    <div className="scene-detail scene-detail--three-col">
      <header className="scene-detail__header">
        <div className="scene-detail__header-left">
          <Link to="/scenes" className="scene-detail__back">
            ← Library
          </Link>
          <span className="sensor-pill">
            {SENSOR_LABELS[scene.sensor_type] ?? scene.sensor_type}
          </span>
          <h2 className="scene-detail__name">{scene.name}</h2>
        </div>
        <div className="scene-detail__header-actions">
          <button
            type="button"
            className="form__submit scene-detail__cta"
            onClick={() => setNewProjectOpen(true)}
          >
            Create project
          </button>
          <button
            type="button"
            className="anomaly-viewer__tool-btn scene-detail__delete"
            onClick={async () => {
              if (
                !window.confirm(
                  `Delete scene "${scene.name}"? This removes its raw files, vendable, annotations, and any saved visualizations sourced from it.`,
                )
              ) {
                return;
              }
              try {
                await deleteScene(scene.id);
                navigate("/scenes");
              } catch (err) {
                if (err instanceof ApiError && err.status === 409) {
                  const detail = parseSceneInUse(err.detail);
                  if (detail) {
                    window.alert(
                      `Cannot delete: still referenced by ${
                        detail.blocking_project_ids.length
                      } project(s). Delete those first:\n${detail.blocking_project_ids.join(
                        ", ",
                      )}`,
                    );
                    return;
                  }
                }
                window.alert(
                  err instanceof ApiError
                    ? `Delete failed: ${err.detail ?? err.status}`
                    : "Delete failed.",
                );
              }
            }}
          >
            Delete scene
          </button>
        </div>
      </header>

      {newProjectOpen && (
        <NewProjectDialog
          fixedSceneId={scene.id}
          fixedSceneLabel={scene.name}
          onClose={() => setNewProjectOpen(false)}
          onCreated={(project) => {
            setNewProjectOpen(false);
            toast.push({
              title: "Project created",
              message: project.name,
              variant: "success",
              link: { to: "/projects", label: "View in Projects" },
              durationMs: 6000,
            });
            // Workspace lands in Step 11; for now drop the user on the
            // Projects landing where their new project is at the top.
            navigate("/projects");
          }}
        />
      )}

      <aside className="scene-detail__rail-left">
        <MetaPanel
          heading="Acquisition"
          rows={[
            ["Sensor scene id", scene.sensor_scene_id, "mono"],
            ["Acquired", formatDate(scene.acquisition_at)],
            ["Processing", scene.processing_level ?? "—"],
            ["Product", scene.product_type ?? "—"],
            ["Onboarded", formatDate(scene.created_at)],
          ]}
        />
        <MetaPanel
          heading="Spatial + bands"
          rows={[
            ["Bbox", formatBbox(scene), "mono"],
            ["Projection", scene.native_projection ?? "—", "mono"],
            ["Bands", String(scene.band_count)],
            ["Cloud cover", formatPercent(scene.cloud_cover_pct)],
            ["Valid pixels", formatPercent(scene.valid_pixel_pct)],
          ]}
        />
        <MetaPanel
          heading="Storage"
          rows={[
            ["Raw", scene.raw_path, "mono"],
            ["Vendable", scene.vendable_path, "mono"],
            ["Thumbnail", scene.thumbnail_path ?? "—", "mono"],
          ]}
        />
        <AnnotationsPanel ctrl={ctrl} />
      </aside>

      <main className="scene-detail__center">
        <SceneVizCenter ctrl={ctrl} />
      </main>

      <aside className="scene-detail__rail-right">
        <SceneVizRail ctrl={ctrl} />
      </aside>
    </div>
  );
}

// Compact metadata panel for the left rail. `rows` is a list of
// [label, value, optional class on dd].
type Row = [string, string] | [string, string, string];

function MetaPanel({ heading, rows }: { heading: string; rows: Row[] }) {
  return (
    <section className="panel viz-panel scene-detail__meta">
      <h3 className="panel__heading">{heading}</h3>
      <dl className="scene-detail__meta-list">
        {rows.map(([label, value, cls]) => (
          <div className="scene-detail__meta-row" key={label}>
            <dt>{label}</dt>
            <dd className={cls ?? ""}>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
