// Model detail — hero header + flow chart + metadata rail.
//
// Pulls the manifest from /api/models/:architecture and pairs it with
// the matching hand-authored React Flow definition under
// frontend/src/data/modelFlows/<arch>.ts.
//
// Sequence diagram: final design/diagrams/models-detail.drawio

import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { getModel } from "../api/models";
import {
  MODEL_FLOW_NODE_TYPES,
  STAGE_COLORS,
} from "../components/ModelFlowNode";
import { MODEL_FLOWS } from "../data/modelFlows";
import type {
  ModelFlowEdge,
  ModelFlowGroupData,
  ModelFlowNode,
  ModelFlowNodeData,
  StageKind,
} from "../data/modelFlows/types";
import { layoutWithElk } from "../lib/elkLayout";
import type { ModelDetail } from "../types";

const SENSOR_LABEL: Record<string, string> = {
  thermal: "Thermal · Landsat 9",
  hyperspectral: "Hyperspectral · PRISMA / EnMAP",
};

const LEGEND_KINDS: { kind: StageKind; label: string }[] = [
  { kind: "input", label: "Input" },
  { kind: "normalize", label: "Normalize" },
  { kind: "encoder", label: "Encoder" },
  { kind: "bottleneck", label: "Bottleneck" },
  { kind: "decoder", label: "Decoder" },
  { kind: "denormalize", label: "Denormalize" },
  { kind: "output", label: "Output" },
  { kind: "mask", label: "Mask" },
  { kind: "loss", label: "Loss" },
  { kind: "special", label: "Special" },
];

function formatParams(p: number): string {
  if (p >= 1_000_000) return `${(p / 1_000_000).toFixed(2)}M`;
  if (p >= 1_000) return `${(p / 1_000).toFixed(0)}K`;
  return String(p);
}

function StageLegend() {
  return (
    <div className="model-detail__legend">
      {LEGEND_KINDS.map(({ kind, label }) => {
        const c = STAGE_COLORS[kind];
        return (
          <span key={kind} className="model-detail__legend-item">
            <span
              className="model-detail__legend-swatch"
              style={{ background: c.bg, borderColor: c.border }}
            />
            {label}
          </span>
        );
      })}
    </div>
  );
}

function MetadataRail({ model }: { model: ModelDetail }) {
  const norm = model.normalization;
  const baked = norm.mode === "baked_in";
  return (
    <aside className="model-detail__rail">
      <section>
        <h3>Sensor</h3>
        <p>{SENSOR_LABEL[model.sensor] ?? model.sensor}</p>
      </section>
      <section>
        <h3>Current checkpoint</h3>
        <dl>
          <dt>file</dt>
          <dd className="mono">{model.current.file}</dd>
          <dt>version</dt>
          <dd>{model.current.version}</dd>
          <dt>epoch</dt>
          <dd>{model.current.epoch}</dd>
          <dt>params</dt>
          <dd>{formatParams(model.current.params)}</dd>
          <dt>val loss</dt>
          <dd>{model.current.val_loss.toFixed(4)}</dd>
          {model.current.encoder_dims && (
            <>
              <dt>encoder dims</dt>
              <dd className="mono">{model.current.encoder_dims}</dd>
            </>
          )}
          {model.current.spectral_dim_D != null && (
            <>
              <dt>spectral dim D</dt>
              <dd>{model.current.spectral_dim_D}</dd>
            </>
          )}
        </dl>
      </section>
      <section>
        <h3>Normalization</h3>
        <p className="model-detail__norm">
          <strong>{baked ? "baked into forward" : "none"}</strong>
        </p>
        {baked && (
          <dl>
            <dt>shape</dt>
            <dd className="mono">
              {norm.stats_shape ? `[${norm.stats_shape.join(", ")}]` : "—"}
            </dd>
            {typeof norm.mean === "number" && (
              <>
                <dt>μ</dt>
                <dd className="mono">{norm.mean.toFixed(4)}</dd>
                <dt>σ</dt>
                <dd className="mono">
                  {typeof norm.std === "number" ? norm.std.toFixed(4) : "—"}
                </dd>
              </>
            )}
          </dl>
        )}
        <p className="model-detail__norm-source">{norm.source}</p>
      </section>
      <section>
        <h3>Inferencer</h3>
        <p className="mono">{model.inferencer}</p>
        <p className="muted mono small">{model.inferencer_module}</p>
      </section>
      {model.alternatives.length > 0 && (
        <section>
          <h3>Alternatives ({model.alternatives.length})</h3>
          <ul className="model-detail__alts">
            {model.alternatives.map((alt, i) => {
              const a = alt as Record<string, unknown>;
              return (
                <li key={String(a.file ?? i)}>
                  <span className="mono">{String(a.file ?? "—")}</span>
                  <br />
                  <span className="muted small">
                    val loss {String(a.val_loss ?? "—")} · {String(a.note ?? "")}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      )}
      <section>
        <h3>Notes</h3>
        <p className="small">{model.notes}</p>
      </section>
      <section>
        <h3>Documentation</h3>
        <p className="mono small">{model.doc}</p>
      </section>
    </aside>
  );
}

export function ModelDetailPage() {
  const { architecture } = useParams<{ architecture: string }>();
  const [model, setModel] = useState<ModelDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!architecture) return;
    let cancelled = false;
    setModel(null);
    setError(null);
    getModel(architecture)
      .then((m) => {
        if (!cancelled) setModel(m);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setError(
            err.status === 404
              ? `No model named "${architecture}".`
              : err.detail ?? `Error: HTTP ${err.status}`,
          );
        } else {
          setError("Could not reach the server.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [architecture]);

  if (!architecture) return null;
  const flow = MODEL_FLOWS[architecture];

  return <ModelDetailInner model={model} flow={flow} error={error} />;
}

interface InnerProps {
  model: ModelDetail | null;
  flow: (typeof MODEL_FLOWS)[string] | undefined;
  error: string | null;
}

function ModelDetailInner({ model, flow, error }: InnerProps) {
  // View-mode toggles (independent so users can mix them):
  //   layoutMode: hand-authored coords vs ELK auto-layout
  //   collapsed:  which group containers are collapsed to a single card
  //   edgeShapes: render shape labels on edges (read from the SOURCE
  //               node's `shape` so the edge tells you what's flowing)
  const [layoutMode, setLayoutMode] = useState<"hand" | "elk">("hand");
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [edgeShapes, setEdgeShapes] = useState(true);

  const groupNodes = useMemo<ModelFlowNode[]>(
    () => (flow?.nodes ?? []).filter((n) => n.type === "modelGroup"),
    [flow],
  );

  // Pre-compute the "rendered" graph given current toggles, before ELK.
  const rawGraph = useMemo<{
    nodes: ModelFlowNode[];
    edges: ModelFlowEdge[];
  } | null>(() => {
    if (!flow) return null;
    const nodes = flow.nodes;
    const edges = flow.edges;

    // Build a lookup of stage shape strings so we can annotate edges.
    const shapeBySource = new Map<string, string | undefined>();
    for (const n of nodes) {
      if (n.type === "modelStage") {
        const d = n.data as ModelFlowNodeData;
        shapeBySource.set(n.id, d.shape);
      }
    }

    // Annotate edges with the SOURCE node's shape — the edge represents
    // what's flowing out of it. Don't clobber an explicit label.
    const annotatedEdges: ModelFlowEdge[] = edges.map((e) => {
      if (!edgeShapes) return { ...e, label: undefined };
      if (e.label) return e;
      const s = shapeBySource.get(e.source);
      return s ? { ...e, label: s } : e;
    });

    // Collapse: for any group in `collapsed`, hide its children and
    // replace the group itself with a single stage card carrying its
    // title + subtitle. Re-wire edges that touched a hidden child to
    // the group's collapsed-card id so the diagram stays connected.
    if (collapsed.size === 0) {
      return { nodes, edges: annotatedEdges };
    }

    const hiddenIds = new Set<string>();
    const groupReplacement = new Map<string, ModelFlowNode>();
    for (const groupId of collapsed) {
      const g = nodes.find((n) => n.id === groupId);
      if (!g || g.type !== "modelGroup") continue;
      const gd = g.data as ModelFlowGroupData;
      // Replacement stage card sits at the group's current top-left,
      // sized like a normal card. ELK will reposition if active.
      const replacement: ModelFlowNode = {
        id: groupId,
        type: "modelStage",
        position: { x: g.position.x, y: g.position.y },
        data: {
          kind: "encoder", // tinted by tone where possible
          label: gd.title,
          shape: gd.subtitle,
          detail: "collapsed — toggle to expand",
        } satisfies ModelFlowNodeData,
      };
      groupReplacement.set(groupId, replacement);
      for (const n of nodes) {
        const parentId = (n as ModelFlowNode & { parentId?: string })
          .parentId;
        if (parentId === groupId) hiddenIds.add(n.id);
      }
    }

    const visibleNodes: ModelFlowNode[] = [];
    for (const n of nodes) {
      if (hiddenIds.has(n.id)) continue;
      const replacement = groupReplacement.get(n.id);
      visibleNodes.push(replacement ?? n);
    }
    // Rewire edges: a hidden child becomes its parent group's id.
    const childToGroup = new Map<string, string>();
    for (const n of nodes) {
      const parentId = (n as ModelFlowNode & { parentId?: string }).parentId;
      if (parentId && collapsed.has(parentId)) {
        childToGroup.set(n.id, parentId);
      }
    }
    const seenEdges = new Set<string>();
    const rewiredEdges: ModelFlowEdge[] = [];
    for (const e of annotatedEdges) {
      const src = childToGroup.get(e.source) ?? e.source;
      const tgt = childToGroup.get(e.target) ?? e.target;
      if (src === tgt) continue; // collapsed-into-self
      const key = `${src}->${tgt}`;
      if (seenEdges.has(key)) continue;
      seenEdges.add(key);
      rewiredEdges.push({ ...e, source: src, target: tgt });
    }
    return { nodes: visibleNodes, edges: rewiredEdges };
  }, [flow, collapsed, edgeShapes]);

  const [laidOut, setLaidOut] = useState<
    { nodes: ModelFlowNode[]; edges: ModelFlowEdge[] } | null
  >(null);

  // Run ELK whenever the layout mode is "elk" and the raw graph changes.
  // In "hand" mode we just pass the hand-authored coords through.
  useEffect(() => {
    if (!rawGraph) return;
    if (layoutMode === "hand") {
      setLaidOut(rawGraph);
      return;
    }
    let cancelled = false;
    layoutWithElk(rawGraph.nodes, rawGraph.edges).then((positioned) => {
      if (!cancelled) setLaidOut({ nodes: positioned, edges: rawGraph.edges });
    });
    return () => {
      cancelled = true;
    };
  }, [rawGraph, layoutMode]);

  const toggleCollapsed = (groupId: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  };

  return (
    <div className="page model-detail">
      <Link to="/models" className="model-detail__back">
        ← All models
      </Link>

      {error && <div className="page__error">{error}</div>}
      {!model && !error && (
        <div className="page__empty">Loading model…</div>
      )}

      {model && (
        <>
          <header className="model-detail__hero">
            <div className="model-detail__hero-left">
              <div className="model-detail__codename">
                <span className="model-detail__codename-name">
                  {model.codename.name}
                </span>
                <span className="model-detail__codename-script">
                  {model.codename.script}
                </span>
              </div>
              <div className="model-detail__codename-meaning">
                <em>{model.codename.meaning}</em>
              </div>
              <h1 className="model-detail__title">{model.label}</h1>
              <p className="model-detail__why">{model.codename.why}</p>
              {flow && (
                <p className="model-detail__desc">{flow.description}</p>
              )}
              <div className="model-detail__architecture-slug mono small">
                {model.architecture}
              </div>
            </div>
          </header>

          <div className="model-detail__body">
            <main className="model-detail__main">
              {flow ? (
                <>
                  <StageLegend />
                  <div className="model-detail__viz-toolbar">
                    <span className="form__label form__label--inline small">
                      Layout
                      <select
                        className="form__select"
                        value={layoutMode}
                        onChange={(e) =>
                          setLayoutMode(e.target.value as "hand" | "elk")
                        }
                      >
                        <option value="hand">Hand-authored</option>
                        <option value="elk">Auto (ELK)</option>
                      </select>
                    </span>
                    <label className="form__label form__label--inline small">
                      <input
                        type="checkbox"
                        checked={edgeShapes}
                        onChange={(e) => setEdgeShapes(e.target.checked)}
                      />{" "}
                      Show tensor shapes on edges
                    </label>
                    {groupNodes.length > 0 && (
                      <span className="model-detail__viz-collapse small">
                        Collapse:
                        {groupNodes.map((g) => {
                          const gd = g.data as ModelFlowGroupData;
                          const on = collapsed.has(g.id);
                          return (
                            <button
                              key={g.id}
                              type="button"
                              className="anomaly-viewer__tool-btn"
                              data-active={on ? "true" : "false"}
                              onClick={() => toggleCollapsed(g.id)}
                              title={gd.title}
                            >
                              {gd.title.split("·")[0].trim()}
                            </button>
                          );
                        })}
                      </span>
                    )}
                  </div>
                  <div className="model-detail__canvas">
                    <ReactFlow
                      nodes={laidOut?.nodes ?? flow.nodes}
                      edges={laidOut?.edges ?? flow.edges}
                      nodeTypes={MODEL_FLOW_NODE_TYPES}
                      fitView
                      fitViewOptions={{ padding: 0.2 }}
                      nodesDraggable={false}
                      nodesConnectable={false}
                      elementsSelectable={false}
                      proOptions={{ hideAttribution: true }}
                      defaultEdgeOptions={{
                        style: { stroke: "#94a3b8", strokeWidth: 1.6 },
                        labelStyle: {
                          fontFamily:
                            "ui-monospace, SFMono-Regular, Menlo, monospace",
                          fontSize: 10,
                          fill: "#475569",
                        },
                        labelBgStyle: { fill: "#f8fafc", opacity: 0.92 },
                        labelBgPadding: [4, 2] as [number, number],
                        labelBgBorderRadius: 3,
                      }}
                    >
                      <Background gap={18} size={1} color="#cbd5e1" />
                      <MiniMap
                        pannable
                        zoomable
                        nodeColor={(n) => {
                          const data = n.data as { kind?: StageKind };
                          if (data?.kind && STAGE_COLORS[data.kind]) {
                            return STAGE_COLORS[data.kind].border;
                          }
                          return "#64748b";
                        }}
                        maskColor="rgba(241, 245, 249, 0.65)"
                        style={{ background: "#f8fafc", border: "1px solid #cbd5e1" }}
                      />
                      <Controls showInteractive={false} />
                    </ReactFlow>
                  </div>
                </>
              ) : model?.family === "classical" ? (
                <div className="page__empty">
                  <p>
                    <strong>Closed-form statistical detector.</strong>{" "}
                    No architecture diagram applies — the math is the
                    diagram. See the description above for the method
                    summary, and check the source under{" "}
                    <code>app/detectors/{model.architecture}_detector.py</code>{" "}
                    if you want the line-by-line implementation.
                  </p>
                </div>
              ) : (
                <div className="page__empty">
                  No flow chart authored for this architecture yet.
                </div>
              )}
            </main>
            <MetadataRail model={model} />
          </div>
        </>
      )}
    </div>
  );
}
