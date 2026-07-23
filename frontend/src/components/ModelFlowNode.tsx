// Custom React Flow node for the Models architecture diagrams.
//
// Each stage in a model flow is rendered as a card with a coloured
// stage badge (top), a bold heading, and an optional shape annotation
// + caption. The colour palette is shared with the StageKindLegend
// shown above the canvas.

import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";

import type { ModelFlowNodeData, StageKind } from "../data/modelFlows/types";

// Light palette tuned for a white / cream canvas. Each kind: a tinted
// pastel background, a saturated border, a darker badge label, and
// near-black body text. WCAG-AA contrast across the board.
export const STAGE_COLORS: Record<StageKind, { bg: string; border: string; badge: string; text: string }> = {
  input:       { bg: "#f1f5f9",  border: "#475569",  badge: "#475569",  text: "#0f172a" },
  normalize:   { bg: "#e0f2fe",  border: "#0284c7",  badge: "#0369a1",  text: "#0c4a6e" },
  encoder:     { bg: "#d1fae5",  border: "#10b981",  badge: "#047857",  text: "#064e3b" },
  bottleneck:  { bg: "#fce7f3",  border: "#db2777",  badge: "#9d174d",  text: "#831843" },
  decoder:     { bg: "#ffedd5",  border: "#f97316",  badge: "#c2410c",  text: "#7c2d12" },
  denormalize: { bg: "#e0f2fe",  border: "#0284c7",  badge: "#0369a1",  text: "#0c4a6e" },
  output:      { bg: "#f1f5f9",  border: "#475569",  badge: "#475569",  text: "#0f172a" },
  mask:        { bg: "#fef9c3",  border: "#eab308",  badge: "#a16207",  text: "#713f12" },
  loss:        { bg: "#fee2e2",  border: "#ef4444",  badge: "#b91c1c",  text: "#7f1d1d" },
  special:     { bg: "#fce7f3",  border: "#ec4899",  badge: "#be185d",  text: "#831843" },
};

const STAGE_LABEL: Record<StageKind, string> = {
  input: "INPUT",
  normalize: "NORMALIZE",
  encoder: "ENCODER",
  bottleneck: "BOTTLENECK",
  decoder: "DECODER",
  denormalize: "DENORMALIZE",
  output: "OUTPUT",
  mask: "MASK",
  loss: "LOSS",
  special: "SPECIAL",
};

export function ModelFlowNode({ data }: NodeProps) {
  const d = data as ModelFlowNodeData;
  const colors = STAGE_COLORS[d.kind];
  return (
    <div
      style={{
        background: colors.bg,
        border: `1.5px solid ${colors.border}`,
        borderRadius: 10,
        padding: "10px 14px",
        minWidth: 200,
        maxWidth: 260,
        color: colors.text,
        fontSize: 13,
        boxShadow: "0 2px 8px rgba(15, 23, 42, 0.08)",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: colors.border, width: 8, height: 8 }} />
      <div
        style={{
          fontSize: 9,
          letterSpacing: "0.08em",
          fontWeight: 700,
          color: colors.badge,
          marginBottom: 4,
        }}
      >
        {STAGE_LABEL[d.kind]}
      </div>
      <div style={{ fontWeight: 600, lineHeight: 1.25 }}>{d.label}</div>
      {d.shape && (
        <div
          style={{
            marginTop: 6,
            fontSize: 11,
            fontFamily:
              "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            color: colors.badge,
            opacity: 0.95,
          }}
        >
          {d.shape}
        </div>
      )}
      {d.detail && (
        <div
          style={{
            marginTop: 4,
            fontSize: 10.5,
            color: colors.text,
            opacity: 0.7,
            lineHeight: 1.35,
          }}
        >
          {d.detail}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ background: colors.border, width: 8, height: 8 }} />
    </div>
  );
}

import { ModelFlowGroup } from "./ModelFlowGroup";

export const MODEL_FLOW_NODE_TYPES = {
  modelStage: ModelFlowNode,
  modelGroup: ModelFlowGroup,
};
