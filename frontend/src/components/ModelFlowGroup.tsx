// Subgraph "group" node for the Models architecture diagrams.
//
// Renders a labelled, rounded translucent backdrop behind a cluster of
// child stage nodes (encoder / decoder / spectral I-O). React Flow
// children opt in via `parentId: "<group-id>"` and `extent: "parent"`.

import type { NodeProps } from "@xyflow/react";

export interface ModelFlowGroupData extends Record<string, unknown> {
  title: string;
  subtitle?: string;
  tone?: "encoder" | "decoder" | "spectral" | "neutral";
  width: number;
  height: number;
}

const TONES: Record<NonNullable<ModelFlowGroupData["tone"]>, { bg: string; border: string; label: string }> = {
  encoder:  { bg: "rgba(16, 185, 129, 0.06)", border: "rgba(16, 185, 129, 0.35)", label: "#34d399" },
  decoder:  { bg: "rgba(249, 115, 22, 0.06)", border: "rgba(249, 115, 22, 0.35)", label: "#fb923c" },
  spectral: { bg: "rgba(236, 72, 153, 0.06)", border: "rgba(236, 72, 153, 0.35)", label: "#f472b6" },
  neutral:  { bg: "rgba(148, 163, 184, 0.05)", border: "rgba(148, 163, 184, 0.3)", label: "#94a3b8" },
};

export function ModelFlowGroup({ data }: NodeProps) {
  const d = data as ModelFlowGroupData;
  const tone = TONES[d.tone ?? "neutral"];
  return (
    <div
      style={{
        width: d.width,
        height: d.height,
        // Tufte: a dashed hairline is enough to say "these belong together".
        background: "transparent",
        border: "1px dashed var(--rule)",
        borderRadius: 0,
        padding: "26px 14px 14px",
        position: "relative",
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 8,
          left: 14,
          fontSize: 13,
          letterSpacing: "0.08em",
          color: tone.label,
          fontVariantCaps: "all-small-caps",
          fontFamily: "var(--font-ui)",
        }}
      >
        {d.title}
        {d.subtitle && (
          <span style={{ marginLeft: 8, fontStyle: "italic", color: "var(--ink-3)", letterSpacing: 0, fontVariantCaps: "normal" }}>
            {d.subtitle}
          </span>
        )}
      </div>
    </div>
  );
}
