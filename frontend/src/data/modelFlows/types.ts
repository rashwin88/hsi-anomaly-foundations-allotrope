// Shared node/edge shape for the Models flow charts.
//
// Each architecture contributes a `ModelFlow` with stage nodes and
// optional group "subgraph" backgrounds. Group nodes carry a labelled
// translucent rect; child stages opt into a parent via `parentId`.

import type { Edge, Node } from "@xyflow/react";

export type StageKind =
  | "input"
  | "normalize"
  | "encoder"
  | "bottleneck"
  | "decoder"
  | "denormalize"
  | "output"
  | "mask"
  | "loss"
  | "special";

export interface ModelFlowNodeData extends Record<string, unknown> {
  kind: StageKind;
  label: string;        // bold heading, e.g. "Encoder · Stage 2"
  shape?: string;       // grey subtitle, e.g. "(B, 64, 64, 64)"
  detail?: string;      // optional caption under shape
}

export interface ModelFlowGroupData extends Record<string, unknown> {
  title: string;
  subtitle?: string;
  tone?: "encoder" | "decoder" | "spectral" | "neutral";
  width: number;
  height: number;
}

export type ModelFlowNode = Node<ModelFlowNodeData | ModelFlowGroupData>;
export type ModelFlowEdge = Edge;

export interface ModelFlow {
  architecture: string;
  description: string;  // 2-4 sentence prose intro for the detail page
  nodes: ModelFlowNode[];
  edges: ModelFlowEdge[];
}
