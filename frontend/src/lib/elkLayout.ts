// ELK auto-layout for the Models flow charts.
//
// ModelDetailPage offers two layout modes: the hand-authored coordinates baked
// into each data/modelFlows/*.ts, and this automatic one. Hand layout is the
// default because it is tuned per architecture; ELK is the escape hatch for a
// graph whose hand coordinates have drifted after editing.
//
// The detail that makes this simple: React Flow and ELK share a coordinate
// convention - a child's position is relative to its parent - so `parentId`
// maps straight onto ELK's nested `children`, and results map back with no
// offset arithmetic.

import type {
  ModelFlowEdge,
  ModelFlowGroupData,
  ModelFlowNode,
} from "../data/modelFlows/types";

// Stage cards are sized by CSS, so ELK needs a stand-in. These match the
// hand-authored geometry, where stages step 260px horizontally and 110px
// vertically.
const STAGE_W = 220;
const STAGE_H = 80;

const LAYOUT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  // DOWN, not RIGHT. Every hand-authored flow reads top-to-bottom
  // (input -> normalize -> encoder -> bottleneck -> decoder -> output), and
  // auto-layout should not rotate the mental model users already have.
  "elk.direction": "DOWN",
  // Edges cross group boundaries (bottleneck sits outside both the encoder and
  // decoder groups), so ELK has to route across the hierarchy rather than
  // treating each group as opaque.
  "elk.hierarchyHandling": "INCLUDE_CHILDREN",
  "elk.layered.spacing.nodeNodeBetweenLayers": "60",
  "elk.spacing.nodeNode": "40",
  "elk.padding": "[top=50,left=50,bottom=50,right=50]",
};

interface ElkNode {
  id: string;
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  children?: ElkNode[];
}

function isGroup(node: ModelFlowNode): boolean {
  return node.type === "modelGroup";
}

function sizeOf(node: ModelFlowNode): { width: number; height: number } {
  if (isGroup(node)) {
    const data = node.data as ModelFlowGroupData;
    return { width: data.width, height: data.height };
  }
  return { width: STAGE_W, height: STAGE_H };
}

export async function layoutWithElk(
  nodes: ModelFlowNode[],
  edges: ModelFlowEdge[],
): Promise<ModelFlowNode[]> {
  // Lazy import: elkjs is ~1 MB and only needed once a user opts into auto
  // layout. vite.config.ts already isolates it in the `flow` chunk.
  // @ts-ignore - elkjs ships no type declarations for the browser bundle path
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  const elk = new ELK();

  // Rebuild the graph as a hierarchy. Group children nest one level deep;
  // everything else sits at the root.
  const childrenOf = new Map<string, ElkNode[]>();
  const roots: ElkNode[] = [];

  for (const node of nodes) {
    const { width, height } = sizeOf(node);
    const elkNode: ElkNode = { id: node.id, width, height };
    if (node.parentId) {
      const siblings = childrenOf.get(node.parentId) ?? [];
      siblings.push(elkNode);
      childrenOf.set(node.parentId, siblings);
    } else {
      roots.push(elkNode);
    }
  }
  for (const root of roots) {
    const kids = childrenOf.get(root.id);
    if (kids && kids.length > 0) root.children = kids;
  }

  const laidOut = await elk.layout({
    id: "root",
    layoutOptions: LAYOUT_OPTIONS,
    children: roots,
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  });

  // Flatten the nested result into id -> geometry.
  const placed = new Map<string, ElkNode>();
  const collect = (list: ElkNode[] | undefined): void => {
    for (const node of list ?? []) {
      placed.set(node.id, node);
      collect(node.children);
    }
  };
  collect(laidOut.children as ElkNode[] | undefined);

  return nodes.map((node) => {
    const result = placed.get(node.id);
    if (!result) return node;

    const next: ModelFlowNode = {
      ...node,
      position: { x: result.x ?? 0, y: result.y ?? 0 },
    };

    // A group has to grow to whatever ELK needed, in both `data` (which the
    // node component reads) and `style` (which React Flow uses to size the
    // container). Update one and not the other and the group clips its
    // children.
    if (isGroup(node) && result.width && result.height) {
      next.data = {
        ...(node.data as ModelFlowGroupData),
        width: result.width,
        height: result.height,
      };
      next.style = {
        ...(node.style ?? {}),
        width: result.width,
        height: result.height,
      };
    }
    return next;
  });
}
