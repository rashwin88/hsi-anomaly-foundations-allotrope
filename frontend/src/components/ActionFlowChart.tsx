// ReactFlow-rendered recipe flowchart embedded in the workspace Action
// card. One per action type; visual language reused from the Models
// page (`modelStage` nodes, stage-kind colour palette).

import { Background, Controls, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { ACTION_FLOWS } from "../data/actionFlows";
import { MODEL_FLOW_NODE_TYPES } from "./ModelFlowNode";

interface ActionFlowChartProps {
  actionType: string;
  height?: number;
}

export function ActionFlowChart({
  actionType,
  height = 360,
}: ActionFlowChartProps) {
  const flow = ACTION_FLOWS[actionType];
  if (!flow) {
    return null;
  }
  return (
    <div className="action-flow-chart" style={{ height }}>
      <ReactFlow
        nodes={flow.nodes}
        edges={flow.edges}
        nodeTypes={MODEL_FLOW_NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        // Zoom is enabled across pinch / Ctrl+wheel / pan-zoom buttons /
        // double-click. Wide range on minZoom so dense recipes can be
        // zoomed all the way out without clipping.
        minZoom={0.2}
        maxZoom={2.5}
        zoomOnScroll
        zoomOnPinch
        zoomOnDoubleClick
        panOnDrag
        // Read-only on the nodes themselves.
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          style: { stroke: "#94a3b8", strokeWidth: 1.4 },
        }}
      >
        <Background gap={16} size={1} color="#cbd5e1" />
        <Controls
          showInteractive={false}
          position="bottom-right"
        />
      </ReactFlow>
    </div>
  );
}
