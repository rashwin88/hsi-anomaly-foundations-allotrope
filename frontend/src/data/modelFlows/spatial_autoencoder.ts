// Pratibimba — pure conv encoder/decoder with thermal normalization.
// Hierarchical conv ladder doubles channels and halves H/W per stage;
// decoder mirrors with ConvTranspose. Each block: Conv(K=4,S=2,P=1)
// → BN → GELU → Dropout(0.3).

import type { ModelFlow } from "./types";

const COL = 320;

const ENC_X = -480;
const ENC_Y = 250;
const ENC_W = 1130;
const ENC_H = 200;
const ENC_PAD_X = 50;
const ENC_PAD_Y = 50;
const ENC_STEP = 260;

const DEC_X = -480;
const DEC_Y = 620;
const DEC_W = 1130;
const DEC_H = 200;
const DEC_PAD_X = 50;
const DEC_PAD_Y = 50;
const DEC_STEP = 260;

export const spatialAutoencoderFlow: ModelFlow = {
  architecture: "spatial_autoencoder",
  description:
    "A vanilla convolutional encoder-decoder. Each encoder block is Conv(K=4, S=2, P=1) → BatchNorm → GELU → Dropout2d(0.3), halving H/W and doubling channels. The decoder mirrors with ConvTranspose blocks (no BN/GELU on the final layer — outputs raw temperatures). Trained with masked MSE; anything the model can't reconstruct (e.g. a wildfire it has never seen) shows up as a high pixel-wise error. Normalization is baked into the forward pass via register_buffer.",
  nodes: [
    { id: "in",   type: "modelStage", position: { x: COL, y: 0   }, data: { kind: "input",     label: "Thermal patch",   shape: "(B, 1, 256, 256)", detail: "Landsat-9 B10, °C" } },
    { id: "norm", type: "modelStage", position: { x: COL, y: 110 }, data: { kind: "normalize", label: "PixelNormalize",  shape: "(x − μ) / σ", detail: "μ=24.58, σ=13.57 (baked)" } },

    {
      id: "g_enc",
      type: "modelGroup",
      position: { x: ENC_X, y: ENC_Y },
      data: { title: "Encoder · 4 conv blocks", subtitle: "K=4 · S=2 · P=1 → halve H,W · double C", tone: "encoder", width: ENC_W, height: ENC_H },
      draggable: false,
      selectable: false,
      style: { width: ENC_W, height: ENC_H, zIndex: -1 },
    },
    { id: "e0", type: "modelStage", position: { x: ENC_PAD_X + 0 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 0", shape: "1 → 32 ch · H/2",  detail: "Conv4 + BN + GELU + Dropout2d" } },
    { id: "e1", type: "modelStage", position: { x: ENC_PAD_X + 1 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 1", shape: "32 → 64 ch · H/4" } },
    { id: "e2", type: "modelStage", position: { x: ENC_PAD_X + 2 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 2", shape: "64 → 128 ch · H/8" } },
    { id: "e3", type: "modelStage", position: { x: ENC_PAD_X + 3 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 3", shape: "128 → 256 ch · H/16" } },

    { id: "z", type: "modelStage", position: { x: COL, y: 480 }, data: { kind: "bottleneck", label: "Bottleneck z", shape: "(B, 256, 16, 16)" } },

    {
      id: "g_dec",
      type: "modelGroup",
      position: { x: DEC_X, y: DEC_Y },
      data: { title: "Decoder · 4 transpose-conv blocks", subtitle: "ConvT(K=4,S=2,P=1) → double H,W · halve C · last block has no BN/GELU", tone: "decoder", width: DEC_W, height: DEC_H },
      draggable: false,
      selectable: false,
      style: { width: DEC_W, height: DEC_H, zIndex: -1 },
    },
    { id: "d3", type: "modelStage", position: { x: DEC_PAD_X + 0 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 3", shape: "256 → 128 ch · ×2", detail: "ConvT4 + BN + GELU" } },
    { id: "d2", type: "modelStage", position: { x: DEC_PAD_X + 1 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 2", shape: "128 → 64 ch · ×2" } },
    { id: "d1", type: "modelStage", position: { x: DEC_PAD_X + 2 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 1", shape: "64 → 32 ch · ×2" } },
    { id: "d0", type: "modelStage", position: { x: DEC_PAD_X + 3 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 0", shape: "32 → 1 ch · ×2", detail: "no BN/GELU — raw output" } },

    { id: "denorm", type: "modelStage", position: { x: COL, y: 850 }, data: { kind: "denormalize", label: "PixelDenormalize", shape: "x · σ + μ" } },
    { id: "out",    type: "modelStage", position: { x: COL, y: 960 }, data: { kind: "output",      label: "Reconstruction x̂", shape: "(B, 1, 256, 256)", detail: "anomaly = (x − x̂)²" } },
    { id: "loss",   type: "modelStage", position: { x: COL + 380, y: 850 }, data: { kind: "loss",  label: "Masked MSE", detail: "Σ(x̂ − x)² · mask  /  Σ mask" } },
  ],
  edges: [
    { id: "e_in_norm", source: "in", target: "norm", animated: true },
    { id: "e_norm_e0", source: "norm", target: "e0", animated: true },
    { id: "e_e0_e1", source: "e0", target: "e1", animated: true, label: "↓2" },
    { id: "e_e1_e2", source: "e1", target: "e2", animated: true, label: "↓2" },
    { id: "e_e2_e3", source: "e2", target: "e3", animated: true, label: "↓2" },
    { id: "e_e3_z",  source: "e3", target: "z",  animated: true },
    { id: "e_z_d3",  source: "z",  target: "d3", animated: true },
    { id: "e_d3_d2", source: "d3", target: "d2", animated: true, label: "×2" },
    { id: "e_d2_d1", source: "d2", target: "d1", animated: true, label: "×2" },
    { id: "e_d1_d0", source: "d1", target: "d0", animated: true, label: "×2" },
    { id: "e_d0_denorm", source: "d0", target: "denorm", animated: true },
    { id: "e_denorm_out", source: "denorm", target: "out", animated: true },
    { id: "e_d0_loss", source: "d0", target: "loss", type: "smoothstep", animated: false, label: "training only", style: { strokeDasharray: "4 4", stroke: "#fca5a5" } },
  ],
};
