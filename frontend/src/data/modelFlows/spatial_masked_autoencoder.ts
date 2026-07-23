// Antardhana — same conv ladder as Pratibimba, trained with random
// pixel-level masking + MSE loss on hidden positions.

import type { ModelFlow } from "./types";

const COL = 320;

const ENC_X = 0;
const ENC_Y = 350;
const ENC_W = 600;
const ENC_H = 200;
const ENC_PAD_X = 50;
const ENC_PAD_Y = 50;
const ENC_STEP = 250;

const DEC_X = 0;
const DEC_Y = 720;
const DEC_W = 600;
const DEC_H = 200;
const DEC_PAD_X = 50;
const DEC_PAD_Y = 50;
const DEC_STEP = 250;

export const spatialMaskedAutoencoderFlow: ModelFlow = {
  architecture: "spatial_masked_autoencoder",
  description:
    "Conv encoder-decoder trained with random pixel-level masking. At each step, ~half the pixels are zeroed and the loss is computed only on those hidden positions — the model has to reconstruct from neighbour context. Identical conv-block recipe to Pratibimba (Conv4-BN-GELU-Dropout) but at a 2-stage depth.",
  nodes: [
    { id: "in",    type: "modelStage", position: { x: COL, y: 0   }, data: { kind: "input", label: "Thermal patch",   shape: "(B, 1, 128, 128)" } },
    { id: "mask",  type: "modelStage", position: { x: COL - 320, y: 110 }, data: { kind: "mask", label: "Random pixel mask", shape: "(B, 1, 128, 128)", detail: "1 = visible, 0 = hidden" } },
    { id: "apply", type: "modelStage", position: { x: COL, y: 110 }, data: { kind: "mask", label: "x ← x · mask", detail: "zero hidden pixels" } },
    { id: "norm",  type: "modelStage", position: { x: COL, y: 220 }, data: { kind: "normalize", label: "PixelNormalize", shape: "μ=24.58, σ=13.57 (baked)" } },

    {
      id: "g_enc", type: "modelGroup", position: { x: ENC_X, y: ENC_Y },
      data: { title: "Encoder · 2 conv blocks", subtitle: "K=4 · S=2 · P=1", tone: "encoder", width: ENC_W, height: ENC_H },
      draggable: false, selectable: false,
      style: { width: ENC_W, height: ENC_H, zIndex: -1 },
    },
    { id: "e0", type: "modelStage", position: { x: ENC_PAD_X + 0 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 0", shape: "1 → 64 ch · H/2", detail: "Conv4 + BN + GELU + Dropout" } },
    { id: "e1", type: "modelStage", position: { x: ENC_PAD_X + 1 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 1", shape: "64 → 128 ch · H/4" } },

    { id: "z", type: "modelStage", position: { x: COL, y: 600 }, data: { kind: "bottleneck", label: "Bottleneck z", shape: "(B, 128, 32, 32)" } },

    {
      id: "g_dec", type: "modelGroup", position: { x: DEC_X, y: DEC_Y },
      data: { title: "Decoder · 2 transpose-conv blocks", subtitle: "ConvT4 · ×2 H,W per block · last has no BN/GELU", tone: "decoder", width: DEC_W, height: DEC_H },
      draggable: false, selectable: false,
      style: { width: DEC_W, height: DEC_H, zIndex: -1 },
    },
    { id: "d1", type: "modelStage", position: { x: DEC_PAD_X + 0 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 1", shape: "128 → 64 ch · ×2" } },
    { id: "d0", type: "modelStage", position: { x: DEC_PAD_X + 1 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 0", shape: "64 → 1 ch · ×2", detail: "no BN/GELU — raw output" } },

    { id: "denorm", type: "modelStage", position: { x: COL, y: 950 }, data: { kind: "denormalize", label: "PixelDenormalize" } },
    { id: "out",    type: "modelStage", position: { x: COL, y: 1060 }, data: { kind: "output", label: "Reconstruction x̂", shape: "(B, 1, 128, 128)" } },
    { id: "loss",   type: "modelStage", position: { x: COL + 380, y: 950 }, data: { kind: "loss", label: "Masked MSE on hidden pixels", detail: "loss only where mask = 0" } },
  ],
  edges: [
    { id: "e_mask_apply", source: "mask", target: "apply", animated: true, label: "mask channel" },
    { id: "e_in_apply",   source: "in",   target: "apply", animated: true },
    { id: "e_apply_norm", source: "apply", target: "norm", animated: true },
    { id: "e_norm_e0",    source: "norm",  target: "e0",   animated: true },
    { id: "e_e0_e1", source: "e0", target: "e1", animated: true, label: "↓2" },
    { id: "e_e1_z",  source: "e1", target: "z",  animated: true },
    { id: "e_z_d1",  source: "z",  target: "d1", animated: true },
    { id: "e_d1_d0", source: "d1", target: "d0", animated: true, label: "×2" },
    { id: "e_d0_denorm", source: "d0", target: "denorm", animated: true },
    { id: "e_denorm_out", source: "denorm", target: "out", animated: true },
    { id: "e_d0_loss", source: "d0", target: "loss", type: "smoothstep", animated: false, label: "training only", style: { strokeDasharray: "4 4", stroke: "#fca5a5" } },
  ],
};
