// Drashta — same 3-channel input shape as Asanskrita, but only the
// thermal channel is z-scored at the encoder boundary; output is
// denormalized back to °C. The mask channels pass through raw.

import type { ModelFlow } from "./types";

const COL = 320;

const ENC_X = 0;
const ENC_Y = 380;
const ENC_W = 600;
const ENC_H = 200;
const ENC_PAD_X = 50;
const ENC_PAD_Y = 50;
const ENC_STEP = 250;

const DEC_X = 0;
const DEC_Y = 750;
const DEC_W = 600;
const DEC_H = 200;
const DEC_PAD_X = 50;
const DEC_PAD_Y = 50;
const DEC_STEP = 250;

export const normalizedMaskedAutoencoderFlow: ModelFlow = {
  architecture: "normalized_masked_autoencoder",
  description:
    "Conv encoder/decoder that is 'mask-aware' — validity and prediction-mask channels are stacked into the input alongside the raw thermal patch, so the network can literally see which pixels are masked. Only the thermal channel is z-scored; the mask channels pass through normalization untouched. Output is denormalized back to °C.",
  nodes: [
    { id: "in",    type: "modelStage", position: { x: COL - 320, y: 0 }, data: { kind: "input", label: "Thermal patch",      shape: "(B, 1, H, W)" } },
    { id: "val",   type: "modelStage", position: { x: COL,        y: 0 }, data: { kind: "mask",  label: "Validity channel",   shape: "(B, 1, H, W)" } },
    { id: "pred",  type: "modelStage", position: { x: COL + 320,  y: 0 }, data: { kind: "mask",  label: "Prediction-mask channel", shape: "(B, 1, H, W)" } },
    { id: "norm",  type: "modelStage", position: { x: COL - 320, y: 130 }, data: { kind: "normalize", label: "PixelNormalize · thermal only", shape: "μ=24.58, σ=13.57 (baked)" } },
    { id: "stack", type: "modelStage", position: { x: COL,        y: 250 }, data: { kind: "special", label: "Stack along channel dim", shape: "(B, 3, H, W)" } },

    {
      id: "g_enc", type: "modelGroup", position: { x: ENC_X, y: ENC_Y },
      data: { title: "Encoder · 2 conv blocks", subtitle: "K=4 · S=2 · P=1", tone: "encoder", width: ENC_W, height: ENC_H },
      draggable: false, selectable: false, style: { width: ENC_W, height: ENC_H, zIndex: -1 },
    },
    { id: "e0", type: "modelStage", position: { x: ENC_PAD_X + 0 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 0", shape: "3 → 64 ch · H/2", detail: "Conv4 + BN + GELU + Dropout" } },
    { id: "e1", type: "modelStage", position: { x: ENC_PAD_X + 1 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 1", shape: "64 → 128 ch · H/4" } },

    { id: "z", type: "modelStage", position: { x: COL, y: 630 }, data: { kind: "bottleneck", label: "Bottleneck z", shape: "(B, 128, H/4, W/4)" } },

    {
      id: "g_dec", type: "modelGroup", position: { x: DEC_X, y: DEC_Y },
      data: { title: "Decoder · 2 transpose-conv blocks", subtitle: "ConvT4 · ×2 H,W per block", tone: "decoder", width: DEC_W, height: DEC_H },
      draggable: false, selectable: false, style: { width: DEC_W, height: DEC_H, zIndex: -1 },
    },
    { id: "d1", type: "modelStage", position: { x: DEC_PAD_X + 0 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 1", shape: "128 → 64 ch · ×2" } },
    { id: "d0", type: "modelStage", position: { x: DEC_PAD_X + 1 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 0", shape: "64 → 1 ch · ×2", detail: "no BN/GELU — raw output" } },

    { id: "denorm", type: "modelStage", position: { x: COL, y: 980 },  data: { kind: "denormalize", label: "PixelDenormalize" } },
    { id: "out",    type: "modelStage", position: { x: COL, y: 1090 }, data: { kind: "output", label: "Reconstruction x̂", shape: "(B, 1, H, W)" } },
    { id: "loss",   type: "modelStage", position: { x: COL + 380, y: 980 }, data: { kind: "loss", label: "Masked L1", detail: "loss only on prediction-mask = 1" } },
  ],
  edges: [
    { id: "e_in_norm", source: "in", target: "norm", animated: true },
    { id: "e_norm_stack", source: "norm", target: "stack", animated: true, label: "thermal · z-score" },
    { id: "e_val_stack",  source: "val",  target: "stack", animated: true, label: "raw" },
    { id: "e_pred_stack", source: "pred", target: "stack", animated: true, label: "raw" },
    { id: "e_stack_e0",   source: "stack", target: "e0",   animated: true },
    { id: "e_e0_e1", source: "e0", target: "e1", animated: true, label: "↓2" },
    { id: "e_e1_z",  source: "e1", target: "z",  animated: true },
    { id: "e_z_d1",  source: "z",  target: "d1", animated: true },
    { id: "e_d1_d0", source: "d1", target: "d0", animated: true, label: "×2" },
    { id: "e_d0_denorm", source: "d0", target: "denorm", animated: true },
    { id: "e_denorm_out", source: "denorm", target: "out", animated: true },
    { id: "e_d0_loss", source: "d0", target: "loss", type: "smoothstep", animated: false, label: "training only", style: { strokeDasharray: "4 4", stroke: "#fca5a5" } },
  ],
};
