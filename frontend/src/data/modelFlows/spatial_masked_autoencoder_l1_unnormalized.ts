// Asanskrita — 3-channel input (input + validity + prediction-mask),
// L1 loss, NO normalize / denormalize. Raw °C in, raw °C out.

import type { ModelFlow } from "./types";

const COL = 320;

const ENC_X = 0;
const ENC_Y = 280;
const ENC_W = 600;
const ENC_H = 200;
const ENC_PAD_X = 50;
const ENC_PAD_Y = 50;
const ENC_STEP = 250;

const DEC_X = 0;
const DEC_Y = 650;
const DEC_W = 600;
const DEC_H = 200;
const DEC_PAD_X = 50;
const DEC_PAD_Y = 50;
const DEC_STEP = 250;

export const spatialMaskedAutoencoderL1UnnormalizedFlow: ModelFlow = {
  architecture: "spatial_masked_autoencoder_l1_unnormalized",
  description:
    "Conv encoder/decoder where validity and prediction masks are passed in as explicit input channels (3-channel input), and normalization is omitted entirely — the model consumes raw °C and produces raw °C. The 'unrefined' (asaṃskṛta) name doubles as a literal description: no PixelNormalize, no PixelDenormalize, no register_buffer scalars. Useful as the floor against which the normalized variant (Drashta) is compared.",
  nodes: [
    { id: "in",    type: "modelStage", position: { x: COL - 320, y: 0 }, data: { kind: "input", label: "Thermal patch",      shape: "(B, 1, H, W)" } },
    { id: "val",   type: "modelStage", position: { x: COL,        y: 0 }, data: { kind: "mask",  label: "Validity channel",   shape: "(B, 1, H, W)", detail: "1 = valid, 0 = nodata" } },
    { id: "pred",  type: "modelStage", position: { x: COL + 320,  y: 0 }, data: { kind: "mask",  label: "Prediction-mask channel", shape: "(B, 1, H, W)", detail: "1 = pixel to predict" } },
    { id: "stack", type: "modelStage", position: { x: COL,        y: 130 }, data: { kind: "special", label: "Stack along channel dim", shape: "(B, 3, H, W)" } },

    {
      id: "g_enc", type: "modelGroup", position: { x: ENC_X, y: ENC_Y },
      data: { title: "Encoder · 2 conv blocks", subtitle: "input fed in raw — no normalize", tone: "encoder", width: ENC_W, height: ENC_H },
      draggable: false, selectable: false, style: { width: ENC_W, height: ENC_H, zIndex: -1 },
    },
    { id: "e0", type: "modelStage", position: { x: ENC_PAD_X + 0 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 0", shape: "3 → 64 ch · H/2", detail: "Conv4 + BN + GELU + Dropout" } },
    { id: "e1", type: "modelStage", position: { x: ENC_PAD_X + 1 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder", label: "Stage 1", shape: "64 → 128 ch · H/4" } },

    { id: "z", type: "modelStage", position: { x: COL, y: 530 }, data: { kind: "bottleneck", label: "Bottleneck z", shape: "(B, 128, H/4, W/4)" } },

    {
      id: "g_dec", type: "modelGroup", position: { x: DEC_X, y: DEC_Y },
      data: { title: "Decoder · 2 transpose-conv blocks", subtitle: "no denormalize — output is raw °C", tone: "decoder", width: DEC_W, height: DEC_H },
      draggable: false, selectable: false, style: { width: DEC_W, height: DEC_H, zIndex: -1 },
    },
    { id: "d1", type: "modelStage", position: { x: DEC_PAD_X + 0 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 1", shape: "128 → 64 ch · ×2" } },
    { id: "d0", type: "modelStage", position: { x: DEC_PAD_X + 1 * DEC_STEP, y: DEC_PAD_Y }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Stage 0", shape: "64 → 1 ch · ×2", detail: "no BN/GELU — raw output" } },

    { id: "out",  type: "modelStage", position: { x: COL, y: 880 }, data: { kind: "output", label: "Reconstruction x̂", shape: "(B, 1, H, W)", detail: "raw °C — no denormalize step" } },
    { id: "loss", type: "modelStage", position: { x: COL + 380, y: 880 }, data: { kind: "loss",  label: "Masked L1", detail: "loss only on prediction-mask = 1" } },
  ],
  edges: [
    { id: "e_in_stack",   source: "in",   target: "stack", animated: true },
    { id: "e_val_stack",  source: "val",  target: "stack", animated: true, label: "channel" },
    { id: "e_pred_stack", source: "pred", target: "stack", animated: true, label: "channel" },
    { id: "e_stack_e0",   source: "stack", target: "e0",   animated: true },
    { id: "e_e0_e1", source: "e0", target: "e1", animated: true, label: "↓2" },
    { id: "e_e1_z",  source: "e1", target: "z",  animated: true },
    { id: "e_z_d1",  source: "z",  target: "d1", animated: true },
    { id: "e_d1_d0", source: "d1", target: "d0", animated: true, label: "×2" },
    { id: "e_d0_out", source: "d0", target: "out", animated: true },
    { id: "e_d0_loss", source: "d0", target: "loss", type: "smoothstep", animated: false, label: "training only", style: { strokeDasharray: "4 4", stroke: "#fca5a5" } },
  ],
};
