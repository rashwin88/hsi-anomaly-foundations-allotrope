// Chakshu — thermal SegFormer MAE.
// 4-stage hierarchical transformer with TRUE token-level masking
// (prediction tokens physically removed from the sequence). Decoder
// fuses all stages at H/4 then learns per-pixel upsampling via
// PixelShuffle(4). Mirrors SegFormer paper Fig. 2 layout.

import type { ModelFlow } from "./types";

const COL = 320;

// Encoder group
const ENC_X = -480;
const ENC_Y = 550;
const ENC_W = 1130;
const ENC_H = 200;
const ENC_PAD_X = 50;
const ENC_PAD_Y = 50;
const ENC_STEP = 260;

// Decoder group
const DEC_X = 110;
const DEC_Y = 810;
const DEC_W = 480;
const DEC_H = 700;
const DEC_PAD_X = 30;
const DEC_PAD_Y = 50;
const DEC_STEP = 100;

export const segformerMaeFlow: ModelFlow = {
  architecture: "segformer_mae",
  description:
    "A 4-stage hierarchical SegFormer transformer wrapped as a Masked Autoencoder. Unlike convolutional MAEs that just zero hidden pixels, this model physically removes prediction tokens from the attention sequence — so the encoder only attends to visible tokens, exactly the behaviour that made the original MAE paper work. The decoder fuses multi-scale features at H/4 via 1×1 projections + bilinear upsample + concat + 1×1 fuse, then learns per-pixel upsampling to full resolution via PixelShuffle(4) instead of bilinear interpolation — critical for point-anomaly fidelity.",
  nodes: [
    // Pre-encoder pipeline
    { id: "in",     type: "modelStage", position: { x: COL, y: 0   }, data: { kind: "input",     label: "Thermal patch",         shape: "(B, 1, 256, 256)" } },
    { id: "norm",   type: "modelStage", position: { x: COL, y: 110 }, data: { kind: "normalize", label: "PixelNormalize",        shape: "μ=24.58, σ=13.57 (baked)" } },
    { id: "patch",  type: "modelStage", position: { x: COL, y: 220 }, data: { kind: "special",   label: "OverlapPatchEmbed",     shape: "→ tokens", detail: "stride < kernel; tokens overlap" } },
    { id: "tokmsk", type: "modelStage", position: { x: COL + 320, y: 330 }, data: { kind: "mask", label: "Token-level masking", detail: "physically remove prediction tokens" } },

    // Encoder group
    {
      id: "g_enc",
      type: "modelGroup",
      position: { x: ENC_X, y: ENC_Y },
      data: { title: "Encoder · 4-stage hierarchical SegFormer", subtitle: "patch-merging halves H,W per stage", tone: "encoder", width: ENC_W, height: ENC_H },
      draggable: false,
      selectable: false,
      style: { width: ENC_W, height: ENC_H, zIndex: -1 },
    },
    { id: "s1", type: "modelStage", position: { x: ENC_PAD_X + 0 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder",    label: "Stage 1", shape: "16 ch · H/4",  detail: "1 block · 1 head · MixFFN · efficient attn" } },
    { id: "s2", type: "modelStage", position: { x: ENC_PAD_X + 1 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder",    label: "Stage 2", shape: "32 ch · H/8",  detail: "1 block · 2 heads" } },
    { id: "s3", type: "modelStage", position: { x: ENC_PAD_X + 2 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder",    label: "Stage 3", shape: "64 ch · H/16", detail: "1 block · 4 heads" } },
    { id: "s4", type: "modelStage", position: { x: ENC_PAD_X + 3 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "bottleneck", label: "Stage 4", shape: "96 ch · H/32", detail: "1 block · 8 heads · bottleneck" } },

    // Decoder group
    {
      id: "g_dec",
      type: "modelGroup",
      position: { x: DEC_X, y: DEC_Y },
      data: { title: "SegFormer Decoder · MLP fuse → PixelShuffle", subtitle: "per-stage 1×1 → upsample → concat → fuse → refine → PixelShuffle(4)", tone: "decoder", width: DEC_W, height: DEC_H },
      draggable: false,
      selectable: false,
      style: { width: DEC_W, height: DEC_H, zIndex: -1 },
    },
    { id: "d1", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 0 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Per-stage 1×1 Conv",     shape: "[16, 32, 64, 96] → 96 ch", detail: "linear projection per F1..F4" } },
    { id: "d2", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 1 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Bilinear upsample",      shape: "→ H/4 across all stages" } },
    { id: "d3", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 2 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Concat along channels",  shape: "(B, 4·96, H/4, W/4)" } },
    { id: "d4", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 3 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "1×1 Fuse Conv + GELU",   shape: "→ 96 ch · H/4" } },
    { id: "d5", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 4 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Refine · 3×3 Conv → GELU → 3×3 Conv", shape: "(B, 1·16, H/4, W/4)", detail: "predict 16 sub-pixels per position" } },
    { id: "d6", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 5 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "PixelShuffle(4)",        shape: "→ (B, 1, 256, 256)", detail: "rearrange channels into 4×4 spatial blocks · learned per-pixel upsample" } },

    // Post-decoder
    { id: "denorm", type: "modelStage", position: { x: COL, y: 1530 }, data: { kind: "denormalize", label: "PixelDenormalize" } },
    { id: "out",    type: "modelStage", position: { x: COL, y: 1640 }, data: { kind: "output",      label: "Reconstruction x̂",     shape: "(B, 1, 256, 256)", detail: "anomaly = |x − x̂|" } },
    { id: "loss",   type: "modelStage", position: { x: COL + 380, y: 1530 }, data: { kind: "loss",  label: "L1 on prediction tokens", detail: "loss only at the removed positions" } },
  ],
  edges: [
    { id: "e_in_norm",   source: "in",     target: "norm",   animated: true, label: "(B, 1, 256, 256)" },
    { id: "e_norm_patch", source: "norm",  target: "patch",  animated: true },
    { id: "e_patch_s1",  source: "patch",  target: "s1",     animated: true },
    { id: "e_mask_s1",   source: "tokmsk", target: "s1",     animated: true, label: "drop tokens", style: { stroke: "#eab308" } },

    { id: "e_s1_s2", source: "s1", target: "s2", animated: true, label: "patch merge ↓" },
    { id: "e_s2_s3", source: "s2", target: "s3", animated: true, label: "patch merge ↓" },
    { id: "e_s3_s4", source: "s3", target: "s4", animated: true, label: "patch merge ↓" },

    { id: "e_s1_d1", source: "s1", target: "d1", type: "smoothstep", animated: false, label: "F1 · skip", style: { stroke: "#94a3b8", strokeDasharray: "4 4" } },
    { id: "e_s2_d1", source: "s2", target: "d1", type: "smoothstep", animated: false, label: "F2 · skip", style: { stroke: "#94a3b8", strokeDasharray: "4 4" } },
    { id: "e_s3_d1", source: "s3", target: "d1", type: "smoothstep", animated: false, label: "F3 · skip", style: { stroke: "#94a3b8", strokeDasharray: "4 4" } },
    { id: "e_s4_d1", source: "s4", target: "d1", type: "smoothstep", animated: false, label: "F4", style: { stroke: "#94a3b8" } },

    { id: "e_d1_d2", source: "d1", target: "d2", animated: true },
    { id: "e_d2_d3", source: "d2", target: "d3", animated: true },
    { id: "e_d3_d4", source: "d3", target: "d4", animated: true },
    { id: "e_d4_d5", source: "d4", target: "d5", animated: true },
    { id: "e_d5_d6", source: "d5", target: "d6", animated: true },

    { id: "e_d6_denorm", source: "d6", target: "denorm", animated: true, label: "(B, 1, 256, 256)" },
    { id: "e_denorm_out", source: "denorm", target: "out", animated: true },

    { id: "e_d6_loss", source: "d6", target: "loss", type: "smoothstep", animated: false, label: "training only", style: { strokeDasharray: "4 4", stroke: "#fca5a5" } },
  ],
};
