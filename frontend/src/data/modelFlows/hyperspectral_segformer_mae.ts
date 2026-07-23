// Indradhanu — hyperspectral SegFormer MAE.
// Layout: vertical pre/post pipeline + horizontal 4-stage encoder block
// + vertical decoder pipeline (5 explicit steps + PixelShuffle) +
// spectral compress / decompress bookends. Mirrors SegFormer paper Fig. 2.

import type { ModelFlow } from "./types";

// Layout grid
const COL = 320;       // central column x

// === Encoder (horizontal 4-stage row) ===
const ENC_X = -480;          // group top-left
const ENC_Y = 660;
const ENC_W = 1130;
const ENC_H = 200;
const ENC_PAD_X = 50;        // padding inside group
const ENC_PAD_Y = 50;
const ENC_STEP = 260;        // distance between stages

// === Decoder pipeline (vertical column) ===
const DEC_X = 110;
const DEC_Y = 920;
const DEC_W = 480;
const DEC_H = 700;
const DEC_PAD_X = 30;
const DEC_PAD_Y = 50;
const DEC_STEP = 100;

export const hyperspectralSegformerMaeFlow: ModelFlow = {
  architecture: "hyperspectral_segformer_mae",
  description:
    "The hyperspectral cousin of Chakshu. Same 4-stage hierarchical SegFormer transformer + token-level masking, but bookended by a learnable Spectral Compressor (165 → D channels via 1×1 conv + LayerNorm + GELU) and Spectral Decompressor (D → 165). The decoder fuses all four encoder stages at H/4 via 1×1 projections + bilinear upsample + concat + 1×1 fuse, refines with a Conv3×3 pair, then uses PixelShuffle(4) for learned per-pixel upsampling to full resolution. Loss is L1 + Spectral-Angle Mapper.",
  nodes: [
    // --- Pre-encoder pipeline (vertical column at COL) ---
    { id: "in",     type: "modelStage", position: { x: COL, y: 0   }, data: { kind: "input",     label: "Hyperspectral patch",   shape: "(B, 165, 128, 128)", detail: "PRISMA / EnMAP reflectance" } },
    { id: "norm",   type: "modelStage", position: { x: COL, y: 110 }, data: { kind: "normalize", label: "PixelNormalize · per-band", shape: "μ, σ shape (1, 165, 1, 1)" } },
    { id: "comp",   type: "modelStage", position: { x: COL, y: 220 }, data: { kind: "special",   label: "SpectralCompressor",    shape: "165 → 32 channels", detail: "1×1 Conv + LayerNorm + GELU" } },
    { id: "patch",  type: "modelStage", position: { x: COL, y: 330 }, data: { kind: "special",   label: "OverlapPatchEmbed",     shape: "→ tokens", detail: "stride < kernel" } },
    { id: "tokmsk", type: "modelStage", position: { x: COL + 320, y: 440 }, data: { kind: "mask", label: "Token-level masking", detail: "physically remove prediction tokens" } },

    // --- Encoder group ---
    {
      id: "g_enc",
      type: "modelGroup",
      position: { x: ENC_X, y: ENC_Y },
      data: { title: "Encoder · 4-stage hierarchical SegFormer", subtitle: "patch-merging halves H,W per stage", tone: "encoder", width: ENC_W, height: ENC_H },
      draggable: false,
      selectable: false,
      style: { width: ENC_W, height: ENC_H, zIndex: -1 },
    },
    { id: "s1", type: "modelStage", position: { x: ENC_PAD_X + 0 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder",    label: "Stage 1", shape: "32 ch · H/4",   detail: "2 blocks · 2 heads · MixFFN · efficient attn" } },
    { id: "s2", type: "modelStage", position: { x: ENC_PAD_X + 1 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder",    label: "Stage 2", shape: "64 ch · H/8",   detail: "2 blocks · 2 heads" } },
    { id: "s3", type: "modelStage", position: { x: ENC_PAD_X + 2 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "encoder",    label: "Stage 3", shape: "160 ch · H/16", detail: "2 blocks · 5 heads" } },
    { id: "s4", type: "modelStage", position: { x: ENC_PAD_X + 3 * ENC_STEP, y: ENC_PAD_Y }, parentId: "g_enc", extent: "parent", data: { kind: "bottleneck", label: "Stage 4", shape: "256 ch · H/32", detail: "2 blocks · 8 heads · bottleneck" } },

    // --- Decoder group ---
    {
      id: "g_dec",
      type: "modelGroup",
      position: { x: DEC_X, y: DEC_Y },
      data: { title: "SegFormer Decoder · MLP fuse → PixelShuffle", subtitle: "per-stage 1×1 → upsample → concat → fuse → refine → PixelShuffle(4)", tone: "decoder", width: DEC_W, height: DEC_H },
      draggable: false,
      selectable: false,
      style: { width: DEC_W, height: DEC_H, zIndex: -1 },
    },
    { id: "d1", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 0 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Per-stage 1×1 Conv",     shape: "[32, 64, 160, 256] → 256 ch", detail: "linear projection per F1..F4" } },
    { id: "d2", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 1 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Bilinear upsample",      shape: "→ H/4 across all stages", detail: "common spatial size" } },
    { id: "d3", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 2 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Concat along channels",  shape: "(B, 4·256, H/4, W/4)", detail: "fuse multi-scale features" } },
    { id: "d4", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 3 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "1×1 Fuse Conv + GELU",   shape: "→ 256 ch · H/4" } },
    { id: "d5", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 4 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "Refine · 3×3 Conv → GELU → 3×3 Conv", shape: "(B, 32·16, H/4, W/4)", detail: "predict 16 sub-pixels per position" } },
    { id: "d6", type: "modelStage", position: { x: DEC_PAD_X, y: DEC_PAD_Y + 5 * DEC_STEP }, parentId: "g_dec", extent: "parent", data: { kind: "decoder", label: "PixelShuffle(4)",        shape: "→ (B, 32, 128, 128)", detail: "rearrange channels into 4×4 spatial blocks · learned per-pixel upsample" } },

    // --- Post-decoder pipeline ---
    { id: "decomp", type: "modelStage", position: { x: COL, y: 1640 }, data: { kind: "special",     label: "SpectralDecompressor", shape: "32 → 165 channels", detail: "inverse 1×1 projection" } },
    { id: "denorm", type: "modelStage", position: { x: COL, y: 1750 }, data: { kind: "denormalize", label: "PixelDenormalize · per-band" } },
    { id: "out",    type: "modelStage", position: { x: COL, y: 1860 }, data: { kind: "output",      label: "Reconstruction x̂",     shape: "(B, 165, 128, 128)" } },
    { id: "loss",   type: "modelStage", position: { x: COL + 380, y: 1750 }, data: { kind: "loss",  label: "L1 + SAM",              detail: "L1 on values + Spectral-Angle Mapper · SAM ramp = 10 epochs" } },
  ],
  edges: [
    // Pre-encoder
    { id: "e_in_norm",   source: "in",     target: "norm",   animated: true, label: "(B, 165, 128, 128)" },
    { id: "e_norm_comp", source: "norm",   target: "comp",   animated: true },
    { id: "e_comp_patch", source: "comp",  target: "patch",  animated: true, label: "(B, 32, 128, 128)" },

    // Patch-embed → encoder Stage 1; token mask intercepts here
    { id: "e_patch_s1",  source: "patch",  target: "s1",     animated: true },
    { id: "e_mask_s1",   source: "tokmsk", target: "s1",     animated: true, label: "drop tokens", style: { stroke: "#eab308" } },

    // Encoder serial
    { id: "e_s1_s2", source: "s1", target: "s2", animated: true, label: "patch merge ↓" },
    { id: "e_s2_s3", source: "s2", target: "s3", animated: true, label: "patch merge ↓" },
    { id: "e_s3_s4", source: "s3", target: "s4", animated: true, label: "patch merge ↓" },

    // Skip connections from each encoder stage into the decoder fuse step (d1)
    { id: "e_s1_d1", source: "s1", target: "d1", type: "smoothstep", animated: false, label: "F1 · skip", style: { stroke: "#94a3b8", strokeDasharray: "4 4" } },
    { id: "e_s2_d1", source: "s2", target: "d1", type: "smoothstep", animated: false, label: "F2 · skip", style: { stroke: "#94a3b8", strokeDasharray: "4 4" } },
    { id: "e_s3_d1", source: "s3", target: "d1", type: "smoothstep", animated: false, label: "F3 · skip", style: { stroke: "#94a3b8", strokeDasharray: "4 4" } },
    { id: "e_s4_d1", source: "s4", target: "d1", type: "smoothstep", animated: false, label: "F4", style: { stroke: "#94a3b8" } },

    // Decoder serial
    { id: "e_d1_d2", source: "d1", target: "d2", animated: true },
    { id: "e_d2_d3", source: "d2", target: "d3", animated: true },
    { id: "e_d3_d4", source: "d3", target: "d4", animated: true },
    { id: "e_d4_d5", source: "d4", target: "d5", animated: true },
    { id: "e_d5_d6", source: "d5", target: "d6", animated: true },

    // Post-decoder
    { id: "e_d6_decomp", source: "d6", target: "decomp", animated: true, label: "(B, 32, 128, 128)" },
    { id: "e_decomp_denorm", source: "decomp", target: "denorm", animated: true },
    { id: "e_denorm_out", source: "denorm", target: "out", animated: true },

    // Loss (training-only)
    { id: "e_d6_loss", source: "d6", target: "loss", type: "smoothstep", animated: false, label: "training only", style: { strokeDasharray: "4 4", stroke: "#fca5a5" } },
  ],
};
