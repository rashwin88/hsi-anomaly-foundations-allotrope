# Chapter 5 — Inference

This chapter covers the *inference* half of Allotrope's foundation-model
stack: how a trained reconstruction model is loaded, fed patches of a
scene, and turned into a per-pixel anomaly heatmap. Training (Chapter 4)
optimises a model end-to-end against masked or checkerboard
reconstruction objectives; inference reuses those same masking tricks
with weights frozen and gradients disabled, and additionally has to
*reassemble* full scenes from overlapping patches and *score* the
residual.

The chapter is split into nine sections, one per file in
[`05_inference/`](05_inference/).

## Sections

- [5.1 The Inferencer Abstraction](05_inference/01_inferencer_abstraction.md) — the
  `FoundationInferencer` base class, device / eval / no_grad invariants,
  factory dispatch.
- [5.2 SpatialAutoencoderInferencer](05_inference/02_spatial_autoencoder_inferencer.md) —
  checkerboard CNN reconstruction, pixel-level masking, per-pixel
  validity-weighted overlap averaging.
- [5.3 MaskedSpatialAutoencoderInferencer](05_inference/03_masked_spatial_autoencoder_inferencer.md) —
  same scheme on the L1 / unnormalised model; reconciliation of the
  apparent combine-line difference.
- [5.4 NormalizedMaskedAutoencoderInferencer](05_inference/04_normalized_masked_autoencoder_inferencer.md) —
  3-channel head with baked-in normalisation; canonical statement of
  the "pixel from the pass that hid it" rule.
- [5.5 SegFormerMAEInferencer](05_inference/05_segformer_mae_inferencer.md) —
  token-masked transformer, checkerboard and random strategies,
  batched tiling, mask erosion, validity-fraction filter, fallback.
- [5.6 HyperspectralSegFormerMAEInferencer](05_inference/06_hyperspectral_segformer_mae_inferencer.md) —
  spectral compressor / decompressor around SegFormer; dual L1 + SAM
  scoring.
- [5.7 The Inference Harness](05_inference/07_inference_harness.md) —
  framework-agnostic numpy orchestrator; score-level vs
  reconstruction-level overlap averaging.
- [5.8 Scoring](05_inference/08_scoring.md) — L1, MSE, SAM,
  combined; percentile thresholding; ROC / AUC.
- [5.9 Summary](05_inference/09_summary.md) — the layering and the
  four invariants every output heatmap satisfies.
