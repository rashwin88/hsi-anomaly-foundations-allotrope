# Chapter 3 — Foundation Model Architectures

This chapter is a textbook walk-through of every architectural component that lives in
[`app/foundation_models/components/`](../app/foundation_models/components/). Each section
explains what the code does, what the theory is, and walks at least one numerical example
by hand. The chapter has been split into per-section files for easier navigation; read the
linked files in order, or jump to whichever component you need.

## Sections

- [00 — Architecture Map](03_model_architectures/00_architecture_map.md) — the seven models, their codenames, and which components compose which.
- [01 — `PixelNormalize` / `PixelDenormalize`](03_model_architectures/01_pixel_normalize.md) — per-channel z-score wrapped in a buffer-backed `nn.Module`.
- [02 — `SpatialEncoder` / `SpatialEncoderBlock`](03_model_architectures/02_spatial_encoder.md) — the conv encoder shared by the four convolutional AEs.
- [03 — `SpatialDecoder` / `SpatialDecoderBlock`](03_model_architectures/03_spatial_decoder.md) — the symmetric ConvTranspose decoder.
- [04 — `SpatialAutoencoder` (Pratibimba)](03_model_architectures/04_spatial_autoencoder_pratibimba.md) — the textbook autoencoder; the cleanest mirror in the collection.
- [05 — `UnNormalizedSpatialAutoencoder` (Asanskrita)](03_model_architectures/05_unnormalized_spatial_autoencoder_asanskrita.md) — the unrefined variant: 3-channel input, no normalization.
- [06 — `NormalizedMaskedSpatialAutoencoder` (Drashta)](03_model_architectures/06_normalized_masked_spatial_autoencoder_drashta.md) — the seer: normalized + mask-aware.
- [07 — `OverlapPatchEmbedding`](03_model_architectures/07_overlap_patch_embedding.md) — strided conv that turns images into token sequences.
- [08 — `EfficientSelfAttention` (ESA)](03_model_architectures/08_efficient_self_attention.md) — multi-head attention with spatially-reduced K and V.
- [09 — `MixFFN`](03_model_architectures/09_mix_ffn.md) — Linear -> DWConv -> GELU -> Linear; replaces explicit position embeddings.
- [10 — `SegFormerBlock`](03_model_architectures/10_segformer_block.md) — the pre-norm transformer block with ESA and MixFFN.
- [11 — `SegFormerEncoder`](03_model_architectures/11_segformer_encoder.md) — 4-stage hierarchical transformer with Stage-1 MAE masking.
- [12 — `SegFormerDecoder`](03_model_architectures/12_segformer_decoder.md) — the lightweight MLP + PixelShuffle decoder.
- [13 — `SegFormerMAE` (Chakshu)](03_model_architectures/13_segformer_mae_chakshu.md) — the transformer MAE wrapper; the first model that genuinely "sees".
- [14 — `TokenMasking`](03_model_architectures/14_token_masking.md) — gather/scatter utilities for MAE plus erosion and checkerboard helpers.
- [15 — `SpectralCompressor` / `SpectralDecompressor`](03_model_architectures/15_spectral_compressor.md) — learned 1x1 conv that turns a 200-band cube into a 24-channel pseudo-image.
- [16 — `HyperspectralSegFormerMAE` (Indradhanu)](03_model_architectures/16_hyperspectral_segformer_mae_indradhanu.md) — Chakshu with the spectral comp/decomp sandwich on either side.
- [17 — `SAMLoss`](03_model_architectures/17_sam_loss.md) — spectral angle loss for shape-aware reconstruction.
- [18 — `BaseModel` registry](03_model_architectures/18_base_model_registry.md) — the small enum that tracks non-foundation utility models (currently the cloud-masker).
- [References](03_model_architectures/references.md) — papers cited across the chapter.

## How to read this chapter

The components factor into four reusable building blocks: conv encoder/decoder, transformer
encoder/decoder, masking utilities, and spectral / loss helpers. Each numbered model
composes a subset. If you read the sections linearly, the dependency order is the same as
the file numbering: normalization helpers first, then conv blocks, then conv autoencoders,
then the transformer primitives, then the SegFormer MAE wrappers, then the spectral and
loss helpers.
