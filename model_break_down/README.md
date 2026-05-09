# Model Break Down

A deep, plain-language walk-through of every model in this repository — what it is, how it learns, how it is used at inference, and the tensor shapes that flow through it. Diagrams use Mermaid. Anomaly detection is the goal of all of these models, but each one attacks the problem from a different angle.

## Mental model: how anomaly detection works here

Two big families of techniques live in this repo. Keep them straight before reading the per-model docs:

1. **Reconstruction-based (the ML models).** Train a neural network to reproduce *normal* satellite imagery from itself. At inference, ask the network to reconstruct a new scene. Wherever the reconstruction differs sharply from reality, the pixel is "surprising" — possibly an anomaly. Think of it like asking a forger who has only ever seen sheep to draw the picture you hand them: when you hand them a wolf, the forgery looks wrong, and that wrongness is the score.

2. **Statistical (the detectors).** Don't train anything. Estimate the multivariate distribution of pixel values in the scene itself, then ask "how unlikely is each pixel under this distribution?" A pixel that sits many standard deviations away — measured properly with covariance — is flagged. Think of it like: build a histogram of the room's height distribution, then anyone taller than the 99.9th percentile gets flagged.

Both families return a `(H, W)` heat-map. Pixels with high scores are anomalies.

## File index

### Reconstruction-based (neural networks)

| # | Sensor | File | Architecture in one sentence |
|---|---|---|---|
| 01 | thermal | [Spatial Autoencoder](01_spatial_autoencoder.md) | A pure-conv encoder/decoder that compresses a 1-band thermal patch and reconstructs it. |
| 02 | thermal | [Spatial Masked Autoencoder (L1 / MSE)](02_spatial_masked_autoencoder.md) | The same conv autoencoder, but trained with random pixel-level masking so it must reconstruct *hidden* pixels from context. |
| 03 | thermal | [Normalized & Unnormalized Masked Autoencoders](03_normalized_unnormalized_masked_autoencoder.md) | The masked autoencoder with the validity / input mask passed in as **explicit channels**, so the network knows which pixels are blanked. |
| 04 | thermal | [SegFormer MAE (thermal)](04_segformer_mae_thermal.md) | A 4-stage hierarchical transformer with token-level masking — true MAE-style: prediction tokens are physically removed from the sequence. |
| 05 | hyperspectral (165 bands) | [Hyperspectral SegFormer MAE](05_hyperspectral_segformer_mae.md) | The thermal SegFormer wrapped with a learnable spectral compressor (165 → D channels) and L1+SAM loss. |

### Statistical detectors (no training)

| # | Sensor | File | Idea |
|---|---|---|---|
| 06 | thermal | [Thermal Global RX](06_thermal_global_rx.md) | Single-band z-score: `(x − μ)² / σ²` over the whole scene. |
| 07 | hyperspectral | [Global RX (Reed-Xiaoli)](07_global_rx_hyperspectral.md) | Multivariate Mahalanobis distance from the scene mean across all spectral bands. |
| 08 | hyperspectral | [Local RX](08_local_rx.md) | Same Mahalanobis distance, but the mean and covariance are estimated from a *ring* of neighbour pixels around each test pixel. |
| 09 | hyperspectral | [MNF + RX (global and local variants)](09_mnf_rx_compression.md) | Run RX, but first compress the cube to its top-k Minimum Noise Fraction components — denoises and shrinks dimensionality. |
| 10 | hyperspectral | [Statistical Ensembler](10_statistical_ensembler.md) | Run Global RX *and* Local RX, normalise both score maps to ranks in [0, 1], fuse with product / max / mean. |
| 11 | thermal | [B10 Adaptive Cloud Masker](11_b10_adaptive_cloud_masker.md) | A 5-component Gaussian Mixture Model on Landsat B10 brightness temperature; clusters whose mean is more than 12 °C colder than the scene median are labelled "cloud". |

## How to read each file

Every per-model doc follows the same structure:

1. **What problem it solves** — sensor, input shape, output shape, why this exists.
2. **Architecture / algorithm** — Mermaid diagram + tensor-shape walk-through.
3. **Methods and classes used** — every concrete piece referenced by name with a one-line job description.
4. **Training** (ML models only) — sequence diagram of the training loop, the loss function written out, masking logic.
5. **Inference** — sequence diagram, full-scene sliding-window logic, anomaly-score computation.
6. **Key knobs** — config parameters and their effects.
7. **Analogies and gotchas** — intuition pumps and the things that catch you off guard.

## Common abstractions worth knowing first

These appear across every ML model:

- **`FoundationTrainer`** ([app/abstract_classes/foundation_trainer.py](../app/abstract_classes/foundation_trainer.py)) — the base class for every trainer. It owns the optimizer (Adam), LR scheduler, checkpointing, the `train()` outer loop, and the dataloader-from-S3 plumbing. Concrete trainers only implement `build_model()`, `compute_loss()`, and `validation_step()`.

- **`FoundationInferencer`** ([app/abstract_classes/foundation_inferencer.py](../app/abstract_classes/foundation_inferencer.py)) — the inference equivalent. Owns weight loading, device selection, and `predict()` (the public entry point that wraps `infer()` in `torch.no_grad()` and moves tensors to device). Concrete inferencers implement `build_model()` and `infer()`.

- **`AnomalyDetector`** ([app/abstract_classes/anomaly_detector.py](../app/abstract_classes/anomaly_detector.py)) — the contract for every statistical detector: `__init__(vendable)`, optional `fit(**kwargs)`, then `detect(cube, validity_mask)` returning `(H, W)`.

- **Tensor layout convention.** Whenever you see `(B, C, H, W)`, that's PyTorch BSQ-style: batch, channels, height, width. This is what the dataloader hands the trainer and what every `nn.Module.forward()` expects. Numpy detectors work in `(C, H, W)` (no batch). The `ImageCubeOperations` helper converts between BSQ, BIL, BIP — see [CONTEXT.md](../CONTEXT.md).

- **Patch sizes.** Models train on small square patches (typically 128×128) but inference is run on full scenes via a sliding window with stride `patch_size // 2` and overlap-averaging. This is implemented in `predict_full_scene()` on each inferencer.

- **Masks, validity, prediction.** Three masks float around. Pin them down:
  - `validity_mask` (1 = valid pixel, 0 = invalid): which pixels are physically real (no cloud, no nodata).
  - `input_mask` (1 = visible, 0 = hidden): which pixels the network is allowed to see this forward pass.
  - `prediction_mask` (1 = target, 0 = not target): which pixels the loss is computed on.
  - Relationship: `input_mask = validity_mask − prediction_mask` for masked autoencoders.

Read in order if you're new: `01 → 02 → 04 → 05` walks you up the ML ladder; `06 → 07 → 08 → 09 → 10` walks you up the statistical ladder.
