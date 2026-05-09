# Checkpoint Inventory — Architecture & Parameter Counts

Snapshot of every `.pt` file under `checkpoints/` (including `checkpoints/spatial_ae/`), loaded with `torch.load(weights_only=False)` and inspected via `model_state_dict`. Architectures were derived directly from tensor shapes; parameter counts are the sum of `numel()` across all tensors saved in `model_state_dict` (this includes BatchNorm running stats and `num_batches_tracked` counters, so the reported totals match what a `state_dict()` round-trip would write).

All checkpoints store: `epoch`, `model_state_dict`, `optimizer_state_dict` (empty in the saved files), `scheduler_state_dict`, `train_loss`, `val_losses`, `avg_val_loss`, `config`.

## Summary table

| File | Family | Params | Encoder dims | Decoder dim | Epoch | Avg Val Loss |
|---|---|---:|---|---|---:|---:|
| `hyperspectral_segformer_mae_v0.1.0_epoch2.pt`   | HyperSpectral SegFormer MAE | 5,205,538 | [32,64,160,256], D=24 | 256 | 2   | 0.04368 |
| `hyperspectral_segformer_mae_v0.1.0_epoch60.pt`  | HyperSpectral SegFormer MAE | 5,205,538 | [32,64,160,256], D=24 | 256 | 60  | 0.08180 |
| `hyperspectral_segformer_mae_v0.1.0_epoch63.pt`  | HyperSpectral SegFormer MAE | 5,205,538 | [32,64,160,256], D=24 | 256 | 63  | 0.08115 |
| `hyperspectral_segformer_mae_v0.1.0_epoch84.pt`  | HyperSpectral SegFormer MAE | 5,205,538 | [32,64,160,256], D=24 | 256 | 84  | 0.07850 |
| `hyperspectral_segformer_mae_v0.1.0_epoch92.pt`  | HyperSpectral SegFormer MAE | 5,205,538 | [32,64,160,256], D=24 | 256 | 92  | 0.07694 |
| `hyperspectral_segformer_mae_v0.2.0_epoch2.pt`   | HyperSpectral SegFormer MAE | 5,507,354 | [32,64,160,256], D=32 | 256 | 2   | 0.03496 |
| `hyperspectral_segformer_mae_v0.2.0_epoch55.pt`  | HyperSpectral SegFormer MAE | 5,507,354 | [32,64,160,256], D=32 | 256 | 55  | 0.05636 |
| `hyperspectral_segformer_mae_v0.2.0_epoch80.pt`  | HyperSpectral SegFormer MAE | 5,507,354 | [32,64,160,256], D=32 | 256 | 80  | 0.05294 |
| `hyperspectral_segformer_mae_v0.2.0_epoch200.pt` | HyperSpectral SegFormer MAE | 5,507,354 | [32,64,160,256], D=32 | 256 | 200 | 0.04349 |
| `segformer_mae_v0.1.0_epoch10.pt`                | SegFormer MAE (thermal) | 3,714,086 | [32,64,160,256] | 256 | 10  | 6.59877 |
| `segformer_mae_v0.8.0_epoch495.pt`               | SegFormer MAE (thermal, tiny) | 406,500 | [16,32,64,96]   | 96  | 495 | 0.25650 |
| `normalized_masked_autoencoder_v1.2.0_epoch69.pt`            | Normalized Masked AE | 267,528 | conv: 3→64→128 | — | 69  | 5.52873 |
| `spatial_autoencoder_v6.1.0_epoch58.pt`                      | Spatial AE | 330,314 | conv: 1→32→64→128 | — | 58  | 20.12617 |
| `spatial_autoencoder_v8.1.0_epoch29.pt`                      | Spatial AE | 1,380,812 | conv: 1→32→64→128→256 | — | 29  | 13.76964 |
| `spatial_autoencoder_v9.1.0_epoch7.pt`                       | Spatial AE | 1,315,978 | conv: 1→64→128→256 | — | 7   | 19.18425 |
| `spatial_masked_autoencoder_v0.1.0_epoch121.pt`              | Spatial Masked AE | 265,480 | conv: 1→64→128 | — | 121 | 17.28919 |
| `spatial_masked_autoencoder_l1_v0.2.0_epoch92.pt`            | Spatial Masked AE (L1) | 265,480 | conv: 1→64→128 | — | 92  | 2.47353 |
| `spatial_masked_autoencoder_l1_v0.4.0_epoch77.pt`            | Spatial Masked AE (L1) | 330,314 | conv: 1→32→64→128 | — | 77  | 1.79195 |
| `spatial_masked_autoencoder_l1_unnormalized_v0.3.0_epoch82.pt` | Spatial Masked AE (L1, unnorm) | 267,524 | conv: 3→64→128 | — | 82  | 2.14911 |
| `spatial_ae/spatial_autoencoder_v0.1.0_epoch5.pt`            | Spatial AE (legacy) | 330,310 | conv: 1→32→64→128 | — | 5   | 713.357 |
| `spatial_ae/spatial_autoencoder_v0.2.0_epoch5.pt`            | Spatial AE (legacy) | 330,310 | conv: 1→32→64→128 | — | 5   | 26.677 |
| `spatial_ae/spatial_autoencoder_v0.2.0_epoch25.pt`           | Spatial AE (legacy) | 330,310 | conv: 1→32→64→128 | — | 25  | 9.115 |
| `spatial_ae/spatial_autoencoder_v0.2.0_epoch30.pt`           | Spatial AE (legacy) | 330,310 | conv: 1→32→64→128 | — | 30  | 9.062 |
| `spatial_ae/spatial_autoencoder_v3.1.0_epoch25.pt`           | Spatial AE | 330,314 | conv: 1→32→64→128 | — | 25  | 18.742 |
| `spatial_ae/spatial_autoencoder_v5.1.0_epoch20.pt`           | Spatial AE | 330,314 | conv: 1→32→64→128 | — | 20  | 6.593 |
| `spatial_ae/spatial_autoencoder_v5.1.0_epoch28.pt`           | Spatial AE | 330,314 | conv: 1→32→64→128 | — | 28  | 6.064 |

Notes on totals: param counts include the `normalize` / `denormalize` per-band mean/std buffers when present (165×2 = 330 floats for hyperspectral; 1×2 buffers for thermal) and all BN running stats. The "training-only" trainable parameter count for transformer models would be slightly lower than the totals shown.

---

## Family 1 — HyperSpectral SegFormer MAE (`HyperspectralSegFormerMAE`)

Defined in [app/foundation_models/components/hyperspectral_seg_former_mae.py](app/foundation_models/components/hyperspectral_seg_former_mae.py).

### Forward path

```
(B, 165, H, W)                                  reflectance cube
  → mask × x        (zero out invalid pixels)
  → PixelNormalize  (per-band z-score, 165 mean+std buffers)
  → SpectralCompressor: Conv2d(165, D, kernel=1) + LayerNorm + GELU
  → SegFormerEncoder (4 stages, in_channels=D)
  → SegFormerDecoder (PixelShuffle, out_channels=D)
  → SpectralDecompressor: Conv2d(D, 165, kernel=1)
  → PixelDenormalize
(B, 165, H, W)
```

### v0.1.0 — D = 24 (5 checkpoints: epochs 2/60/63/84/92)

| Submodule | Params |
|---|---:|
| `normalize` (mean, std buffers) | 330 |
| `denormalize` (mean, std buffers) | 330 |
| `compressor` | 4,081 |
| `encoder` (4-stage SegFormer, embed_dims=[32,64,160,256], blocks=[2,2,2,2]) | 3,326,976 |
| `decoder` (multi-scale fusion, decoder_dim=256) | 1,869,696 |
| `decompressor` | 4,125 |
| **Total** | **5,205,538** |

Training config: `learning_rate=1e-4`, `T_max=500`, `eta_min=1e-6` cosine schedule. Best val loss observed at epoch 92 (0.0769).

### v0.2.0 — D = 32 (4 checkpoints: epochs 2/55/80/200)

| Submodule | Params |
|---|---:|
| `normalize`, `denormalize` | 660 |
| `compressor` | 5,441 |
| `encoder` (same SegFormer config; channels start at D=32) | 3,331,072 |
| `decoder` (decoder_dim=256, but reconstructs to D=32 ch) | 2,164,736 |
| `decompressor` | 5,445 |
| **Total** | **5,507,354** |

Training config: `learning_rate=1e-3`, `T_max=200`. Best val loss at epoch 2 (0.0350) — model later overfits/regresses; epoch 200 lands at 0.0435.

---

## Family 2 — SegFormer MAE (thermal) (`SegFormerMAE`)

Defined in [app/foundation_models/components/seg_former_mae.py](app/foundation_models/components/seg_former_mae.py). Single-band thermal reconstruction; identical structure to the hyperspectral variant minus the spectral compressor/decompressor.

### Forward path

```
(B, 1, H, W) → mask → PixelNormalize → SegFormerEncoder → SegFormerDecoder → PixelDenormalize → (B, 1, H, W)
```

### `segformer_mae_v0.1.0_epoch10.pt` — 3,714,086 params

- `embed_dims = [32, 64, 160, 256]`, `num_blocks = [2, 2, 2, 2]`, `decoder_dim = 256`
- Submodules: `encoder` 3,316,256 + `decoder` 397,826 + 4 normalize/denormalize buffers
- Training: `lr=1e-4`, `T_max=300`. Val loss 6.5988 at epoch 10.

### `segformer_mae_v0.8.0_epoch495.pt` — 406,500 params

A much smaller variant (architecture changed, ~9× fewer parameters):

- `embed_dims = [16, 32, 64, 96]`, `num_blocks = [1, 1, 1, 1]`, `decoder_dim = 96`
- Submodules: `encoder` 252,304 + `decoder` 154,192 + 4 buffers
- Training: `lr=1e-4`, `T_max=500`, ran for 495 epochs. Val loss 0.2565.

---

## Family 3 — Spatial Autoencoder (`SpatialAutoencoder`)

Defined in [app/foundation_models/components/spatial_auto_encoder.py](app/foundation_models/components/spatial_auto_encoder.py). Pure-conv encoder/decoder; each "stage" is `Conv2d(k=4, stride=2) → BatchNorm2d → ReLU` for the encoder and `ConvTranspose2d` for the decoder.

### Top-level checkpoints

| Checkpoint | Stages (encoder) | Params | Notes |
|---|---|---:|---|
| `spatial_autoencoder_v6.1.0_epoch58.pt` | 1 → 32 → 64 → 128 | 330,314 | Includes `normalize`/`denormalize` buffers |
| `spatial_autoencoder_v8.1.0_epoch29.pt` | 1 → 32 → 64 → 128 → 256 | 1,380,812 | One extra stage, doubles channel depth |
| `spatial_autoencoder_v9.1.0_epoch7.pt` | 1 → 64 → 128 → 256 | 1,315,978 | Wider first stage, 3 stages |

### `checkpoints/spatial_ae/` — legacy / experimental sweeps

All seven files use the same 1→32→64→128 three-stage architecture. The `v0.x` ones lack the `normalize`/`denormalize` modules (330,310 params); `v3.1.0` and `v5.1.0` add them (330,314 params).

| Checkpoint | Params | Avg Val Loss |
|---|---:|---:|
| `spatial_autoencoder_v0.1.0_epoch5.pt`  | 330,310 | 713.357 |
| `spatial_autoencoder_v0.2.0_epoch5.pt`  | 330,310 | 26.677 |
| `spatial_autoencoder_v0.2.0_epoch25.pt` | 330,310 | 9.115 |
| `spatial_autoencoder_v0.2.0_epoch30.pt` | 330,310 | 9.062 |
| `spatial_autoencoder_v3.1.0_epoch25.pt` | 330,314 | 18.742 |
| `spatial_autoencoder_v5.1.0_epoch20.pt` | 330,314 | 6.593 |
| `spatial_autoencoder_v5.1.0_epoch28.pt` | 330,314 | 6.064 |

---

## Family 4 — Spatial Masked Autoencoder

Same conv encoder/decoder backbone as the spatial AE, but trained with input masking and L1 reconstruction loss. Sources: [spatial_encoder.py](app/foundation_models/components/spatial_encoder.py), [spatial_decoder.py](app/foundation_models/components/spatial_decoder.py), trainers under [app/foundation_models/trainers/](app/foundation_models/trainers/).

| Checkpoint | Encoder stages | Params | Decoder out | Notes |
|---|---|---:|---:|---|
| `spatial_masked_autoencoder_v0.1.0_epoch121.pt`              | 1 → 64 → 128 | 265,480 | 1 | MSE loss |
| `spatial_masked_autoencoder_l1_v0.2.0_epoch92.pt`            | 1 → 64 → 128 | 265,480 | 1 | L1 loss |
| `spatial_masked_autoencoder_l1_v0.4.0_epoch77.pt`            | 1 → 32 → 64 → 128 | 330,314 | 1 | L1, deeper encoder; **lowest val loss in family** at 1.792 |
| `spatial_masked_autoencoder_l1_unnormalized_v0.3.0_epoch82.pt` | 3 → 64 → 128 | 267,524 | 1 | 3-channel input, no `normalize`/`denormalize` modules in saved state |

---

## Family 5 — Normalized Masked Autoencoder

`normalized_masked_autoencoder_v1.2.0_epoch69.pt` — **267,528 params**

Defined in [app/foundation_models/components/normalized_masked_spatial_auto_encoder.py](app/foundation_models/components/normalized_masked_spatial_auto_encoder.py).

| Submodule | Params |
|---|---:|
| `normalize` / `denormalize` (1ch each) | 4 |
| `encoder` (3-channel input, two conv stages: 3→64→128) | 135,106 |
| `decoder` (mirrored, single-channel output) | 132,418 |
| **Total** | **267,528** |

Training: `lr=1e-3`, `T_max=300`, train_loss 1.796 at epoch 69, but val losses higher (avg 5.529) — possibly indicating overfit or a domain mismatch between training and validation patches.

---

## Reproducing this inventory

```python
import torch
from pathlib import Path

for path in sorted(Path("checkpoints").rglob("*.pt")):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["model_state_dict"]
    total = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
    print(f"{path}\t{total:,}\t{obj.get('epoch')}\t{obj.get('avg_val_loss'):.5f}")
```
