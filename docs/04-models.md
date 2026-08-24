# 4. Foundation models

Seven neural architectures. All do the same job: **reconstruct the image, and treat what
they get wrong as the anomaly.**

```
app/foundation_models/
  components/    the nn.Module architectures
  trainers/      7 trainers   (trainer_factory.get_trainer)
  inferencers/   5 inferencers (inferencer_factory.get_inferencer)
```

Note the asymmetry: **7 trainers, 5 inferencers.** Two architectures can be trained but not
run. See [9. Known issues](09-known-issues.md).

## The roster

Each carries a Sanskrit codename, and the **backend selects models by codename**, not by
class name.

| Slug | Codename | Kind |
|---|---|---|
| `spatial_autoencoder` | **Pratibimba** प्रतिबिंब | conv AE — the baseline |
| `spatial_masked_autoencoder` | **Antardhana** अंतर्धान | conv MAE, L2 |
| `spatial_masked_autoencoder_l1` | **Tirohita** तिरोहित | conv MAE, L1, heavy masking |
| `spatial_masked_autoencoder_l1_unnormalized` | **Asanskrita** असंस्कृत | conv MAE, loss in raw °C |
| `normalized_masked_autoencoder` | **Drashta** द्रष्टा | conv MAE, explicit mask channels |
| `segformer_mae` | **Chakshu** चक्षु | transformer MAE (thermal) |
| `hyperspectral_segformer_mae` | **Indradhanu** इंद्रधनु | transformer MAE + spectral bottleneck |

Shipped checkpoints: **Chakshu** 406 k params, val 0.2565 @ epoch 495. **Indradhanu**
5.5 M params, val 0.0435 @ epoch 200.

## Why masked reconstruction

The model never sees a labelled anomaly. Instead we hide part of the input and make it
predict the hidden part from context. To do that well it must learn what this kind of
terrain normally looks like.

At inference, anything it *can't* predict from context is, by definition, unlike the
normal it learned. That surprise is the anomaly score.

**This is why every pixel must be reconstructed while hidden.** If a pixel were visible
during its own reconstruction the model would just copy it, the residual would be ~0, and
nothing would ever look anomalous. Hence two-pass complementary masking at inference —
run twice with opposite masks, and take each pixel from the pass where it was hidden.

## Two design axes

**L2 → L1.** The dominant evolution across the roster. L2 punishes large errors quadratically,
so it pushes the model to fit outliers well — and outliers are exactly what we want it to
fail on. L1 is robust to them. Later models are all L1.

**Pixel masking vs token removal.** The conv models zero out pixels and add mask channels.
The SegFormer models physically **delete tokens** before the encoder, so no compute is spent
on hidden regions and no information leaks. Token removal gives a stronger encoder; pixel
masking is simpler and works for CNNs.

## Building blocks

**Conv family** — `SpatialEncoder` / `SpatialDecoder`, stride-2 conv blocks
(`Conv2d → BatchNorm → GELU → Dropout`). Patch size must divide by `2**num_stages`. The
masked variants take **3 channels**: `[pixels, validity_mask, input_mask]`.

**SegFormer family** — hierarchical 4-stage transformer producing a feature pyramid at
H/4, H/8, H/16, H/32. Total stride 32, so **patch size must be divisible by 32**.
- `OverlapPatchEmbedding` — Stage 1 deliberately uses **non-overlapping** patches
  (`k=4, s=4, p=0`) so token removal leaks nothing. Stages 2–4 overlap (`k=3, s=2`).
- `EfficientSelfAttention` — reduces K/V spatially to keep attention affordable.
- `MixFFN` — feed-forward with a depthwise 3×3 instead of positional encodings.
- `SegFormerDecoder` — ends in `pixel_shuffle` rather than bilinear upsampling, for
  point-anomaly fidelity (a bilinear upsample would smear a single hot pixel).

**Indradhanu's spectral bottleneck** — 165 bands is too many to attend over. A
`SpectralCompressor` (`Conv2d 1×1 → BatchNorm2d`) squeezes to 24–32 channels; a
`SpectralDecompressor` (`Conv2d 1×1`, **no norm, no activation**) expands back. The
asymmetry is intentional: the compressor stabilises the encoder's input distribution, while
the decompressor's output must stay unconstrained because it feeds denormalisation.

Its loss is `L1 + λ(t)·SAM`, with λ **ramped in** over the first ~10–20 epochs — SAM's
gradient is unstable early when reconstructions are near-random.

**`PixelNormalize` / `PixelDenormalize`** use *registered buffers*, so normalisation
statistics travel **inside the checkpoint's `state_dict`**. You cannot swap stats without
rebuilding the model.

## Training

```bash
python scripts/train_foundation_model.py configs/<experiment>.json
```

That is the entire entry point — no sweep runner, no CLI flags. `configs/` holds ~22 JSON
files.

`TrainingConfig` blocks: `model_config` (a discriminated union on `model_type`), `data`,
`checkpoint`, `lr_schedule`, `hot_storage`, `wandb`, plus `learning_rate`, `device`,
`resume_from`, `resume_mode`.

Things that surprise people:

- **An "epoch" is a fixed sample budget, not a full pass.** `train_samples_per_epoch` maps
  patch size → count, and models train on **several patch sizes within one epoch**.
- **Only surviving patches count.** Patches under `MIN_VALID_PIXEL_FRACTION = 0.4` valid
  are dropped, and batches where everything is dropped are skipped entirely. Losses are
  re-weighted by kept-count so the epoch mean is sample-weighted.
- **Optimizer is Adam. Always.** There is no SGD path.
- **Checkpoints keep top-K by `avg_val_loss`.** Normalisation buffers and the full config
  are saved alongside the weights.
- **Hot storage** syncs a subset of S3 shards to local disk once and reuses them; without
  it, shards stream from S3 through `pipe: aws s3 cp`.

`resume_mode` is `"resume"` (weights + optimizer + epoch + scheduler) or `"finetune"`
(weights only).

> Several config knobs are **read by nothing**: `warmup_epochs`, `save_to_s3`, and
> `masking_range` on some trainers. Don't assume setting them does anything —
> see [9. Known issues](09-known-issues.md).

## Inference

`InferenceConfig` → `get_inferencer()` → `predict_full_scene()`.

The scene is tiled into overlapping patches; each is reconstructed twice under complementary
masks; results are accumulated and **overlap-averaged**. The validity mask is eroded first
(default kernel 15) so the patch-embedding receptive field never straddles the scene edge.

Reconstruction → anomaly map is then handled by the shared scoring module, covered in
[5. Detectors](05-detectors.md#scoring--turning-residuals-into-a-heatmap).

## Adding a model

1. Architecture in `components/`.
2. Trainer subclassing `FoundationTrainer` — implement `build_model`, `compute_loss`
   (returns `(loss, num_kept)`), `validation_step`.
3. Inferencer subclassing `FoundationInferencer` — implement `build_model`, `infer`.
4. Add the name to `FoundationModelName` and register in **both** factories.
5. Add a capabilities entry in `backend/allotrope/foundation_models/resolver.py`, or the
   product cannot see it.

Step 5 is the one people forget.

---

**Next:** [5. Detectors](05-detectors.md) · [6. Backend](06-backend.md)
