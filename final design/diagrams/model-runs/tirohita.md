# Tirohita — `spatial_masked_autoencoder_l1`

> तिरोहित · *Tirohita* · **hidden** — same masking as Antardhana, but the
> loss penalises bias instead of variance.

Architecturally identical to [Antardhana](antardhana.md) and
[Pratibimba](pratibimba.md). The single difference is the **training loss**:

| | Tirohita | Antardhana | Pratibimba |
|---|---|---|---|
| Training masking | random 50 % | random 50 % | none |
| Training loss | **L1** | MSE | MSE |
| Default scoring | **`L1`** | MSE | MSE |
| Inference path | identical | identical | (canonical) |

| | |
|---|---|
| **Architecture** | `spatial_masked_autoencoder_l1` |
| **`foundation_model_name`** | `spatial_autoencoder` *(routes to the Pratibimba inferencer)* |
| **Sensor family** | thermal |
| **Default patch / stride / batch** | 128 / 64 / 8 |
| **Inferencer** | [`spatial_autoencoder_inferencer.py`](../../../app/foundation_models/inferencers/spatial_autoencoder_inferencer.py) |
| **Model module** | [`spatial_auto_encoder.py`](../../../app/foundation_models/components/spatial_auto_encoder.py) |

---

## Inference path

**Read [Pratibimba](pratibimba.md) §1–§7.** Same `SpatialAutoencoder`
forward, same `SpatialAutoencoderInferencer.infer` two-pass pixel-level
checkerboard, same `predict_full_scene`. Only the loaded checkpoint
file differs.

The resolver entry:

```python
"spatial_masked_autoencoder_l1": ModelCapabilities(
    architecture="spatial_masked_autoencoder_l1",
    foundation_model_name="spatial_autoencoder",
    scoring_methods=("L1", "MSE"),
    default_scoring_method="L1",                     # ← only line that differs from Antardhana
    default_patch_size=128,
    default_stride=64,
    pixel_stats_relpath=_THERMAL_STATS,
),
```

---

## L1 vs MSE — what changes in the score map

The model was *trained* with L1 loss (`mean(|x - x̂|)` per patch), so it
optimises the **median** of the residual distribution, not the mean.
At inference time:

- Sharper edges in the reconstruction. L1's gradient is constant in
  magnitude (`±1`), so big residuals don't dominate gradient flow the
  way they do under MSE; the model doesn't blur over outliers.
- More robust under heavy-tailed noise. A few rogue hot pixels won't
  warp the rest of the patch's reconstruction.
- Default scoring is `L1` to match — `score = mean(|x - x̂|)` over the
  single channel. Switching to MSE at scoring time is allowed but
  not calibrated against this checkpoint's training objective.

For everything mechanical (diagrams, encoder/decoder ladder, scoring
formulas, complexity) defer to [Pratibimba](pratibimba.md).

## File map

Identical to [Pratibimba's](pratibimba.md#file-map). Only the checkpoint
file (under `<MODELS_DIR>/spatial_masked_autoencoder_l1/`) differs.
