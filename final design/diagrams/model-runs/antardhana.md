# Antardhana — `spatial_masked_autoencoder`

> अन्तर्धान · *Antardhana* · **vanishing** — trained on randomly disappeared pixels.

Same architecture as [Pratibimba](pratibimba.md) — `SpatialAutoencoder`,
single thermal channel, 3-stage strided-conv encoder + transposed-conv
decoder. The differences are entirely in **training**, not inference:

| | Antardhana | Pratibimba |
|---|---|---|
| Training masking | random 50 % token mask | none (clean inputs) |
| Training loss | **MSE** | MSE |
| Inference path | **identical** to Pratibimba | identical |
| Default scoring | `MSE` | `MSE` |

| | |
|---|---|
| **Architecture** | `spatial_masked_autoencoder` |
| **`foundation_model_name`** | `spatial_autoencoder` *(routes to the Pratibimba inferencer)* |
| **Sensor family** | thermal |
| **Default patch / stride / batch** | 128 / 64 / 8 |
| **Inferencer** | [`spatial_autoencoder_inferencer.py`](../../../app/foundation_models/inferencers/spatial_autoencoder_inferencer.py) |
| **Model module** | [`spatial_auto_encoder.py`](../../../app/foundation_models/components/spatial_auto_encoder.py) |

---

## Inference path

**Read [Pratibimba](pratibimba.md) §1–§7.** Antardhana resolves to the
exact same `SpatialAutoencoderInferencer` class, the exact same
`SpatialAutoencoder` model class, and the exact same checkpoint shape.
The resolver entry in
[`backend/allotrope/foundation_models/resolver.py`](../../../backend/allotrope/foundation_models/resolver.py)
makes this explicit:

```python
"spatial_masked_autoencoder": ModelCapabilities(
    architecture="spatial_masked_autoencoder",
    foundation_model_name="spatial_autoencoder",     # routes to SpatialAutoencoderInferencer
    scoring_methods=("L1", "MSE"),
    default_scoring_method="MSE",
    default_patch_size=128,
    default_stride=64,
    pixel_stats_relpath=_THERMAL_STATS,
),
```

So: **same forward, same two-pass pixel-level checkerboard, same scoring**.
What differs is *which weights you load* — Antardhana's checkpoint comes
from the `spatial_masked_autoencoder/` folder, where the model was
trained against random 50 % pixel-mask augmentation.

---

## Why a separate codename if the architecture is the same

Different *training distribution*. Pratibimba never saw masked inputs
during training, but the inferencer always feeds it masked inputs. So
Pratibimba is being asked to do something out-of-distribution at
inference time — it works, but it's a generalisation hope. Antardhana
was *trained* on random masking, so the checkerboard masking we use at
inference is much closer to what it saw during gradient descent.

Practical effect: cleaner reconstructions in textured regions, less
"halo" on the high-frequency boundaries between checker cells.

For everything else — the shape of `predict_full_scene`, the two-pass
`infer`, the `forward`, the encoder/decoder ladder, scoring, complexity
— defer to [Pratibimba](pratibimba.md). Diagrams there apply verbatim.

## File map

Identical to [Pratibimba's](pratibimba.md#file-map). Only the loaded
checkpoint file (under `<MODELS_DIR>/spatial_masked_autoencoder/`)
differs.
