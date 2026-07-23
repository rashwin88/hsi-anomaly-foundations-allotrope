# Drashta — `normalized_masked_autoencoder`

> द्रष्टा · *Drashta* · **observer** — sees the masks alongside the data.

Drashta is Asanskrita with the PixelNormalize buffer put back. Same
3-channel mask-stacking trick into a `SpatialEncoder(in_channels + 2, ...)`,
same pixel-level checkerboard two-pass, but the model normalises and
denormalises at the boundary so the encoder sees z-scored thermal values
rather than raw brightness temperatures.

| | |
|---|---|
| **Architecture** | `normalized_masked_autoencoder` |
| **`foundation_model_name`** | `normalized_masked_autoencoder` |
| **Sensor family** | thermal |
| **Encoder input channels** | **3** (1 thermal + validity_mask + input_mask) |
| **Default patch / stride / batch** | 128 / 64 / 8 |
| **Default scoring** | `L1` |
| **PixelNormalize buffer** | **yes** (loaded from `thermal_pixel_consts.json`) |
| **Inferencer** | [`normalized_masked_autoencoder_inferencer.py`](../../../app/foundation_models/inferencers/normalized_masked_autoencoder_inferencer.py) |
| **Model module** | [`normalized_masked_spatial_auto_encoder.py`](../../../app/foundation_models/components/normalized_masked_spatial_auto_encoder.py) |

---

## 1 · Caller's view

Same as
[Indradhanu §1](indradhanu.md#1--callers-view--one-model-in-the-per-model-loop).
Resolver entry:

```python
"normalized_masked_autoencoder": ModelCapabilities(
    architecture="normalized_masked_autoencoder",
    foundation_model_name="normalized_masked_autoencoder",
    scoring_methods=("L1", "MSE"),
    default_scoring_method="L1",
    default_patch_size=128,
    default_stride=64,
    pixel_stats_relpath=_THERMAL_STATS,         # ← only line that differs from Asanskrita
),
```

`pixel_stats_relpath = constants/thermal_pixel_consts.json`. The
inferencer loads `mean=[24.5756...]`, `std=[13.5743...]` at construction
and registers them as buffers on `PixelNormalize` and `PixelDenormalize`
inside the model.

---

## 2 · `predict_full_scene` — patch loop

`NormalizedMaskedAutoencoderInferencer.predict_full_scene` — same
sliding-window structure as Pratibimba's. See
[Pratibimba §2](pratibimba.md#2--predict_full_scene--the-patch-loop)
for the diagram and code.

---

## 3 · Two-pass — `NormalizedMaskedAutoencoderInferencer.infer`

Mechanically identical to Asanskrita's two-pass — the inferencer doesn't
know whether the model normalises internally or not. From
[`normalized_masked_autoencoder_inferencer.py:64`](../../../app/foundation_models/inferencers/normalized_masked_autoencoder_inferencer.py):

```python
def infer(self, tensor, mask):
    # tensor: (B, 1, H, W)   mask: (B, 1, H, W)
    _, _, h, w = tensor.shape
    checker     = self._build_checkerboard(h, w, invert=False)
    checker_inv = 1 - checker

    # Pass 1: keep checker=1 cells visible
    x_hat_1, _ = self.model(tensor, validity_mask=mask, input_mask=checker     * mask)

    # Pass 2: keep checker=0 cells visible (complement)
    x_hat_2, _ = self.model(tensor, validity_mask=mask, input_mask=checker_inv * mask)

    reconstruction = x_hat_1 * checker_inv + x_hat_2 * checker
    return reconstruction * mask
```

Compare to [Asanskrita's `infer`](asanskrita.md#3--two-pass--maskedspatialautoencoderinferencerinfer)
— byte-for-byte the same shape. The split between the two inferencers
is purely so each can pin its model class without strict-load drama.

---

## 4 · `forward` — `NormalizedMaskedSpatialAutoencoder`

Verbatim from
[`normalized_masked_spatial_auto_encoder.py`](../../../app/foundation_models/components/normalized_masked_spatial_auto_encoder.py):

```python
def forward(self, x, validity_mask=None, input_mask=None):
    # x: (B, 1, H, W)   validity_mask, input_mask: (B, 1, H, W)
    x = x * input_mask                            # zero hidden pixels
    if self.normalize is not None:
        x = self.normalize(x)                     # ← Drashta vs Asanskrita
    model_input = torch.cat([x, validity_mask, input_mask], dim=1)
    # model_input: (B, 3, H, W) in z-score space (only the thermal ch)
    z = self.encoder(model_input)                 # (B, 128, H/8, W/8)
    x_hat = self.decoder(z)                        # (B, 1, H, W)
    if self.denormalize is not None:
        x_hat = self.denormalize(x_hat)           # ← Drashta vs Asanskrita
    return x_hat, z
```

```mermaid
sequenceDiagram
    autonumber
    participant M  as NormalizedMaskedSpatialAutoencoder
    participant E  as SpatialEncoder (3 ch in)
    participant D  as SpatialDecoder (1 ch out)

    Note over M: x (B, 1, 128, 128)<br/>validity_mask (B, 1, 128, 128)<br/>input_mask (B, 1, 128, 128)
    M->>M: x = x · input_mask
    M->>M: PixelNormalize · per-channel z-score
    Note over M: only the thermal channel is z-scored;<br/>masks pass through untouched
    M->>M: model_input = cat([x_normalized, validity_mask, input_mask], dim=1)
    Note over M: model_input (B, 3, 128, 128)
    M->>E: encode(model_input)
    Note over E: 3 strided-conv stages<br/>channels [3, 32, 64, 128]
    E-->>M: z (B, 128, 16, 16)
    M->>D: decode(z)
    Note over D: 3 transposed-conv stages<br/>channels [128, 64, 32, 1]
    D-->>M: x_hat (B, 1, 128, 128)<br/>(in z-score space)
    M->>M: PixelDenormalize · multiply by σ, add μ
    Note over M: returns (x_hat, z)
```

**Key subtlety**: `PixelNormalize` runs on `x` *before* the masks are
concatenated, so the masks stay in `[0, 1]` and only the thermal channel
gets z-scored. After decode, `PixelDenormalize` only acts on the
single-channel output — there's nothing to undo on the masks.

---

## 5 · Encoder / decoder structure

Identical to [Asanskrita §5](asanskrita.md#5--encoder--decoder-structure)
— same 3-channel input, same `[3, 32, 64, 128]` encoder progression,
same mirror decoder. The PixelNormalize / PixelDenormalize layers wrap
the whole encoder-decoder stack but don't change the conv ladder.

---

## 6 · Scoring

`L1` by default — same training story as
[Tirohita](tirohita.md#l1-vs-mse--what-changes-in-the-score-map). Available
methods: `{L1, MSE}`. Note that the residual `(x - x̂)` is computed in
**physical brightness-temperature space** (after `PixelDenormalize`),
so the L1 score is in units of K and is directly interpretable.

---

## 7 · Drashta vs Asanskrita

[Asanskrita](asanskrita.md#7--how-asanskrita-differs-from-drashta-in-one-table)
already carries the side-by-side. Drashta is the version *with* the
normalize buffer; the architectural test is whether that buffer earns
its keep on this dataset.

In [`backend/allotrope/foundation_models/resolver.py`](../../../backend/allotrope/foundation_models/resolver.py)
this shows up as one bit of capability metadata
(`pixel_stats_relpath = _THERMAL_STATS` vs `None`) plus two different
`(architecture, model_class, inferencer_class)` triples:

| | model class | inferencer | pixel_stats |
|---|---|---|---|
| Asanskrita | `UnNormalizedSpatialAutoencoder` | `MaskedSpatialAutoencoderInferencer` | `None` |
| Drashta | `NormalizedMaskedSpatialAutoencoder` | `NormalizedMaskedAutoencoderInferencer` | `thermal_pixel_consts.json` |

Same forward graph topology, same inferencer behaviour. The only thing
that changes is whether two `Conv2d`-shaped buffer ops sit on the
encoder/decoder boundaries.

---

## 8 · Constructor in_channels caveat

A subtle gotcha for anyone wiring their own `model_config`: `in_channels`
in this constructor means **the thermal channel count, not the encoder
input channel count**. Inside the constructor:

```python
def __init__(self, in_channels=1, base_channels=32, num_stages=3, ...):
    ...
    self.encoder = SpatialEncoder(in_channels + 2, base_channels, num_stages, ...)
    #                                            ^^^ wires in mask channels
    self.decoder = SpatialDecoder(in_channels, base_channels, num_stages, ...)
```

So passing `in_channels=1` gives a 3-channel encoder input. The
[`resolver._model_config_from_manifest`](../../../backend/allotrope/foundation_models/resolver.py)
takes the `encoder_dims` (3) from the manifest, subtracts 2, and feeds
1 back to the constructor — the inverse arithmetic that keeps the saved
checkpoint and the live class in lockstep.

The same `max(1, encoder_input - 2)` rule applies to Asanskrita.

## File map

| Component | Path |
|---|---|
| Architecture | [`normalized_masked_spatial_auto_encoder.py`](../../../app/foundation_models/components/normalized_masked_spatial_auto_encoder.py) |
| Inferencer | [`normalized_masked_autoencoder_inferencer.py`](../../../app/foundation_models/inferencers/normalized_masked_autoencoder_inferencer.py) |
| Encoder | [`spatial_encoder.py`](../../../app/foundation_models/components/spatial_encoder.py) |
| Decoder | [`spatial_decoder.py`](../../../app/foundation_models/components/spatial_decoder.py) |
| Pixel normalize | [`pixel_normalization.py`](../../../app/foundation_models/components/pixel_normalization.py) |
| Pixel stats file | `app/constants/thermal_pixel_consts.json` |
| Scoring | [`scoring.py`](../../../app/utils/anomaly_detection/scoring.py) |
| Resolver entry | [`resolver.py`](../../../backend/allotrope/foundation_models/resolver.py) |
| Worker recipe | [`_anomaly_scoring_run.py`](../../../backend/allotrope/action_types/_anomaly_scoring_run.py) |
