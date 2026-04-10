# SegFormer MAE Reconstruction Network for Multi-Band Hyperspectral Anomaly Detection

## 1. Motivation

The thermal SegFormer MAE demonstrated that reconstruction-based anomaly detection via masked autoencoders is effective for single-band satellite imagery. However, single-band thermal detection is fundamentally limited to anomalies that manifest thermally — our analysis showed only 28% of ground-truth anomalies produced a thermal signature above +2σ. The remaining 72% require spectral information to detect.

Hyperspectral imagery provides 165 contiguous spectral bands spanning 460–2450nm (after our preprocessing pipeline). Anomalies that are thermally invisible often have distinctive spectral signatures — different materials reflect light differently across wavelengths, even when their temperature is identical to the background. A reconstruction network operating on the full spectral cube can learn the spectral covariance structure of normal backgrounds, and any deviation from this learned structure produces a detectable reconstruction residual across the spectral dimension.

The challenge is scale: the input grows from `(B, 1, H, W)` to `(B, 165, H, W)` — a 165× increase in input dimensionality. Naively applying the thermal architecture would produce a parameter explosion in the patch embedding and decoder layers, and memory consumption would make training impractical. A spectral compression stage, inspired by MNF (Minimum Noise Fraction) but learned end-to-end, addresses this.

### Multi-Sensor Training

A key requirement is mixed-sensor training. PRISMA and EnMAP scenes are preprocessed onto a common 165-band wavelength grid (460–2450nm, 10nm spacing, atmospheric windows excluded). Both sensors produce identical `(165, 128, 128)` patches with identical wavelength arrays. The model trains on mixed shards containing patches from both sensors, learning a sensor-agnostic spectral representation.

### Combined Loss: L1 + Spectral Angle

The thermal model uses L1 loss on masked pixels. For hyperspectral data, we add a **Spectral Angle Mapper (SAM)** loss head. L1 penalises magnitude errors (brightness), while SAM penalises angular errors (spectral shape). Together they ensure the reconstruction preserves both the overall reflectance level and the characteristic spectral signature of each material.

This is critical for anomaly detection: an anomaly may have similar overall brightness to the background but a distinctly different spectral shape. SAM ensures the model learns to reconstruct spectral shapes accurately, making shape deviations detectable as anomalies.


## 2. Architecture Overview

```
Input Patch (B, 165, H, W)
    │
    ▼
Pixel Normalisation (register_buffer: mean[165], std[165])
    │
    ▼
Apply Validity Mask (zeros invalid pixels across all 165 bands)
    │
    ▼
Spectral Compressor: Conv2d(165, D, kernel_size=1)     ← learned MNF
    │  (B, 165, H, W) → (B, D, H, W)  where D ≈ 24–32
    ▼
Random Token Masking (training) / Structured Masking (inference)
    │
    ▼
SegFormer Encoder (4-stage hierarchical transformer, in_channels=D)
    │  outputs: F1, F2, F3, F4 at progressively reduced resolutions
    ▼
SegFormer Decoder (fuse multi-scale features via PixelShuffle)
    │  output: (B, D, H, W) in compressed spectral space
    ▼
Spectral Decompressor: Conv2d(D, 165, kernel_size=1)  ← inverse projection
    │  (B, D, H, W) → (B, 165, H, W)
    ▼
Pixel Denormalisation (register_buffer: mean[165], std[165])
    │
    ▼
Output Reconstruction (B, 165, H, W)
    │
    ├──→ L1 Loss (masked valid pixels, per-band magnitude)
    └──→ SAM Loss (masked valid pixels, spectral angle)
```

The **Spectral Compressor** and **Spectral Decompressor** are the key additions compared to the thermal architecture. They form a learned bottleneck that reduces the spectral dimension before any spatial processing, analogous to MNF but trained end-to-end with the reconstruction objective.


## 3. Pixel Normalisation

### Per-Band Statistics

Unlike thermal data (scalar mean/std), hyperspectral data requires **per-band** normalisation. Each of the 165 wavelength bands has its own reflectance distribution — VNIR bands (460–910nm) have higher reflectance over vegetation than SWIR bands (1460–2450nm), and the distributions vary by land cover type.

The normalisation is:

$$
x_{norm}^{(c)} = \frac{x_{raw}^{(c)} - \mu^{(c)}}{\sigma^{(c)}} \quad \text{for each band } c \in \{1, ..., 165\}
$$

where $\mu^{(c)}$ and $\sigma^{(c)}$ are computed offline from the training data using only valid pixels. These are stored as registered buffers with shape `(1, 165, 1, 1)`:

```python
class PixelNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # mean, std: (165,) arrays → stored as (1, 165, 1, 1) buffers
        self.register_buffer('mean', torch.tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        # x: (B, 165, H, W)
        return (x - self.mean) / self.std
```

After normalisation, each band independently has approximately zero mean and unit variance. The spectral compressor then operates in a well-conditioned space where all bands contribute equally to the learned compression.

### Mask-Then-Normalise Ordering

The same ordering rationale as thermal applies. Validity masking is applied before normalisation:

```python
x = x * validity_mask            # invalid pixels → 0.0 reflectance
x = self.normalize(x)            # 0.0 → ≈ -μ/σ per band (typically -1 to -3)
                                  # real data → [-1, +2] per band
```

Invalid pixels end up at band-specific negative values (roughly $-\mu^{(c)}/\sigma^{(c)}$), clearly distinguishable from valid data. The spectral compressor and encoder can readily identify masked positions from their distinctive spectral signature (all bands at their respective below-range values simultaneously).


## 4. Spectral Compressor & Decompressor

### Motivation: Why Not Feed 165 Channels Directly?

Three problems arise with 165 direct input channels:

1. **Stage 1 Patch Embedding explosion**: `Conv2d(165, embed_dim, K=4, S=4)` has 165 × embed_dim × 16 parameters. At embed_dim=32 this is 84K parameters — tractable but doing heavy spectral compression inside a spatial operation.

2. **Decoder PixelShuffle explosion**: The pre-shuffle conv produces `out_channels × upscale² = 165 × 16 = 2,640` channels. With decoder_dim=256, this single layer has 6.1M parameters.

3. **Memory**: Input tensors at batch size 32 consume 330MB vs 2MB for thermal. Intermediate activations scale similarly.

### Design: Pointwise Spectral Projection

The compressor is a **1×1 convolution** — it operates on each pixel independently, mixing only the spectral dimension:

$$
z^{(d)}_{(i,j)} = \sum_{c=1}^{165} W^{(d,c)} \cdot x^{(c)}_{(i,j)} + b^{(d)} \quad \text{for each compressed channel } d \in \{1, ..., D\}
$$

```python
class SpectralCompressor(nn.Module):
    def __init__(self, in_channels=165, compressed_channels=24):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, 165, H, W) → (B, D, H, W)
        return self.compress(x)
```

This is mathematically equivalent to applying a `(D × 165)` matrix to each pixel's spectrum — exactly like PCA/MNF, but with learned weights optimised for the reconstruction objective rather than variance/SNR ordering. Parameters: `165 × D + D` (weights + bias) ≈ 4,000 for D=24.

The decompressor is the inverse projection:

$$
\hat{x}^{(c)}_{(i,j)} = \sum_{d=1}^{D} V^{(c,d)} \cdot z^{(d)}_{(i,j)} + a^{(c)}
$$

```python
class SpectralDecompressor(nn.Module):
    def __init__(self, compressed_channels=24, out_channels=165):
        super().__init__()
        self.decompress = nn.Conv2d(compressed_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, D, H, W) → (B, 165, H, W)
        return self.decompress(x)
```

### Choosing the Compressed Dimension D

MNF analysis of hyperspectral anomaly detection typically retains 10–30 components. For reconstruction (which is harder than detection — we need to recover all spectral detail, not just discriminative features):

- **D = 24**: Retains ~95% of spectral variance for typical land cover scenes. Compression ratio: 165/24 ≈ 7×. This is the recommended starting point.
- **D = 32**: More conservative, retains ~98%. Minimal quality loss, slight parameter increase.
- **D = 16**: Aggressive compression. May lose fine spectral features needed for narrow absorption line reconstruction.

The dimension D should be configurable in `SegFormerMAEConfig`. Training experiments will determine the optimal value by monitoring SAM loss — if spectral angle error plateaus at a given D, increasing it further provides no benefit.

### Interaction with the Rest of the Network

After compression, the entire SegFormer encoder-decoder operates in D-dimensional spectral space. This means:

- **Stage 1 Patch Embedding**: `Conv2d(D, embed_dims[0], K=4, S=4)` — small and efficient
- **Decoder PixelShuffle**: produces `D × 16` channels — e.g., 24 × 16 = 384 channels (manageable)
- **All attention, FFN, feature fusion**: operates on spatial patterns in D-dimensional spectral space
- **The model never sees 165 channels internally** — only the compressor/decompressor touch the full spectral dimension

### End-to-End Training

The compressor/decompressor weights are trained jointly with the encoder-decoder. The gradients from both the L1 and SAM losses flow back through the decompressor into the encoder-decoder and through the encoder-decoder into the compressor. The compressor learns to retain the spectral information that minimises reconstruction error, while discarding noise and redundancy.

This is superior to fixed PCA/MNF compression because:
- PCA optimises for variance preservation, which may retain high-variance noise
- MNF optimises for SNR ordering, which is ideal for detection but not necessarily for reconstruction
- The learned compressor optimises directly for the training objective (L1 + SAM reconstruction loss)


## 5. Masking Strategy

### Token Geometry

With 128×128 input patches and Stage 1 patch embedding (K=4, S=4, P=0), each patch produces:

$$
\frac{128}{4} \times \frac{128}{4} = 32 \times 32 = 1024 \text{ tokens}
$$

Each token covers a 4×4 spatial block across D compressed spectral channels. Token dimension: `D × 4 × 4`. For D=24, each token is a 384-dimensional vector.

### Pixel Validity to Token Validity

The validity cube is `(165, H, W)` but is pixel-level binary (all 165 bands agree at each pixel after our preprocessing pipeline). We use band 0 as a spatial validity proxy:

```python
spatial_validity = validity_cube[0]  # (H, W), same for all bands
```

Token validity is computed by average-pooling the spatial validity with the same kernel/stride as Stage 1 patch embedding:

```python
token_mask = avg_pool2d(spatial_validity, kernel_size=4, stride=4)
token_valid = (token_mask > 0.5)  # Token valid if >50% of its 4×4 block is valid
```

### Random Masking During Training

From the valid tokens, we randomly select `mask_ratio` (default 50%) as prediction targets. These are **physically removed** from the encoder's input sequence:

```
1024 total tokens
├── ~300 invalid tokens (scene edges, nodata) → kept in sequence as context
├── ~362 visible valid tokens (50% of ~724 valid) → fed to encoder  
└── ~362 masked valid tokens (50% of ~724 valid) → removed, prediction targets
```

Invalid tokens remain in the sequence because they carry useful information: the encoder can learn that "this region has no data" is a spatial boundary condition that helps predict nearby valid tokens.

### Mask Ratio Considerations

With 165 bands, each visible token carries significantly more information than in the thermal case (384-dim vs 16-dim). Spectral channels are also highly correlated (10nm spacing). This means:

- **Spatial reconstruction** may be "too easy" — the model can lean on spectral redundancy
- **Higher mask ratios** (60–75%) may be needed to force genuine spatial reasoning
- **Starting at 50%** for consistency with the two-pass inference strategy, then experimenting upward

This should be configurable via `mask_ratio` in the config.


## 6. SegFormer Encoder

### Stage 1: Non-Overlapping Patch Embedding + Token Removal

The encoder operates on the compressed representation `(B, D, H, W)` where D ≈ 24.

**Patch Embedding**:
$$
Conv2d(D, C_1, K{=}4, S{=}4, P{=}0): \quad (B, D, H, W) \rightarrow (B, C_1, \frac{H}{4}, \frac{W}{4})
$$

Reshaped to token sequence: $(B, \frac{H}{4} \cdot \frac{W}{4}, C_1)$, then LayerNorm.

For 128×128 input: 1024 tokens of dimension $C_1$.

**Non-overlapping patches** (P=0) are critical for MAE:
- Each token's receptive field is exactly its own 4×4 block
- No information leakage between adjacent tokens
- Token removal creates a clean information gap — the encoder cannot infer masked content from overlapping receptive fields

**Token Removal (Gather)**:
```python
# tokens: (B, 1024, C1)
# keep_mask: (B, 1024), 1=keep, 0=remove

# Sort: kept tokens first, removed tokens last
sorted_indices = keep_mask.argsort(dim=1, descending=True)
num_kept = keep_mask.sum(dim=1).max()

# Gather only kept tokens
gather_indices = sorted_indices[:, :num_kept]                    # (B, num_kept)
kept_tokens = torch.gather(tokens, 1, gather_indices.unsqueeze(-1).expand(-1, -1, C1))
# kept_tokens: (B, num_kept, C1) — shorter sequence
```

The encoder's transformer blocks process only `kept_tokens`. Attention cost: $O(num\_kept^2)$ instead of $O(1024^2)$ — approximately 4× savings at 50% masking.

**Transformer Blocks** at Stage 1:

Each block applies:
1. LayerNorm → Efficient Self-Attention → Dropout → Residual Add
2. LayerNorm → MixFFN → Dropout → Residual Add

**ESA at Stage 1** (sparse tokens):
When tokens have been removed, N ≠ H×W. The spatial reduction convolution (which expects a 2D grid) cannot be applied directly. ESA falls back to **full attention** on the sparse token set:

$$
Q = kept\_tokens \cdot W_Q, \quad K = kept\_tokens \cdot W_K, \quad V = kept\_tokens \cdot W_V
$$
$$
Attn = softmax\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

With ~512 kept tokens, full attention is (512² = 262K operations per head) — very efficient.

**MixFFN at Stage 1** (sparse tokens):
The depthwise 3×3 convolution requires a 2D spatial grid. When tokens are sparse, MixFFN skips the depthwise conv and uses only the linear expansion/contraction:

$$
Linear(C_1 \rightarrow C_1 \times E) \rightarrow GELU \rightarrow Linear(C_1 \times E \rightarrow C_1)
$$

No spatial mixing at Stage 1 during sparse processing. Spatial context comes from attention only.

**Token Restoration (Scatter)**:

After Stage 1's transformer blocks, masked tokens are scattered back as zeros:

```python
# kept_tokens: (B, num_kept, C1) — encoded
# gather_indices: (B, num_kept) — original positions

full_tokens = torch.zeros(B, 1024, C1, device=device)
full_tokens.scatter_(1, gather_indices.unsqueeze(-1).expand(-1, -1, C1), kept_tokens)
# full_tokens: (B, 1024, C1) — zeros at masked positions
```

Reshaped to 2D feature map: $(B, C_1, \frac{H}{4}, \frac{W}{4})$

### Stages 2–4: Dense Processing (No Gather/Scatter)

Stages 2–4 operate on the full 2D feature map from the previous stage. **No token removal** — the gather/scatter pattern is exclusive to Stage 1.

The zeros scattered into Stage 1's output are handled naturally by the overlapping patch embeddings at Stages 2–4:

**Stage 2 Patch Embedding**:
$$
Conv2d(C_1, C_2, K{=}3, S{=}2, P{=}1): \quad (B, C_1, \frac{H}{4}, \frac{W}{4}) \rightarrow (B, C_2, \frac{H}{8}, \frac{W}{8})
$$

The stride-2, kernel-3 convolution with padding pools over overlapping 3×3 regions. Most Stage 2 tokens receive contributions from both encoded and zero (masked) Stage 1 tokens. The zeros are "diluted" — at 50% masking, each 3×3 window contains ~4-5 non-zero values, providing sufficient signal.

**ESA at Stages 2–4** (dense tokens):
Full spatial reduction is used because tokens form a complete 2D grid:

$$
K_{reduced}, V_{reduced} = SpatialReduction(K, V, reduction\_ratio=R)
$$

where `SpatialReduction` is a `Conv2d(C, C, K=R, S=R)` that non-overlappingly pools the K/V tokens.

**MixFFN at Stages 2–4** (dense tokens):
Full depthwise 3×3 convolution is applied, providing local spatial context and implicit positional encoding.

### Stage Progression Table

| Stage | Input Resolution | Embed Dim $C_i$ | Heads | Reduction $R$ | Blocks | Token Removal |
|-------|-----------------|-----------------|-------|---------------|--------|---------------|
| 1 | $\frac{H}{4} \times \frac{W}{4}$ | $C_1$ | $h_1$ | 8 | $n_1$ | **Yes** (gather/scatter) |
| 2 | $\frac{H}{8} \times \frac{W}{8}$ | $C_2$ | $h_2$ | 4 | $n_2$ | No (dense) |
| 3 | $\frac{H}{16} \times \frac{W}{16}$ | $C_3$ | $h_3$ | 2 | $n_3$ | No (dense) |
| 4 | $\frac{H}{32} \times \frac{W}{32}$ | $C_4$ | $h_4$ | 1 | $n_4$ | No (dense) |

Typical B0 config: $C = [32, 64, 160, 256]$, $h = [1, 2, 5, 8]$, $n = [2, 2, 2, 2]$.

The encoder outputs four feature maps: $F_1, F_2, F_3, F_4$ at resolutions $\frac{H}{4}$ through $\frac{H}{32}$.


## 7. SegFormer Decoder

### Multi-Scale Feature Fusion

The decoder fuses the four encoder feature maps into a single full-resolution output in compressed spectral space $(B, D, H, W)$.

**Step 1 — Channel unification**: Each feature map $F_i$ is projected to a common decoder dimension $C_{dec}$:

$$
F_i' = Conv2d_{1 \times 1}(C_i \rightarrow C_{dec})(F_i) \quad \text{for } i \in \{1,2,3,4\}
$$

**Step 2 — Spatial unification**: All feature maps are upsampled to Stage 1 resolution $(\frac{H}{4}, \frac{W}{4})$ via bilinear interpolation:

$$
F_i'' = Upsample\left(F_i', size=\left(\frac{H}{4}, \frac{W}{4}\right)\right)
$$

**Step 3 — Concatenation**:

$$
F_{cat} = Concat(F_1'', F_2'', F_3'', F_4'') \quad \Rightarrow \quad (B, 4 \cdot C_{dec}, \frac{H}{4}, \frac{W}{4})
$$

**Step 4 — Fusion convolution**:

$$
F_{fused} = GELU(Conv2d_{1 \times 1}(4 \cdot C_{dec} \rightarrow C_{dec})(F_{cat}))
$$

**Step 5 — Refinement at quarter resolution**:

$$
F_{refined} = Conv2d_{3 \times 3}(C_{dec} \rightarrow C_{dec})(GELU(F_{fused}))
$$

**Step 6 — PixelShuffle upsampling to full resolution**:

$$
Conv2d_{3 \times 3}(C_{dec} \rightarrow D \times 4^2)(F_{refined}) \quad \Rightarrow \quad (B, D \times 16, \frac{H}{4}, \frac{W}{4})
$$
$$
PixelShuffle(4): \quad (B, D \times 16, \frac{H}{4}, \frac{W}{4}) \rightarrow (B, D, H, W)
$$

For D=24: the pre-shuffle conv produces 24 × 16 = 384 channels — manageable. Without the spectral compressor, this would be 165 × 16 = 2,640 channels.

### Why PixelShuffle, Not Bilinear Upsampling

PixelShuffle rearranges channels into spatial positions — each output pixel is independently predicted from a unique set of channel values. Bilinear interpolation averages neighboring pixels, which:
- **Blurs point anomalies**: A single anomalous pixel gets diluted by its neighbors
- **Creates false gradients**: The transition from normal to anomalous is artificially smoothed
- **Reduces reconstruction error at anomaly locations**: The detector becomes less sensitive

For anomaly detection, PixelShuffle is essential: the reconstruction error at an anomalous pixel reflects only the model's ability to predict *that specific pixel*, not an average of it and its neighbors.


## 8. Spectral Decompressor

After the decoder produces $(B, D, H, W)$ in compressed spectral space, the decompressor projects back to the full 165-band representation:

$$
\hat{x}_{norm} = Conv2d_{1 \times 1}(D \rightarrow 165)(z_{decoded})
$$

This is the inverse of the compressor — a `(165 × D)` learned matrix applied per-pixel. The output is in normalised spectral space.

The decompressor has no activation function — the output must be free to represent any value in the normalised range, including negative values (reflectance below the per-band mean).

### Denormalisation

The denormalisation module converts back to physical reflectance:

$$
\hat{x}_{raw}^{(c)} = \hat{x}_{norm}^{(c)} \cdot \sigma^{(c)} + \mu^{(c)}
$$

Using the same registered buffers as the normaliser.


## 9. Loss Functions

### Loss Computation Domain

Both losses are computed in **raw reflectance space** (after denormalisation). The model normalises internally for stable training, but the loss compares the denormalised output `x_hat` against the raw input `x`. This means:

- Loss values are in **physical reflectance units** — directly interpretable (e.g. L1=0.005 means 0.5% reflectance error per band)
- Bands with higher natural variance (e.g. vegetation red edge) contribute more to L1, which is appropriate since they carry more information
- Anomaly detection inference operates in raw space — training loss matches inference scores
- No artificial equalisation of bands that would suppress diagnostically important SWIR signals

Note: normalisation still happens inside the model for numerical stability, but the loss sees denormalised outputs.

### L1 Reconstruction Loss

Per-pixel, per-band absolute error at masked valid positions:

$$
\mathcal{L}_{L1} = \frac{1}{|M| \cdot C} \sum_{(i,j) \in M} \sum_{c=1}^{C} \left| x^{(c)}_{(i,j)} - \hat{x}^{(c)}_{(i,j)} \right|
$$

where $M$ is the set of masked valid pixel positions and $C = 165$ is the number of bands.

L1 is preferred over MSE for the same reasons as in the thermal model: it is less sensitive to outliers (anomalies in the training data) and produces more robust gradients. MSE would disproportionately penalise high-error pixels, which are often the unlabelled anomalies we want to ignore during training.

### Spectral Angle Mapper (SAM) Loss

SAM measures the angular distance between the predicted and true spectra at each masked valid pixel:

$$
SAM_{(i,j)} = \arccos\left( \frac{\mathbf{x}_{(i,j)} \cdot \hat{\mathbf{x}}_{(i,j)}}{\|\mathbf{x}_{(i,j)}\| \cdot \|\hat{\mathbf{x}}_{(i,j)}\| + \epsilon} \right)
$$

where $\mathbf{x}_{(i,j)}$ and $\hat{\mathbf{x}}_{(i,j)}$ are the 165-dimensional spectral vectors at pixel $(i,j)$, and $\epsilon$ is a small constant (e.g., $10^{-8}$) for numerical stability.

The SAM loss is averaged over masked valid positions:

$$
\mathcal{L}_{SAM} = \frac{1}{|M|} \sum_{(i,j) \in M} SAM_{(i,j)}
$$

SAM values are in radians: 0 = identical spectral shape, $\frac{\pi}{2}$ = orthogonal spectra. For normal reconstruction, SAM is typically 0.01–0.10 radians (0.6°–5.7°).
c**Why SAM matters:**

Consider two failure modes:
1. **Correct shape, wrong brightness**: The model predicts a spectrum with the right relative band ratios but scaled too high or low. L1 catches this; SAM does not (angle is preserved under scaling).
2. **Correct brightness, wrong shape**: The model predicts a spectrum with the right mean reflectance but distorted band ratios. SAM catches this; L1 may not penalise it enough if the per-band errors partially cancel.

The combined loss ensures both failure modes are penalised.

### Combined Loss

$$
\mathcal{L}_{total} = \mathcal{L}_{L1} + \lambda(t) \cdot \mathcal{L}_{SAM}
$$

where $\lambda(t)$ is a time-varying weight that ramps from 0 to $\lambda_{max}$ over the first $T_{ramp}$ epochs:

$$
\lambda(t) = \lambda_{max} \cdot \min\left(1, \frac{t}{T_{ramp}}\right)
$$

**Why a ramp?** Early in training, the model is still learning basic spectral structure — weights are random, reconstructions are poor, and spectral angles are near-random. SAM gradients at this stage are noisy and can destabilise the optimiser. The L1 loss should dominate initially to establish correct per-band magnitudes. Once the model produces roughly correct spectra (after $T_{ramp}$ epochs), SAM becomes meaningful and drives the model to refine spectral shape fidelity.

**Recommended values:**
- $\lambda_{max} = 1.0$ — gives roughly equal weighting once fully ramped (L1 and SAM are typically in similar magnitude ranges 0.01–0.10)
- $T_{ramp} = 20$ epochs — allows ~4% of a 500-epoch run for the model to learn basic structure before SAM kicks in fully

Both $\lambda_{max}$ and $T_{ramp}$ are configurable. Setting $T_{ramp} = 0$ disables the ramp (immediate full SAM weighting). Setting $\lambda_{max} = 0$ disables SAM entirely (L1 only, equivalent to thermal model).

### Trimmed Loss

The trimmed loss strategy from the thermal model can be applied to the combined loss. Per-pixel combined losses are computed, sorted, and the top $\tau\%$ are discarded:

1. Compute $\ell_{(i,j)} = L1_{(i,j)} + \lambda \cdot SAM_{(i,j)}$ for each masked valid pixel
2. Sort $\ell$ ascending
3. Keep only the bottom $(1 - \tau)\%$
4. Average the kept losses

**Trimming is disabled by default** (`trim_fraction=0.0`).

**Caution from thermal experiments:** 5% trimming on thermal data degraded AUC from 0.91 to 0.53. The thermally subtle anomalies (mean z-score 0.73) were not in the top 5% of per-pixel losses — hard background pixels (complex terrain boundaries, mixed land cover transitions) produced higher reconstruction errors than the anomalies. Trimming discarded the hard-but-normal pixels that the model most needed to learn from, while the actual anomalies remained in the kept set, contaminating the training signal.

**Implications for hyperspectral:** The situation may differ. Hyperspectral anomalies with distinctive spectral signatures across 165 bands could produce genuinely extreme reconstruction errors that dominate the loss distribution — unlike thermal anomalies which were spectrally subtle. A material anomaly (e.g., exposed metal vs vegetation) may produce high L1 *and* high SAM simultaneously, placing it firmly in the trimmed set. However, this must be validated empirically before enabling. The risk of trimming away hard-but-normal background pixels remains.

**Recommendation:** Start with `trim_fraction=0.0`. After initial training, analyse the per-pixel loss distribution on validation data with known anomaly labels. If anomaly pixels consistently fall in the top 1-5% of losses, trimming may help. If they don't (as in the thermal case), trimming will hurt.

### Validity Masking in the Loss

The loss mask combines the MAE prediction mask with the spatial validity:

$$
loss\_mask_{(i,j)} = prediction\_mask_{(i,j)} \times validity_{(i,j)}
$$

Only pixels that are both:
- **Masked** (part of the MAE prediction task — the model predicted from context, not from itself)
- **Valid** (genuine surface reflectance, not nodata/cloud/haze)

contribute to the loss. This is simpler than the thermal case because our preprocessing pipeline ensures validity is strictly binary at the pixel level — all 165 bands agree.


## 10. Training Configuration

### Optimiser

AdamW with decoupled weight decay:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda_{wd} \theta_t \right)
$$

Recommended: $\eta = 1 \times 10^{-4}$, $\lambda_{wd} = 1 \times 10^{-4}$.

### Learning Rate Schedule

Cosine annealing with linear warmup:
- **Warmup** (epochs 1–5): linear ramp from $\eta_{min}$ to $\eta_{max}$
- **Cosine decay** (epochs 6–N):

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{total} - T_{warmup}} \cdot \pi\right)\right)
$$

Recommended: $\eta_{max} = 1 \times 10^{-4}$, $\eta_{min} = 1 \times 10^{-6}$, warmup = 5 epochs.

### Batch Size and Gradient Accumulation

Due to 165× larger input tensors, batch size must be smaller than thermal:
- **Thermal**: 32–64 per batch
- **Hyperspectral**: 8 per mini-batch

With only 8 samples per mini-batch, gradient noise is high and BatchNorm statistics are unstable. **Gradient accumulation** addresses this by accumulating gradients over N mini-batches before updating weights:

```
gradient_accumulation_steps = 4
effective_batch_size = 8 × 4 = 32   (matches thermal)

Mini-batch 1:  loss₁ / 4  →  backward()     gradients accumulate
Mini-batch 2:  loss₂ / 4  →  backward()     gradients accumulate
Mini-batch 3:  loss₃ / 4  →  backward()     gradients accumulate
Mini-batch 4:  loss₄ / 4  →  backward()     gradients accumulate
               optimizer.step()              NOW update weights
               optimizer.zero_grad()         reset for next cycle
```

Loss is divided by `accumulation_steps` before `backward()` so the gradient magnitude matches what a single batch of size 32 would produce. This keeps the learning rate meaningful regardless of the accumulation setting.

Configurable via `gradient_accumulation_steps` in the data config. Setting to 1 disables accumulation (falls back to base trainer behaviour).

### Dropout

Rate 0.3, applied within transformer blocks (after ESA and within MixFFN). Same rationale as thermal: forces redundant representations, prevents over-reliance on any single feature.

### Multi-Scale Training

| Patch Size | Train Samples/Epoch | Test Samples/Epoch | Rationale |
|-----------|--------------------|--------------------|-----------|
| 128×128 | 30,000 | 5,000 | Primary training scale |

Unlike thermal (which trains at 4 scales), hyperspectral initially trains at a single scale. The much larger per-patch memory footprint makes multi-scale training expensive. Once the single-scale model is validated, additional scales can be added.


## 11. Inference Pipeline

### Two-Pass Reconstruction

Identical to the thermal strategy:

**Pass 1**: Remove checkerboard "black" tokens, encode "white" tokens, reconstruct "black" tokens.

**Pass 2**: Remove "white" tokens, encode "black" tokens, reconstruct "white" tokens.

**Combine**: Each pixel's reconstruction comes from the pass where its token was masked.

The compressor and decompressor run once per pass (they are pointwise operations), so the cost is:
- 2× spectral compression (165 → D)
- 2× encoder-decoder forward (in D-dimensional space)
- 2× spectral decompression (D → 165)

### Full-Scene Reconstruction

Sliding window with 50% overlap:

1. Extract patches via `PatchPlanGenerator` (stride = 64 for 128×128 patches)
2. Batch patches for GPU efficiency (batch size limited by memory)
3. Filter: skip patches with <10% valid pixels
4. Two-pass inference per batch
5. Scatter reconstructions into full scene
6. Overlap-average in overlapping regions
7. Erode mask at scene edges (7-pixel erosion)

### Anomaly Score Map

Two complementary anomaly scores per pixel:

**L1 residual** (magnitude anomaly):
$$
A_{L1}(i,j) = \frac{1}{C} \sum_{c=1}^{C} |x^{(c)}_{(i,j)} - \hat{x}^{(c)}_{(i,j)}|
$$

**SAM residual** (shape anomaly):
$$
A_{SAM}(i,j) = \arccos\left( \frac{\mathbf{x}_{(i,j)} \cdot \hat{\mathbf{x}}_{(i,j)}}{\|\mathbf{x}_{(i,j)}\| \cdot \|\hat{\mathbf{x}}_{(i,j)}\|} \right)
$$

These can be used independently or combined. SAM is particularly valuable for detecting materials with distinctive spectral signatures but normal overall reflectance levels.


## 12. Configuration

### SegFormerMAEConfig (Hyperspectral Extensions)

```python
class SegFormerHyperspectralMAEConfig(BaseModel):
    # Spectral compression
    in_channels: int = 165                  # Common grid band count
    compressed_channels: int = 24           # Learned spectral compression target
    
    # Encoder (same structure as thermal)
    embed_dims: list[int] = [32, 64, 160, 256]
    num_heads: list[int] = [1, 2, 5, 8]
    reduction_ratios: list[int] = [8, 4, 2, 1]
    num_blocks: list[int] = [2, 2, 2, 2]
    
    # Decoder
    decoder_dim: int = 256
    expansion_ratio: int = 4
    drop_rate: float = 0.3
    
    # MAE
    mask_ratio: float = 0.5
    trim_fraction: float = 0.0
    erosion_kernel_size: int = 1            # Validity mask erosion during training (1=minimal)
    
    # Loss
    sam_weight: float = 1.0                 # λ_max for SAM loss
    sam_ramp_epochs: int = 20               # T_ramp: epochs to linearly ramp SAM weight from 0 → λ_max

# Data config (shared across all trainers):
    batch_size: int = 8
    gradient_accumulation_steps: int = 4    # Effective batch = 8 × 4 = 32
    pixel_stats_path: str                   # Path to JSON with per-band mean/std (165 values each)
```

### Erosion Kernel Size

The validity mask is eroded at patch boundaries to exclude pixels whose OPE receptive fields overlap with invalid regions. Different settings for training vs inference:

| Context | Setting | Value | Rationale |
|---------|---------|-------|-----------|
| **Training** | `model_config.erosion_kernel_size` | 1 | Minimal erosion — use maximum training data |
| **Inference** | `InferenceConfig.erosion_kernel_size` | 15 | Conservative — avoid edge artifacts in anomaly maps |

### Loss Value Reference

Loss is computed in **raw reflectance space**. L1 is in reflectance units; SAM is in radians.

| Phase | L1 | SAM | λ | Combined |
|-------|-----|-----|---|----------|
| Epoch 0 | ~0.08–0.10 | ~0.3–0.4 rad (20–25°) | 0.0 | ~0.08–0.10 |
| Epoch 10 | ~0.03–0.05 | ~0.15–0.25 rad (10–15°) | 0.5 | ~0.10–0.15 |
| Epoch 20 | ~0.02–0.03 | ~0.08–0.15 rad (5–8°) | 1.0 | ~0.10–0.15 |
| Epoch 100 | ~0.005–0.015 | ~0.04–0.08 rad (2–5°) | 1.0 | ~0.04–0.08 |
| Epoch 500 | ~0.003–0.008 | ~0.02–0.05 rad (1–3°) | 1.0 | ~0.02–0.05 |


## 13. Summary of Key Design Decisions

| Decision | Choice | Rationale |
|---------|--------|-----------|
| Spectral compression | Learned 1×1 Conv (MNF-inspired) | Reduces 165→24 channels; trained end-to-end for reconstruction, not variance |
| Compression dimension | D=24 (configurable) | Retains ~95% spectral variance; keeps encoder/decoder parameter counts manageable |
| Input normalisation | Per-band z-score (165 means, 165 stds) | Each band contributes equally to compression and loss |
| Mask-then-normalise | Same as thermal | Masked pixels get distinctive per-band below-range values |
| Token masking | MAE with physical token removal at Stage 1 only | Stages 2–4 process dense features (overlapping convolutions dilute zeros) |
| Loss domain | Raw reflectance space | Interpretable physical units; matches inference; no artificial band equalisation |
| Loss function | L1 + λ·SAM (combined, trimmed) | L1 captures magnitude; SAM captures spectral shape; trimming rejects anomalies |
| SAM weight λ | 1.0 max, linearly ramped over 20 epochs | Early training needs L1 stability; SAM gradients are noisy before basic structure is learned |
| Decoder upsampling | PixelShuffle(4) | No blurring; each pixel independently predicted; critical for point anomaly detection |
| Batch size | 8 mini-batch × 4 accumulation = 32 effective | Memory constraint from 165-channel input; accumulation matches thermal's batch=32 |
| Erosion | Training=1, Inference=15 | Training: maximise data. Inference: conservative boundaries |
| Mask ratio | 50% (configurable, may increase) | Matches two-pass inference; spectral redundancy may warrant 60–75% |
| Multi-sensor | Common 165-band grid | PRISMA and EnMAP produce identical tensor shapes; model is sensor-agnostic |
| Anomaly scoring | Dual: L1 residual + SAM residual | Shape anomalies (SAM) vs magnitude anomalies (L1) — complementary signals |
