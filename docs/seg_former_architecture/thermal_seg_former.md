# SegFormer MAE Reconstruction Network for Single-Band Thermal Anomaly Detection

## 1. Motivation

The core task is anomaly detection via reconstruction error. We train a network to reconstruct normal background thermal patterns. At inference, pixels the network cannot reconstruct well — those with high reconstruction error — are flagged as anomalous. The underlying assumption is that the network learns what "normal" looks like, and anything deviating from normality produces a detectable reconstruction residual.

Our earlier spatial autoencoder demonstrated that this approach works for single-band Landsat thermal data. However, we identified a fundamental train-inference mismatch: the autoencoder is trained on complete patches but evaluated using checkerboard masking, where alternating cells are nulled before the forward pass. The model never sees masked input during training, yet must handle it gracefully at inference.

The Masked Autoencoder (MAE) framework eliminates this mismatch. During training, a random subset of the input is masked and the model learns to predict the masked regions from the visible context. At inference, the same masking operation is applied — the model performs the exact task it was trained for.

Additionally, MAE provides a natural form of data augmentation. Each time the same patch is presented during training, a different random mask is applied, creating a different prediction task. The model cannot memorise specific patches because the task changes every time. This directly addresses the overfitting problem observed with scene-level train-validation splits, where the standard autoencoder memorised training scenes within 10–15 epochs.


## 2. Architecture Overview

The network consists of five major components arranged sequentially:

```
Input Patch (1, H, W)
    │
    ▼
Pixel Normalisation (register_buffer: mean, std)
    │
    ▼
Random Masking (training) / Structured Masking (inference)
    │
    ▼
SegFormer Encoder (4-stage hierarchical transformer)
    │  outputs: F1, F2, F3, F4 at progressively reduced resolutions
    ▼
MLP Reconstruction Decoder (fuse multi-scale features)
    │
    ▼
Pixel Denormalisation (register_buffer: mean, std)
    │
    ▼
Output Reconstruction (1, H, W)
```

For single-band thermal input, the Spectral Compressor described in the original multi-band architecture is bypassed — there is only one input channel, so no spectral compression is needed. The `Conv2d` with kernel size 1 that would map $C_{in} \rightarrow C_{compressed}$ is unnecessary when $C_{in} = 1$. The input passes directly to the masking stage. When the architecture is later applied to hyperspectral data (176+ bands), the Spectral Compressor is reintroduced.


## 3. Pixel Normalisation

### Why Normalisation Matters

Raw thermal data arrives in degrees Celsius, typically ranging from 15°C to 55°C for the Indian subcontinent. Without normalisation, the network must learn the absolute temperature scale from scratch — a waste of model capacity. More critically, normalisation determines the behaviour of masked pixels during the masking stage.

### Implementation

We use fixed dataset-level statistics stored as registered buffers:

$$
x_{norm} = \frac{x_{raw} - \mu_{dataset}}{\sigma_{dataset}}
$$

where $\mu_{dataset}$ and $\sigma_{dataset}$ are computed once, offline, from the training data using only valid (cloud-free, non-fill) pixels.

```python
class PixelNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std
```

The `register_buffer` call is critical. These tensors:
- Are **not** learnable parameters — no gradient, no optimizer update
- **Are** saved in the model's `state_dict` — serialised with every checkpoint
- **Are** automatically moved to the correct device via `.to(device)`
- **Are** restored on `load_state_dict` — inference uses the exact same constants as training

After normalisation, the data lives in approximately the $[-2, +2]$ range with zero mean and unit variance. This is the regime where GELU activations, BatchNorm layers, and attention mechanisms operate most effectively.

### The Mask-Then-Normalise Ordering

The order of masking and normalisation is important and non-obvious. We apply the validity mask **before** normalisation:

```python
def forward(self, x, mask=None):
    if mask is not None:
        x = x * mask            # masked pixels → 0°C in raw space
    x = self.normalize(x)       # 0°C → ≈ -2.3 in normalised space
                                # real data → [-0.7, +1.7]
```

This ordering is chosen because of how masked pixels interact with the encoder. When masking is applied in raw space, invalid pixels become 0°C. After normalisation, these pixels land at approximately $\frac{0 - 28.5}{12.3} \approx -2.3$ — clearly outside the typical normalised data range of $[-0.7, +1.7]$. The encoder can readily distinguish masked pixels from real data based on their distinctive value.

If we instead normalised first and then masked, the masked pixels would land at 0.0 in normalised space — the dataset mean. The encoder would see these pixels as perfectly normal mean-temperature pixels, indistinguishable from genuine data. This contamination degrades both reconstruction quality and anomaly detection performance. Empirically, the mask-then-normalise ordering produced substantially lower validation loss (avg_val 6.9 vs 18.6) and better anomaly AUC.

### Denormalisation

The decoder's output is in normalised space. A symmetric denormalisation module converts back to degrees Celsius:

$$
x_{raw} = x_{norm} \cdot \sigma_{dataset} + \mu_{dataset}
$$

The loss can be computed in either space — normalised space avoids any scale-dependent artefacts and is slightly cleaner, but denormalised space produces MSE values in interpretable physical units (°C²).


## 4. Masking Strategy

### During Training: Random Token Masking (MAE-style)

The SegFormer encoder patchifies the input into non-overlapping spatial tokens before processing. For single-band thermal input at $128 \times 128$ with a patch size of $4 \times 4$, this produces $32 \times 32 = 1024$ tokens, each of dimension $1 \times 4 \times 4 = 16$.

We randomly select a fraction of these tokens to mask. The masked tokens are **removed** from the encoder's input sequence entirely — they are not replaced with zeros or a learnable `[MASK]` token. The encoder only processes the visible tokens.

```
1024 tokens total
    │
    ├── 256 visible tokens (25%) → fed to encoder
    │
    └── 768 masked tokens (75%) → removed, not processed by encoder
```

This true token removal is the critical distinction between MAE and simpler masking approaches (e.g., zeroing pixels in a convolutional network). In a convolutional autoencoder, masked pixels must remain in the spatial grid because convolutions operate on fixed grids — the best we can do is set them to a distinctive value and hope the encoder learns to ignore them. In a transformer, we can literally remove the masked tokens from the sequence. The encoder never sees them, cannot learn shortcuts from them, and must build its representation entirely from genuine context.

#### Why the Model Cannot Memorise

With a standard autoencoder, presenting the same training patch $N$ times produces $N$ identical training signals — same input, same target, same gradients. The model memorises the patch after sufficient repetitions.

With MAE, the same patch presented $N$ times produces $N$ **different** training tasks. Each presentation uses a different random mask, so the model sees a different 25% of the patch and must predict a different 75%. The number of unique masks for 1024 tokens with 75% masking is $\binom{1024}{768}$ — combinatorially vast. Over the entire training run, the model never sees the same task twice. It must learn spatial thermal relationships ("when the river is 22°C and the adjacent bank is 35°C, the transition zone between them follows a characteristic gradient") rather than memorise specific patches.

This property directly addresses the overfitting observed with scene-level splits, where the standard autoencoder's training loss dropped to 1.9 while validation remained at 25 after 66 epochs. With MAE, the effective training diversity is multiplied by the number of unique masks, making 100,000 patches behave like millions of unique training examples.

#### Masking Ratio

We use a masking ratio of 50% to match the two-pass checkerboard inference strategy. During inference, each pass masks 50% of the tokens (the checkerboard pattern), so the model should be trained with the same ratio for consistency.

If the masking ratio during training differs substantially from inference (e.g., train at 75%, infer at 50%), the encoder receives a different density of context tokens than it was trained on. While transformers are somewhat robust to this, minimising the train-inference gap is preferable.

#### Implementation

```python
def random_mask(num_tokens, mask_ratio, device):
    """
    Generate a random binary mask for token removal.

    Returns:
        visible_indices: (num_visible,) — indices of tokens to keep
        masked_indices:  (num_masked,)  — indices of tokens to predict
    """
    num_masked = int(num_tokens * mask_ratio)
    noise = torch.rand(num_tokens, device=device)
    sorted_indices = noise.argsort()

    masked_indices = sorted_indices[:num_masked]
    visible_indices = sorted_indices[num_masked:]

    return visible_indices.sort().values, masked_indices.sort().values
```

### During Inference: Structured Checkerboard Masking

At inference, we replace the random mask with a deterministic checkerboard pattern. This ensures every token receives a reconstruction from context, and the two-pass strategy covers all tokens.

**Pass 1:** Mask the "black" cells of the checkerboard. The encoder processes only the "white" cells. The decoder reconstructs the "black" cells from "white" cell context.

**Pass 2:** Mask the "white" cells. The encoder processes only the "black" cells. The decoder reconstructs the "white" cells from "black" cell context.

**Combination:** Each token's final reconstruction comes from the pass where it was masked — ensuring every pixel was predicted from context, never from itself.

```python
def checkerboard_mask(h_tokens, w_tokens, cell_size, invert=False):
    """
    Generate a structured checkerboard mask over the token grid.

    cell_size controls the granularity: cell_size=1 masks individual tokens,
    cell_size=2 masks 2x2 blocks of tokens, etc.

    Returns a binary mask of shape (h_tokens, w_tokens) where 1 = visible, 0 = masked.
    """
    rows = torch.arange(h_tokens) // cell_size
    cols = torch.arange(w_tokens) // cell_size
    grid = (rows[:, None] + cols[None, :]) % 2
    if invert:
        grid = 1 - grid
    return grid
```

The cell size parameter controls the spatial scale of masking. For point anomalies spanning 1–3 pixels, cell size 1 (individual tokens) is appropriate. For extended anomalies spanning 5–10 pixels, a larger cell size (2–4 tokens, corresponding to 8–16 pixels) ensures the anomaly can fall entirely within a masked block and cannot be reconstructed from itself.


## 5. SegFormer Encoder

### Overlap Patch Embedding

The first stage transforms the spatial input into a sequence of token embeddings. For single-band thermal input, $C_{in} = 1$.

A `Conv2d` with kernel size $K=7$, stride $S=4$, and padding $P=3$ maps the input:

$$
x[(B, 1, H, W)] \rightarrow x'[(B, C_1, \frac{H}{4}, \frac{W}{4})]
$$

Each output token covers a $7 \times 7$ receptive field with $4 \times 4$ stride — the overlap is intentional. Overlapping patch embeddings preserve local continuity across token boundaries, unlike ViT's non-overlapping patches which create hard discontinuities.

The output is reshaped from spatial $(B, C_1, H', W')$ to sequence form $(B, H' \times W', C_1)$ and layer-normalised.

For a $128 \times 128$ input with stride 4, this produces $32 \times 32 = 1024$ tokens.

### Token Removal (MAE)

After patchification, the masking is applied. The randomly selected visible tokens are gathered; the masked tokens are discarded:

```python
# tokens: (B, 1024, C1)
# visible_indices: (num_visible,) e.g. 512 indices for 50% masking

visible_tokens = tokens[:, visible_indices, :]  # (B, 512, C1)
# Only visible_tokens enter the encoder
```

The encoder processes a shorter sequence — 512 tokens instead of 1024. This is not only principled (the encoder never sees masked tokens) but also computationally efficient (attention cost scales quadratically with sequence length, so halving the sequence reduces cost by 4×).

### Multi-Head Efficient Self Attention (ESA)

The efficient self-attention mechanism operates on the visible tokens only. The full mechanics are preserved from the base architecture:

1. Compute queries $Q$ from all visible tokens at full resolution
2. Spatially reduce the visible tokens by factor $R$ using a strided convolution to produce a reduced token set
3. Compute keys $K$ and values $V$ from the reduced set
4. Standard multi-head attention between full-resolution queries and reduced keys/values

$$
Q[(B, N_{vis}, C)] = x_{vis} \times W_Q^T
$$

For spatial reduction, the visible tokens must be placed back into a 2D grid (with gaps where masked tokens were removed) before the strided convolution. The gaps can be filled with zeros — the convolution's spatial reduction will pool over them, and since only a small fraction of any pooling window is zero (50% masking ratio with $R=8$ reduction means each pooling window still contains several real tokens), the impact is minimal.

Alternatively, the visible tokens can be scattered into a full-sized grid, the reduction convolution applied, and the resulting reduced tokens gathered. This is slightly more expensive but avoids any contamination from zero-filled gaps.

The reduction ratios per stage are:

| Stage | Spatial Resolution | Channel Dim $C_i$ | Reduction Ratio $R$ | Transformer Blocks |
|-------|-------------------|-------------------|---------------------|-------------------|
| 1 | $\frac{H}{4} \times \frac{W}{4}$ | $C_1 = 64$ | 8 | 2 |
| 2 | $\frac{H}{8} \times \frac{W}{8}$ | $C_2 = 128$ | 4 | 2 |
| 3 | $\frac{H}{16} \times \frac{W}{16}$ | $C_3 = 320$ | 2 | 2 |
| 4 | $\frac{H}{32} \times \frac{W}{32}$ | $C_4 = 512$ | 1 | 2 |

Early stages have the most tokens and need aggressive spatial reduction in ESA. Later stages have fewer tokens and can afford full attention ($R=1$).

### Mix-FFN

After each ESA block, the Mix-FFN injects spatial information via a depthwise convolution sandwiched between two linear projections:

1. Linear expansion: $C_i \rightarrow C_i \times E$ (expansion ratio $E = 4$)
2. Reshape to spatial form
3. Depthwise convolution ($K=3, S=1, P=1$, groups $= C_i \times E$): each channel independently learns a $3 \times 3$ spatial pattern
4. Reshape back to token form
5. GELU activation
6. Linear projection: $C_i \times E \rightarrow C_i$
7. Residual connection

The depthwise convolution is what gives SegFormer positional awareness without explicit positional encodings. Each channel learns its own spatial filter — effectively detecting local spatial patterns (edges, gradients, textures) at each scale. This is particularly relevant for thermal data, where local spatial gradients carry diagnostic information about land cover boundaries.

### Multi-Stage Token Handling

After Stage 1 processes the visible tokens, the transition to Stage 2 requires attention. Stage 2's overlap patch embedding expects a full 2D feature map, not a sparse set of tokens. Two approaches:

**Approach A — Scatter and re-embed:** Place the encoded visible tokens back into a full 2D grid (with zeros at masked positions), apply Stage 2's patch embedding convolution over the full grid, then extract the new visible tokens at Stage 2's resolution. The patch embedding's convolution pools over $2 \times 2$ spatial regions, so most Stage 2 tokens will contain at least some real information even with 50% masking.

**Approach B — Dense encoder, sparse loss:** Process all tokens through the encoder (including masked positions filled with a learned mask embedding), but compute the loss only at originally masked positions. This is simpler but loses the computational efficiency advantage and the "no information leakage" property of true token removal.

Approach A is preferred for its principled handling of the masking.


## 6. MLP Reconstruction Decoder

### Architecture

The decoder takes the encoded features from all 4 encoder stages and fuses them to reconstruct the full-resolution input. Since the encoder only processed visible tokens, the decoder must also reconstruct the masked tokens.

1. **Reintroduce masked positions.** After encoding, create learnable mask tokens with positional encodings at the masked positions. These mask tokens carry no information about the original pixel values — they signal "predict this location."

$$
tokens_{full} = scatter(tokens_{encoded}, visible\_indices) + scatter(tokens_{mask}, masked\_indices)
$$

2. **Multi-scale feature fusion.** For each encoder stage $i$, the feature map $F_i$ (now containing both encoded visible tokens and learnable mask tokens) is unified to a common channel dimension $C_{embed}$:

$$
F_i[(B, C_i, H_i, W_i)] \rightarrow (Conv2D \; K=1) \rightarrow F_i'[(B, C_{embed}, H_i, W_i)]
$$

3. **Upsample** all four feature maps to the Stage 1 resolution ($\frac{H}{4} \times \frac{W}{4}$):

$$
F_i'[(B, C_{embed}, H_i, W_i)] \rightarrow (Upsample) \rightarrow F_i'[(B, C_{embed}, \frac{H}{4}, \frac{W}{4})]
$$

4. **Concatenate** along the channel dimension:

$$
F_{fused}[(B, 4 \times C_{embed}, \frac{H}{4}, \frac{W}{4})] = Concat(F_1', F_2', F_3', F_4')
$$

5. **Fuse** with a $1 \times 1$ convolution:

$$
F_{fused}'[(B, C_{embed}, \frac{H}{4}, \frac{W}{4})]
$$

6. **Upsample** to full resolution via a $4\times$ bilinear upsample or a learned transposed convolution:

$$
F_{fused}'[(B, C_{embed}, \frac{H}{4}, \frac{W}{4})] \rightarrow F_{out}[(B, C_{embed}, H, W)]
$$

7. **Final projection** to the input channel count (1 for thermal):

$$
F_{out}[(B, C_{embed}, H, W)] \rightarrow (Conv2D \; K=3, S=1, P=1) \rightarrow out[(B, 1, H, W)]
$$

The final convolution has **no activation function and no BatchNorm**. The output must be free to represent any value in the normalised data range, including negative values (temperatures below the dataset mean). BatchNorm would force the output to zero-mean unit-variance — incompatible with reconstructing the actual data distribution. Any nonlinearity (ReLU, GELU) would clip the output range, preventing reconstruction of values on one side of the activation's threshold.


## 7. Loss Function

### Masked Reconstruction Loss

The loss is computed **only at masked positions**. Visible positions are excluded because the model had direct access to them via the encoder — including them in the loss would reward the trivial identity mapping rather than genuine spatial understanding.

$$
\mathcal{L} = \frac{1}{|M|} \sum_{(i,j) \in M} (x_{(i,j)} - \hat{x}_{(i,j)})^2
$$

where $M$ is the set of masked pixel positions, $x$ is the normalised input, and $\hat{x}$ is the reconstruction.

### Trimmed Loss for Anomaly Robustness

The training data may contain unlabelled anomalies. If an anomaly pixel is masked during training, the model should not be penalised for failing to predict an inherently unpredictable value. The trimmed loss discards the top $\tau\%$ of per-pixel losses before averaging:

1. Compute per-pixel MSE at all masked positions
2. Sort the per-pixel losses in descending order
3. Discard the top $\tau\%$ (e.g., $\tau = 5$)
4. Average the remaining losses

$$
\mathcal{L}_{trimmed} = \frac{1}{|M'|} \sum_{(i,j) \in M'} (x_{(i,j)} - \hat{x}_{(i,j)})^2
$$

where $M' \subset M$ excludes the highest-error pixels.

This automatically rejects anomalous pixels from the training signal without requiring explicit anomaly labels or pseudo-labels from a separate detector. The model learns only from the predictable background majority.

### Validity Masking in the Loss

The loss must also exclude invalid pixels (cloud-masked, fill values, nodata). The combined loss mask is:

$$
loss\_mask = random\_mask \cdot validity\_mask
$$

Only pixels that are both masked (part of the MAE prediction task) and valid (genuine surface measurements) contribute to the loss. Invalid pixels at masked positions are ignored — the model is neither rewarded nor penalised for predicting cloud or fill values.


## 8. Training Configuration

### Optimiser

AdamW with decoupled weight decay. Weight decay acts as L2 regularisation on the model weights, preventing them from growing large and encouraging simpler solutions that generalise better.

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

where $\lambda$ is the weight decay coefficient (typically $1 \times 10^{-4}$).

### Learning Rate Schedule

Cosine annealing with linear warmup:

- **Warmup phase** (epochs 1–5): linear ramp from $lr_{min}$ to $lr_{max}$. This prevents the large, random gradients in early training from destabilising the transformer's attention layers before the weights have moved into a reasonable region.
- **Cosine decay phase** (epochs 6–100): smooth decay from $lr_{max}$ to $lr_{min}$ following:

$$
lr_t = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{total} - T_{warmup}} \cdot \pi\right)\right)
$$

Recommended values: $lr_{max} = 1 \times 10^{-4}$, $lr_{min} = 1 \times 10^{-6}$, warmup = 5 epochs.

A learning rate of $1 \times 10^{-3}$ was found to be too aggressive for scene-level splits — the model reached its generalisation sweet spot within 3–5 epochs and then overshot into overfitting. The lower rate of $1 \times 10^{-4}$ approaches the same region more gradually, allowing the cosine decay to refine the solution over a longer training horizon.

### Dropout

Dropout at rate 0.3 is applied within the transformer blocks (after ESA attention and within Mix-FFN). During training, 30% of activations are randomly zeroed, forcing the network to learn redundant representations and preventing over-reliance on any single feature.

During validation and inference (`model.eval()`), dropout is disabled — the full network capacity is available, producing the best possible reconstruction.

The combination of dropout (during training) and BatchNorm (using per-batch statistics during training, running statistics during eval) means training loss is systematically higher than validation loss. This is expected and not a sign of underfitting. A training loss of 40 with dropout 0.3 and noisy BatchNorm corresponds to a true reconstruction ability of approximately 7 in validation — which is the number that matters.

### Batch Size and Multi-Scale Training

Training uses a fixed number of samples per epoch across multiple patch sizes:

| Patch Size | Samples per Epoch | Rationale |
|-----------|-------------------|-----------|
| 64×64 | 50,000 | Most abundant, captures fine-grained local detail |
| 128×128 | 12,500 | Moderate scale |
| 256×256 | 3,125 | Broad spatial context |
| 512×512 | 780 | Full landscape patterns |

Larger patches are sampled less frequently because they are fewer in number (each scene produces fewer 512×512 patches than 64×64 patches) and because each large patch provides proportionally more spatial information per sample.

The epoch definition is fixed across all sizes, enabling a single learning rate schedule, unified checkpointing, and consistent logging.


## 9. Inference Pipeline

### Full-Scene Reconstruction

At inference, the full satellite scene is processed using a sliding window:

1. Extract overlapping patches via `PatchPlanGenerator` with a specified patch size and stride (typically stride = patch_size / 2 for 50% overlap).
2. For each patch, run two-pass checkerboard inference:
   - **Pass 1:** Remove "black" checkerboard tokens, encode visible "white" tokens, decode to reconstruct "black" tokens
   - **Pass 2:** Remove "white" checkerboard tokens, encode visible "black" tokens, decode to reconstruct "white" tokens
   - **Combine:** Each pixel's value comes from the pass where it was masked
3. Accumulate reconstructions in overlap regions using weighted averaging.
4. Compute per-pixel reconstruction error: $A(i,j) = (x(i,j) - \hat{x}(i,j))^2$

### Anomaly Score Map

The per-pixel reconstruction error is the anomaly score. Background pixels that the network has learned to predict from context produce low error. Anomaly pixels — whose thermal signature deviates from what the spatial context predicts — produce high error.

The score map is thresholded to produce binary detections. Connected component analysis groups adjacent high-scoring pixels into discrete anomaly candidates. Each candidate is characterised by its centroid location, spatial extent, and peak/mean anomaly score.

### Relationship to Checkerboard Cell Size

The checkerboard cell size at inference determines the spatial resolution of anomaly detection:

- **Cell size 1** (individual tokens, 4×4 pixels): maximum resolution, detects point anomalies spanning a single token. Each masked token is surrounded by dense context on all sides.
- **Cell size 2–4** (blocks of tokens, 8–16 pixels): lower resolution, detects extended anomalies. The model must predict larger regions from more distant context, which is harder — reconstruction error is higher for everything, but anomalies produce disproportionately higher error because their spatial extent exceeds what context can explain.

The cell size should match the expected anomaly scale. For thermal anomaly detection at 30m Landsat resolution, a cell size of 1–2 tokens (4–8 pixels, 120–240 metres) is appropriate for detecting industrial thermal discharges, underground fires, or urban heat anomalies.


## 10. Cloud Masking Considerations

### The Train-Inference Mismatch Problem

During training, the provider's QA_PIXEL cloud mask is available — it reliably identifies clouds, cloud shadows, cirrus, and fill values. The training pipeline masks these pixels, and the model never sees cloud-contaminated data.

At production inference, QA_PIXEL is not available (the scene arrives without quality flags). A separate cloud detection model (or simpler GMM-based approach) must generate the cloud mask. This model is less accurate than the provider mask — it misses some clouds and cloud edges.

The missed cloud pixels enter the reconstruction network. Since the model was never trained on cloud temperatures (which can range from $-40$°C to $-120$°C for high-altitude clouds), it cannot reconstruct them. The reconstruction error is very high. These cloud remnants appear as false anomalies in the detection map.

### Resolution

Two strategies address this mismatch:

**Strategy 1 — Unified cloud mask model.** Train a SegFormer-B0 cloud segmentation model on Landsat scenes using QA_PIXEL as the training label. Use this same cloud model at both training and inference. The model has the same biases and failure modes in both phases — if it misses a cloud type during training, it misses the same type at inference. The reconstruction network learns to handle whatever leaks through, consistently.

**Strategy 2 — Temperature-based sanity check.** In addition to the learned cloud mask, apply a hard temperature threshold: any pixel below $-10$°C is flagged as cloud regardless of the mask model's prediction. For the Indian subcontinent, no land surface temperature reaches $-10$°C at Landsat overpass time. This catches the catastrophic failure case of thick cloud decks with extremely cold brightness temperatures that the cloud model might miss.

The recommended approach is Strategy 1 with Strategy 2 as a safety net.


## 11. Evaluation Metrics

### Why F1 Fails at Extreme Imbalance

For anomaly detection with 205 labelled anomalies in 40 million pixels (0.0005% prevalence), F1 at any single threshold produces near-zero values. At the optimal threshold, perhaps 4,400 pixels are flagged. Even with perfect recall, precision is at most $205/4400 \approx 4.6\%$. In practice, recall is also low because many anomalies are not thermally distinctive. F1 collapses.

### Recommended Metrics

**AUC-ROC** (Area Under the Receiver Operating Characteristic curve): measures the detector's ability to rank anomaly pixels higher than background pixels across all possible thresholds. AUC = 0.85 means a randomly chosen anomaly pixel has an 85% probability of scoring higher than a randomly chosen background pixel.

**AUC-PR** (Area Under the Precision-Recall curve): more informative than ROC for extreme class imbalance. It measures whether the highest-scoring pixels are enriched with true anomalies.

**Detection rate at fixed false alarm rates**: operationally meaningful — "at a false alarm rate of 1%, the detector captures X% of true anomalies." This tells a user the trade-off between detection completeness and false alarm burden.

**Rank analysis**: sort all pixels by anomaly score. Report the percentile at which the median anomaly pixel falls. If the median anomaly is at the 99.5th percentile, the detector concentrates anomalies in the top 0.5% of pixels — effective even if F1 is near zero.

### Thermal Detection Limitations

Single-band thermal detection is fundamentally limited to anomalies that manifest thermally. For the ground truth dataset analysed (205 anomalies), the mean anomaly temperature was 36.6°C against a scene mean of 33.5°C — a z-score of only 0.73. 72% of anomalies fell within ±2σ of the background distribution. Only 28% (58 pixels) were above +2σ — the thermally distinctive minority.

The reconstruction-based detector can reliably detect the 28% that are thermally extreme. The remaining 72% are labelled as anomalies for reasons that do not produce a thermal signature (material type, regulatory status, land use) and require spectral (hyperspectral) detection to identify.

This is not a model failure — it is a fundamental limit of the sensing modality. Reporting this limit honestly, with the z-score analysis, is itself a contribution.


## 12. Summary of Key Design Decisions

| Decision | Choice | Rationale |
|---------|--------|-----------|
| Architecture | SegFormer (transformer) | Enables true MAE token removal; attention captures multi-scale spatial context |
| Masking strategy | MAE (random token removal in training) | Eliminates train-inference mismatch; prevents memorisation; natural data augmentation |
| Masking ratio | 50% | Matches two-pass checkerboard inference ratio |
| Normalisation order | Mask then normalise | Preserves strong masking signal (0°C → -2.3 in normalised space) |
| Output activation | None (linear) | Output must represent full temperature range including below-mean values |
| Loss | Trimmed MSE at masked positions only | Rejects unlabelled anomalies; computed only where model predicted from context |
| Learning rate | 1e-4 with cosine decay | Lower rate prevents overshooting the generalisation sweet spot on scene-level splits |
| Regularisation | Dropout 0.3 + weight decay 1e-4 | Dropout prevents memorisation; weight decay constrains weight magnitudes |
| Validation split | Scene-level holdout | Tests generalisation to unseen locations — the realistic deployment scenario |
| Cloud handling | Unified cloud mask model for both train and inference | Eliminates train-inference mismatch in preprocessing |
| Evaluation | AUC-ROC, AUC-PR, detection rate at fixed FAR | F1 is inappropriate for 205 anomalies in 40M pixels |