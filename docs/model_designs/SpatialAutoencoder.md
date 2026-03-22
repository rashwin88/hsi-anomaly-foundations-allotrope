### Spatial Autoencoder for Single-Band Thermal Data

#### Motivation

For single-band data (e.g. Landsat thermal Band 10), spectral compression is not applicable — there is only one channel. Instead, we compress and reconstruct in the spatial domain. The network learns a lower-dimensional spatial representation of the thermal scene. Regions that reconstruct poorly (high per-pixel error) are flagged as anomalous.

#### Spatial Encoder

The input is a single-band thermal patch of size $(B, 1, H, W)$. We progressively halve the spatial resolution while doubling the channel count using strided convolutions.

Each stage applies a `Conv2d` with kernel size $K=4$, stride $S=2$, padding $P=1$. The output spatial dimensions follow:

$$
H' = \frac{H + 2P - K}{S} + 1 = \frac{H}{2}
$$

The three encoding stages are:

$$
(B, 1, 128, 128) \rightarrow (Conv2D, BN, GELU) \rightarrow (B, C_b, 64, 64)
$$

$$
(B, C_b, 64, 64) \rightarrow (Conv2D, BN, GELU) \rightarrow (B, 2C_b, 32, 32)
$$

$$
(B, 2C_b, 32, 32) \rightarrow (Conv2D, BN, GELU) \rightarrow (B, 4C_b, 16, 16)
$$

Where $C_b$ is the base channel count (default 32). The bottleneck representation is $(B, 4C_b, 16, 16)$.

##### Why BatchNorm + GELU?

Unlike the spectral compressor (which is kept linear for interpretability), the spatial encoder needs nonlinearity. Spatial patterns — edges, thermal gradients, texture — are inherently nonlinear and require activation functions to learn.

##### Information Capacity

Each layer preserves roughly the same total capacity: $C_b \times 64 \times 64 \approx 2C_b \times 32 \times 32 \approx 4C_b \times 16 \times 16$. This means no single layer is forced to do a drastic lossy compression — each step refines gradually.

#### Spatial Decoder

The decoder mirrors the encoder using `ConvTranspose2d` (learnable upsampling) with kernel size $K=4$, stride $S=2$, padding $P=1$. The output spatial dimensions follow:

$$
H' = (H - 1) \times S - 2P + K = 2H
$$

The three decoding stages are:

$$
(B, 4C_b, 16, 16) \rightarrow (ConvTranspose2D, BN, GELU) \rightarrow (B, 2C_b, 32, 32)
$$

$$
(B, 2C_b, 32, 32) \rightarrow (ConvTranspose2D, BN, GELU) \rightarrow (B, C_b, 64, 64)
$$

$$
(B, C_b, 64, 64) \rightarrow (ConvTranspose2D) \rightarrow (B, 1, 128, 128)
$$

The final layer has no BatchNorm or activation — the output should be free to match the original input scale and distribution without normalization constraints.

##### Why ConvTranspose2d?

Each output pixel is a learned combination of bottleneck features. The alternative is bilinear interpolation followed by a `Conv2d` (upsample-then-refine), which avoids checkerboard artifacts but is less expressive. If checkerboard artifacts appear during training, switch to the interpolation approach.

#### Full Autoencoder

The spatial autoencoder wires encoder and decoder together:

$$
x[(B, 1, H, W)] \rightarrow Encoder \rightarrow z[(B, 4C_b, \frac{H}{8}, \frac{W}{8})] \rightarrow Decoder \rightarrow \hat{x}[(B, 1, H, W)]
$$

#### Loss and Anomaly Detection

Training uses masked MSE — invalid pixels (fill, cloud, etc.) are zeroed before the forward pass and excluded from the loss:

$$
\mathcal{L} = \frac{\sum_{b,c,i,j} (\hat{x} - x)^2 \cdot M}{\sum M}
$$

Where $M$ is the binary validity mask (1 = valid, 0 = invalid).

At inference, the per-pixel anomaly score is the reconstruction error:

$$
A_{(i,j)} = (\hat{x}_{(i,j)} - x_{(i,j)})^2
$$

Regions the network cannot reconstruct well (high error) are flagged as anomalous.

#### Tuning the Bottleneck

The base channel count $C_b$ controls compression aggressiveness:

| $C_b$ | Bottleneck shape | Bottleneck values | Compression ratio vs $128 \times 128$ |
|--------|-----------------|-------------------|---------------------------------------|
| 16 | $(B, 64, 16, 16)$ | 16,384 | 1:1 |
| 32 | $(B, 128, 16, 16)$ | 32,768 | 2:1 (overcomplete) |
| 8 | $(B, 32, 16, 16)$ | 8,192 | 1:2 |

Adding a 4th encoder/decoder stage reduces spatial dims to $8 \times 8$, forcing a much tighter bottleneck.
