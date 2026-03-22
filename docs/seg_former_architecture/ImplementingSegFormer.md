### Implementation of the SegFormer Architecture

#### Spectral Compressor (Learnable MNF Analogue)

Before any spatial processing, we compress the spectral channels from $C_{in}$ (e.g. 256 bands) down to $C_{compressed}$ using a $1 \times 1$ convolution. This is conceptually analogous to MNF (Minimum Noise Fraction) — it learns a linear projection of the input spectra that is optimized end-to-end with the reconstruction objective.

At each spatial position $(i, j)$, the output channel $m$ is:

$$
z_{(b, m, i, j)} = \sum_{c=1}^{C_{in}} W_{(m,c)} \cdot x_{(b,c,i,j)} + b_m
$$

Each output channel is a learned weighted sum across all input spectral bands — a linear combination, same family as PCA/MNF but trained jointly with the full network.

1. Apply a `Conv2d` with kernel size $1$, stride $1$, padding $0$ to map $C_{in} \rightarrow C_{compressed}$
$$
x[(B, C_{in}, H, W)] \rightarrow (Conv2D \: K=1, S=1, P=0) \rightarrow x'[(B, C_{compressed}, H, W)]
$$

2. Apply `BatchNorm2d` to stabilize per-channel statistics across the batch. We use BatchNorm rather than LayerNorm here because the data is still in spatial $(B, C, H, W)$ form and BN normalizes per-channel across the batch dimension.

3. No nonlinearity is applied — keeping the projection strictly linear preserves interpretability. The learned weight matrix $W[(C_{compressed}, C_{in})]$ can be inspected post-training to see which spectral bands contribute most to each compressed channel, effectively revealing the "virtual MNF components" the network discovered.

The output $x'[(B, C_{compressed}, H, W)]$ becomes the input to the Overlap Patch Embedding, so $C_{compressed}$ replaces $C_{in}$ in all downstream stages.

#### Overlap Patch Embedding
The input to this stage is now $ (B, C_{compressed} , H, W) $. We apply the `Conv2d` with a kernel size `K`, stride `S` and padding `P`. The output of the convolution is a tensor of size $ (B,C_{out}, H', W') $.  Where

$$
H' = floor\left(\frac{H+2P -K} {S}\right) + 1
$$

$$ 
W' = floor\left(\frac{W+2P -K} {S}\right) + 1
$$

This basically means that each kernel filter is $ (C_{in}, K, K) $ in size and produces one output channel we have $ C_{out} $ such filters. The output of this has to be fed into a transformer block, but the shape is not appropriate for feeding into a transformer, because what we have as a result of the convolution is a spatial feature map, and transformers expect a sequence of token vectors. So we do the following

1. Reshape the output of the convolution into a $ (B, H' \times W', C_{out}) $ shape.
2. Then apply a layer norm. 

##### Detour on Layer norm mechanics
LayerNorm normalizes across the last dimension. So, if we slice out the last dimension say the batch element `b` and token (position) `i` then that vector has a length of $C_{out}$

```
len(x[b,i,:]) = c_out
```

We compute the mean and variance across this vector and normalize each element with a scale and shift $\gamma, \beta$ (And the scale and shift are shared across all tokens and batches) and are each vectors of size $C_{out}$. For each embedding dimension of each token, we have a $\gamma$ and a $\beta$

So effectively, if we consider the token $n$ in batch $b$, we have for $C_{out}$ embedding dimensions,

$$
\hat{x}_{{(b,n,j)}} = \gamma_j * \frac{x_{(b,n,j)} - \mu_{(b,n)}}{\sigma_{(b,n)} + \epsilon} + \beta_j
$$

#### Mix Transformer
This is where the actual learning happens. This stage also has many components and the first is 

##### Multi-Head Efficient Self Attention (ESA)
If the image patch size is large and we have many spatial locations from the convolution then the attention operation can become very expensive indeed, so we use a more efficient attention mechanism that changes the way Keys and Values are handled.

1. The output of the over lap patch embedding is a $(B,N,C_{out})$ tensor. First, we will treat this tensor as is to get the query matrix. Considering the input to the attention as $x$ which is of shape $(B,N, C_{out})$, we take a query weight matrix $W_Q$ of shape $(C_{out}, C_{out})$ and perform a matrix multiplication

$$
Q[(B, N, C_{out})] = x [(B,N,C_{out})] \times W_Q^T[(C_{out}, C_{out})]
$$
2. Note that in the case of matrix multiplication involving tensors of greater than 2 dimensions in shape, everything happens in the last 2 dimensions.
3. We then reshape the input into its original shape
$$
x [(B,N,C_{out})] (reshape)\rightarrow x'[(B,C_{out}, H',W')]
$$
4. This is the first step of efficieny in self attention. We then use a convolutional filter of kernel size $R$ and stride $R$ on this to produce a tensor of shape. This is a spatial reduction
$$ 
x'[(B,C_{out}, H',W')] \rightarrow (Conv2D \: Str: R, Size : R ) \rightarrow y'[(B,C_{out}, \frac{H'}{R}, \frac{W'}{R})]
$$
5. We then reshape it into a lower token form
$$
y'[(B,C_{out}, \frac{H'}{R}, \frac{W'}{R})] \rightarrow (reshape) \rightarrow y'[(B,\frac{N}{R^2}, C_{out})]
$$
6. Another layer norm is applied to stabilize features.
7. We then compute the $K$ and $V$ matrices as
$$
K[(B, \frac{N}{R^2}, C_{out})] = y'[(B,\frac{N}{R^2}, C_{out})] \times W_K^T[(C_{out}, C_{out})]
$$
$$
V[(B, \frac{N}{R^2}, C_{out})] = y'[(B,\frac{N}{R^2}, C_{out})] \times W_V^T[(C_{out}, C_{out})]
$$
8. We then set the number of heads as $H$ and then re-shape the query, key and value matrices as 
$$
Q[(B, N, C_{out})] \rightarrow (Reshape) \rightarrow Q[(B,N, H,d)] \rightarrow (Transpose) \rightarrow Q[(B,H, N, d)]
$$
$$
K[(B, \frac{N}{R^2}, C_{out})] \rightarrow (Reshape) \rightarrow K[(B, \frac{N}{R^2}, H, d)] \rightarrow (Transpose) \rightarrow K[(B, H, \frac{N}{R^2}, d)]
$$
$$
V[(B, \frac{N}{R^2}, C_{out})] \rightarrow (Reshape) \rightarrow V[(B, \frac{N}{R^2}, H, d)] \rightarrow (Transpose) \rightarrow V[(B, H, \frac{N}{R^2}, d)]
$$
9. We then perform the attention operation as follows, we begin by computing the attention scores:
$$
AttnScore[(B,H,N,\frac{N}{R^2})] = \frac{Q[(B,H,N,d)] @ K^T[(B,H,d,\frac{N}{R^2})]}{\sqrt{d}}
$$
$$
Attn[(B,H,N,\frac{N}{R^2})] = SoftMax(AttnScore, dim=-1)
$$
10. We then get the final output as 
$$
out[(B,H,N,d)] = Attn[(B,H,N,\frac{N}{R^2})] @ V[(B, H, \frac{N}{R^2}, d)] ->(Reshape) -> out[(B,N,C_out)]
$$
11. The output is fed into a linear layer as 
$$
out[(B,N,C_out)] = out[(B,N,C_out)] @ W[(C_out, C_out)]
$$

12. Then we add the residual connection as 
$$
final[(B,N,C_out)] = out[(B,N,C_out)] + x[(B,N,C_{out})]
$$

##### Mix-FFN
1. Apply LayerNorm to the output of the ESA block.
2. Apply a linear layer to expand the embedding dimension
$$
x'[(B, N, C_{out} \times E)] = x[(B, N, C_{out})] \times W_1^T[(C_{out}, C_{out} \times E)]
$$
where $E$ is the expansion ratio (typically 4).

3. Reshape to spatial form for the depthwise convolution
$$
x'[(B, N, C_{out} \times E)] \rightarrow (Reshape) \rightarrow x'[(B, C_{out} \times E, H', W')]
$$

4. Apply a depthwise convolution with kernel size 3, stride 1, padding 1 and groups $= C_{out} \times E$.
$$
x'[(B, C_{out} \times E, H', W')] \rightarrow (DWConv \: K=3, S=1, P=1, \: groups=C_{out} \times E) \rightarrow x'[(B, C_{out} \times E, H', W')]
$$

##### Detour on Depthwise Convolution mechanics
In a standard `Conv2d`, each filter has the shape $(C_{in}, K, K)$ — it looks at **all** input channels at each spatial position and sums across them. If we have $C_{out}$ such filters, the weight tensor has shape $(C_{out}, C_{in}, K, K)$.

In a depthwise convolution, we set `groups` $= C_{in}$ and $C_{out} = C_{in}$. This means each channel gets its own independent filter of shape $(1, K, K)$. The weight tensor has shape $(C_{in}, 1, K, K)$.

For a standard conv, the output at channel $m$, position $(i,j)$ is:
$$
out[m, i, j] = \sum_c \sum_p \sum_q \: input[c, \: i+p, \: j+q] \cdot W[m, \: c, \: p, \: q]
$$

For a depthwise conv, there is no sum over $c$ — each channel is convolved independently:
$$
out[c, i, j] = \sum_p \sum_q \: input[c, \: i+p, \: j+q] \cdot W[c, \: 0, \: p, \: q]
$$

Channel 0's filter only sees channel 0, channel 1's filter only sees channel 1, etc. No cross-channel mixing happens.

| Type | Weight shape | Param count |
|------|-------------|-------------|
| Standard Conv2d | $(C_{out}, C_{in}, K, K)$ | $C_{out} \times C_{in} \times K^2$ |
| Depthwise Conv2d | $(C_{in}, 1, K, K)$ | $C_{in} \times K^2$ |

For $C_{in} = C_{out} = 128, K = 3$: standard $= 147,456$ params, depthwise $= 1,152$ params — $128\times$ fewer.

Cross-channel mixing is not needed here because the linear layers before and after the DWConv already handle it. The DWConv's only job is to inject **spatial** information — each channel independently learns a $3 \times 3$ spatial pattern. This is what gives SegFormer positional awareness without needing explicit positional encodings.

5. Reshape back to token form
$$
x'[(B, C_{out} \times E, H', W')] \rightarrow (Reshape) \rightarrow x'[(B, N, C_{out} \times E)]
$$

6. Apply GELU activation.

7. Apply a linear layer to project back to the original embedding dimension
$$
out[(B, N, C_{out})] = x'[(B, N, C_{out} \times E)] \times W_2^T[(C_{out} \times E, C_{out})]
$$

8. Add the residual connection
$$
final[(B, N, C_{out})] = out[(B, N, C_{out})] + x[(B, N, C_{out})]
$$

#### The 4-Stage Encoder

SegFormer has 4 stages. Each stage consists of an Overlap Patch Embedding followed by $N$ repeated Transformer Blocks (ESA + Mix-FFN). Each stage progressively reduces the spatial resolution and increases the channel dimension.

| Stage | Spatial Resolution | Channel Dim | Reduction Ratio $R$ |
|-------|-------------------|-------------|---------------------|
| 1 | $\frac{H}{4} \times \frac{W}{4}$ | $C_1$ | 8 |
| 2 | $\frac{H}{8} \times \frac{W}{8}$ | $C_2$ | 4 |
| 3 | $\frac{H}{16} \times \frac{W}{16}$ | $C_3$ | 2 |
| 4 | $\frac{H}{32} \times \frac{W}{32}$ | $C_4$ | 1 |

Early stages have more tokens so they need more spatial reduction in ESA. Later stages have fewer tokens and can afford full attention ($R=1$).

Each stage $i$ outputs a feature map $F_i$ of shape $(B, C_i, H_i, W_i)$.

#### MLP Decode Head (Reconstruction)

Since we are building a reconstruction network for anomaly detection (not a segmentation network), the decode head must reconstruct the full input image rather than produce class labels. The decode head takes the outputs from all 4 stages and fuses them.

1. For each stage $i$, take the output $F_i[(B, C_i, H_i, W_i)]$ and apply a `Conv2d` with $K=1, S=1, P=0$ to unify the channel dimensions
$$
F_i[(B, C_i, H_i, W_i)] \rightarrow (Conv2D \: K=1, S=1, P=0) \rightarrow F_i'[(B, C_{embed}, H_i, W_i)]
$$

2. Upsample all 4 feature maps to the same spatial size (the resolution of Stage 1, i.e. $\frac{H}{4} \times \frac{W}{4}$)
$$
F_i'[(B, C_{embed}, H_i, W_i)] \rightarrow (Upsample) \rightarrow F_i'[(B, C_{embed}, \frac{H}{4}, \frac{W}{4})]
$$

3. Concatenate all 4 along the channel dimension
$$
F_{fused}[(B, 4 \times C_{embed}, \frac{H}{4}, \frac{W}{4})] = Concat(F_1', F_2', F_3', F_4')
$$

4. Apply a `Conv2d` with $K=1, S=1, P=0$ to fuse the concatenated features
$$
F_{fused}[(B, 4 \times C_{embed}, \frac{H}{4}, \frac{W}{4})] \rightarrow (Conv2D \: K=1, S=1, P=0) \rightarrow F_{fused}'[(B, C_{embed}, \frac{H}{4}, \frac{W}{4})]
$$

5. Upsample to the original input spatial resolution
$$
F_{fused}'[(B, C_{embed}, \frac{H}{4}, \frac{W}{4})] \rightarrow (Upsample \: 4\times) \rightarrow F_{fused}'[(B, C_{embed}, H, W)]
$$

6. Apply a final `Conv2d` with kernel size $K=3$, stride $S=1$ and padding $P=1$ to project back to the original number of input channels (e.g. 256 spectral bands)
$$
F_{fused}'[(B, C_{embed}, H, W)] \rightarrow (Conv2D \: K=3, S=1, P=1) \rightarrow out[(B, C_{in}, H, W)]
$$

At inference time, anomalies are detected by computing the per-pixel reconstruction error between the input and the output. Regions the network cannot reconstruct well (high error) are flagged as anomalous.

