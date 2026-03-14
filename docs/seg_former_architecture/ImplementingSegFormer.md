### Implementation of the SegFormer Architecture

#### Overlap Patch Embedding
Let us imagine that the input image has a size $ (B, C , H, W) $. We first apply the `Conv2d` with a kernel size `K`, stride `S` and padding `P`. The output of the convolution is a tensor of size $ (B,C_{out}, H', W') $.  Where

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
Q[(B, N, C_{out})] \rightarrow (Reshape) \rightarrow Q[(B,N, H,d)] \rightarrow (Transpose) \rightarrow Q[(B,H, N, d)]
$$
$$
V[(B, \frac{N}{R^2}, C_{out})] \rightarrow (Reshape) \rightarrow V[(B, \frac{N}{R^2}, H, d)] \rightarrow (Transpose) \rightarrow V[(B, H, \frac{N}{R^2}, d)]
$$


