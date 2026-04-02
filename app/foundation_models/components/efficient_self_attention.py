"""
Efficient Self-Attention (ESA) -- the core attention mechanism in SegFormer.

Standard self-attention computes Q, K, V from all N tokens, giving O(N^2) cost
in the attention matrix. ESA reduces this by spatially compressing the key and
value tokens by a reduction ratio R, while keeping queries at full resolution.

    Standard attention cost:  O(N * N)     = O(N^2)
    ESA attention cost:       O(N * N/R^2) = O(N^2 / R^2)

At Stage 1 (N=1024 tokens, R=8): attention matrix is 1024x16   instead of 1024x1024
At Stage 2 (N=256 tokens,  R=4): attention matrix is 256x16    instead of 256x256
At Stage 3 (N=64 tokens,   R=2): attention matrix is 64x16     instead of 64x64
At Stage 4 (N=16 tokens,   R=1): full attention, no reduction  (16x16 is already cheap)

The spatial reduction is implemented as a strided Conv2d on the K/V path:
    - Reshape tokens back to 2D spatial form (B, C, H', W')
    - Apply Conv2d(kernel=R, stride=R) -- non-overlapping learned spatial pooling
    - Each output token summarizes an R x R region of input tokens
    - Flatten back to sequence form for K/V projection

Multi-head attention splits the embedding dimension across heads so each head
can learn different attention patterns (e.g., one head focuses on nearby tokens,
another on distant spatial relationships).
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with spatial reduction on K and V.

    Data flow:
        Q path (full resolution):
            (B, N, C) -> Linear(C, C) -> reshape -> (B, heads, N, head_dim)

        K/V path (spatially reduced when R > 1):
            (B, N, C)
                -> transpose to (B, C, N) -> reshape to (B, C, H', W')
                -> Conv2d(stride=R) -> (B, C, H'/R, W'/R)
                -> flatten(2) -> transpose -> (B, N_reduced, C)    where N_reduced = N/R^2
                -> LayerNorm
            (B, N_reduced, C)
                -> Linear(C, C) each for K and V
                -> reshape -> (B, heads, N_reduced, head_dim)

        Attention:
            scores = Q @ K^T * (1/sqrt(head_dim))   -> (B, heads, N, N_reduced)
            weights = softmax(scores, dim=-1)        -> (B, heads, N, N_reduced)
            out = weights @ V                        -> (B, heads, N, head_dim)

        Output:
            (B, heads, N, head_dim) -> transpose -> (B, N, heads, head_dim)
            -> reshape -> (B, N, C) -> Linear(C, C) -> (B, N, C)

    Each element scores[b, h, i, j] represents how much full-resolution
    token i should attend to reduced-token j (which summarizes an R x R
    spatial region). This is the key insight: queries ask fine-grained
    questions, but keys/values provide spatially-aggregated answers.

    Args:
        embed_dim:        Token embedding dimension C (must be divisible by num_heads)
        num_heads:        Number of attention heads; head_dim = embed_dim // num_heads
        reduction_ratio:  Spatial reduction factor R for K and V
                          Typical values: R=8 (Stage 1), R=4 (Stage 2),
                          R=2 (Stage 3), R=1 (Stage 4, full attention)
    """

    def __init__(self, embed_dim, num_heads, reduction_ratio):
        super().__init__()

        self.num_heads = num_heads
        # Each head operates on a slice of the embedding: head_dim = C // num_heads
        # e.g., embed_dim=64, num_heads=2 -> head_dim=32
        self.head_dim = embed_dim // num_heads
        # Scale factor 1/sqrt(d_k) for attention scores.
        # Without scaling, the dot product Q @ K^T grows in magnitude with head_dim,
        # pushing softmax into saturation (nearly all weight on one token, near-zero
        # gradients for the rest). Dividing by sqrt(d_k) keeps score variance ~ 1.
        self.scale = self.head_dim ** -0.5

        # Separate linear projections for Q, K, V: each maps (*, C) -> (*, C)
        # Internally these are weight matrices W_q, W_k, W_v of shape (C, C).
        # The operation is y = xW^T + b (PyTorch transposes W internally).
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        # Output projection: recombines the concatenated head outputs back to embed_dim
        # After attention, heads are concatenated: (B, N, heads * head_dim) = (B, N, C)
        # This linear layer lets heads interact and produce the final representation.
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Spatial reduction for K/V path (only when R > 1)
        # Conv2d with kernel_size=R, stride=R performs non-overlapping spatial pooling:
        #   (B, C, H', W') -> (B, C, H'/R, W'/R)
        # This is like a learned average pooling -- each output summarizes an R x R
        # spatial region, reducing the token count by R^2.
        self.reduction_ratio = reduction_ratio

        if self.reduction_ratio > 1:
            self.reduction = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=self.reduction_ratio,
                stride = reduction_ratio
            )
            # LayerNorm after reduction stabilizes the reduced token representations
            # before they are projected into K and V.
            self.reduction_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Token sequence (B, N, C) where N = H * W
            H: Spatial height of the token grid (needed for reshape to 2D)
            W: Spatial width of the token grid (needed for reshape to 2D)

        Returns:
            out: (B, N, C) -- same shape as input, each token updated via attention
        """

        # x: (B, N, C)  where N = H * W (full grid) or N < H * W (after token removal)
        B, N, C = x.shape

        # --- Q: full resolution queries ---
        Q_x = self.q(x)
        # Q_x: (B, N, C)

        # Split embedding dim into heads: (B, N, C) -> (B, N, heads, head_dim)
        # Then transpose to put heads before tokens: -> (B, heads, N, head_dim)
        Q_x = torch.reshape(Q_x, (B,N, self.num_heads, self.head_dim))
        Q_x = Q_x.transpose(1,2) # (B, num_heads, N, head_dim)

        # --- K, V: spatially reduced keys and values ---
        # Spatial reduction requires reshaping tokens to a full 2D grid (B, C, H, W),
        # which only works when N == H * W. After MAE token removal at Stage 1,
        # N < H * W, so the reshape would crash.
        #
        # When N != H*W, we fall back to full attention (no spatial reduction).
        # With ~512 tokens after 50% masking, full attention costs 512*512 = 262K
        # elements per head -- very cheap, no efficiency concern.
        can_reduce = (self.reduction_ratio > 1) and (N == H * W)

        if can_reduce:
            # Reshape tokens back to 2D spatial form for the reduction Conv2d
            # (B, N, C) -> transpose -> (B, C, N) -> reshape -> (B, C, H, W)
            reshaped_x = x.transpose(1,2) # (B, C, N)
            reshaped_x = torch.reshape(reshaped_x, (B,C,H,W))
            # reshaped_x: (B, C, H, W)

            # Apply non-overlapping spatial reduction via strided convolution
            reduced_x = self.reduction(reshaped_x) # (B, C, H//R, W//R)

            # Flatten back to sequence form: (B, C, H//R, W//R) -> (B, N_reduced, C)
            # where N_reduced = (H//R) * (W//R) = N / R^2
            reduced_x = reduced_x.flatten(2).transpose(1,2) # (B, N_reduced, C)

            # Normalize the reduced tokens before K/V projection
            reduced_x = self.reduction_norm(reduced_x) # (B, N_reduced, C)

        else:
            # Either R=1 (Stage 4, full attention by design)
            # or N != H*W (Stage 1 after token removal, full attention as fallback)
            reduced_x = x
            # reduced_x: (B, N, C)  (no reduction, K/V at same resolution as Q)

        # Project reduced tokens into K and V spaces
        # Use -1 for the token dimension to handle both reduced and unreduced cases
        K_x = self.k(reduced_x) # (B, N_reduced, C)
        K_x = K_x.reshape(B, -1, self.num_heads, self.head_dim)
        K_x = K_x.transpose(1,2) # (B, num_heads, N_reduced, head_dim)

        V_x = self.v(reduced_x) # (B, N_reduced, C)
        V_x = V_x.reshape(B, -1, self.num_heads, self.head_dim)
        V_x = V_x.transpose(1,2) # (B, num_heads, N_reduced, head_dim)

        # --- Attention ---
        # scores[b, h, i, j] = how much should full-res token i attend to reduced token j
        # Shape: (B, heads, N, head_dim) @ (B, heads, head_dim, N_reduced) = (B, heads, N, N_reduced)
        scores = Q_x @ K_x.transpose(-2,-1) * self.scale # (B, heads, N, N_reduced)
        weights = F.softmax(scores, dim = -1) # (B, heads, N, N_reduced) -- rows sum to 1
        # Weighted sum of values: each token gets a context-aggregated representation
        # (B, heads, N, N_reduced) @ (B, heads, N_reduced, head_dim) = (B, heads, N, head_dim)
        out = weights @ V_x # (B, heads, N, head_dim)

        # --- Recombine heads ---
        # (B, heads, N, head_dim) -> transpose -> (B, N, heads, head_dim)
        out = out.transpose(1,2) # (B, N, heads, head_dim)
        # Concatenate heads: (B, N, heads, head_dim) -> (B, N, heads * head_dim) = (B, N, C)
        out = torch.reshape(out, (B,N,self.head_dim*self.num_heads)) # (B, N, C)

        # Final linear projection: lets heads interact and produces output representation
        out = self.proj(out) # (B, N, C)

        return out
