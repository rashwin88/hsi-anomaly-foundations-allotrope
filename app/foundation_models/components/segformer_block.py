"""
SegFormer Transformer Block -- the repeating unit within each encoder stage.

Each block applies two sublayers with Pre-Norm residual connections:

    x = x + Dropout(ESA(LayerNorm(x)))        attention sublayer
    x = x + Dropout(MixFFN(LayerNorm(x)))     feed-forward sublayer

Pre-Norm means LayerNorm is applied BEFORE each sublayer, not after. This is
the modern standard (GPT-2, ViT, SegFormer) because:
    - Gradients flow through the residual path unmodified (no norm blocking them)
    - Training is more stable, especially for deeper models
    - Each sublayer always receives normalized input, keeping activations well-behaved

Dropout is applied AFTER the sublayer but BEFORE the residual add. This means
only the new contribution (from attention or FFN) is randomly dropped -- the
original representation flowing through the skip connection is always preserved.
This forces the model to not over-rely on any single attention output or FFN
feature, while keeping the base information highway intact.

Each SegFormer encoder stage stacks multiple blocks (e.g., 2 blocks per stage
in SegFormer-B0). Within a stage, all blocks share the same embed_dim, num_heads,
and reduction_ratio -- the spatial resolution and channel count only change at
the OverlapPatchEmbedding transitions between stages.
"""

import torch
import torch.nn as nn

from app.foundation_models.components.overlap_patch_embedding import OverlapPatchEmbedding
from app.foundation_models.components.efficient_self_attention import EfficientSelfAttention
from app.foundation_models.components.mix_ffn import MixFFN

class SegFormerBlock(nn.Module):
    """
    Single transformer block: Pre-Norm ESA + residual, then Pre-Norm MixFFN + residual.

    Data flow:
        Input:  (B, N, C)

        Attention sublayer:
            (B, N, C) -> LayerNorm -> ESA -> Dropout -> + residual from input
        (B, N, C)

        FFN sublayer:
            (B, N, C) -> LayerNorm -> MixFFN -> Dropout -> + residual from attention output
        (B, N, C)

        Output: (B, N, C)

    The residuals chain sequentially: x -> attn_out -> ffn_out.
    Each residual connects to the immediately preceding representation,
    NOT back to the original input. This ensures information accumulates
    through both sublayers.

    Args:
        embed_dim:        Token embedding dimension C
        num_heads:        Number of attention heads for ESA
        reduction_ratio:  Spatial reduction R for ESA's K/V path
        expansion_ratio:  Hidden dim multiplier E for MixFFN (default 4)
        drop_ratio:       Dropout rate applied after ESA and MixFFN (default 0.0)
    """

    def __init__(self, embed_dim, num_heads, reduction_ratio, expansion_ratio = 4, drop_ratio=0.0):
        super().__init__()

        # Pre-norm for attention sublayer
        self.norm1 = nn.LayerNorm(embed_dim)
        # Efficient self-attention: full-res Q, spatially-reduced K/V
        self.attn = EfficientSelfAttention(embed_dim, num_heads, reduction_ratio)

        # Pre-norm for FFN sublayer
        self.norm2 = nn.LayerNorm(embed_dim)
        # Position-aware feed-forward with depthwise convolution
        self.ffn = MixFFN(embed_dim, expansion_ratio)

        # Dropout applied to sublayer outputs before residual add
        # Only the new contribution is dropped; the skip connection is preserved
        self.drop = nn.Dropout(drop_ratio)


    def forward(self, x, H, W):
        """
        Args:
            x: Token sequence (B, N, C) where N = H * W
            H: Spatial height of the token grid
            W: Spatial width of the token grid

        Returns:
            ffn: (B, N, C) -- tokens updated by attention and feed-forward
        """
        # x: (B, N, C)

        # --- Attention sublayer ---
        # Pre-norm -> ESA -> dropout -> residual add
        attn = self.drop(self.attn(self.norm1(x), H, W))
        attn = x + attn
        # attn: (B, N, C) -- input enriched with global context via attention

        # --- FFN sublayer ---
        # Pre-norm -> MixFFN -> dropout -> residual add
        # Note: residual connects to attn (attention output), not original x
        ffn = self.drop(self.ffn(self.norm2(attn), H, W))
        ffn = attn + ffn
        # ffn: (B, N, C) -- further enriched with local spatial context via depthwise conv

        return ffn
