"""
SegFormer Encoder -- 4-stage hierarchical transformer with MAE token removal.

Each stage consists of:
    1. OverlapPatchEmbedding -- downsample spatial dims and project to new embed_dim
    2. N x SegFormerBlock   -- transformer blocks (ESA + MixFFN) at that resolution
    3. LayerNorm            -- normalize the stage output

MAE token removal happens at Stage 1 only:

    WITHOUT masking (inference pass without masking, or stages 2-4):
        Image -> OPE -> 1024 tokens -> Blocks -> Norm -> reshape to 2D

    WITH masking (Stage 1 during training or inference):
        Image -> OPE -> 1024 tokens
            -> remove prediction tokens -> ~512 visible tokens
            -> Blocks (only on visible + invalid tokens)
            -> Norm
            -> scatter back into full 1024-token grid (zeros at masked positions)
            -> reshape to 2D

    Token removal only happens at Stage 1. After scattering back and reshaping
    to 2D, Stage 2's OPE (Conv2d stride=2) pools over 3x3 regions, so most
    Stage 2 tokens contain at least some real information even with 50% masking.
    Stages 2-4 process ALL tokens -- the information loss from masking naturally
    dilutes through the spatial reduction.

Concrete example for 128x128 input with 50% masking:

    Stage 1: (B,1,128,128) -> OPE -> 1024 tokens -> REMOVE 512 -> 512 tokens
             -> Blocks -> Norm -> scatter back to 1024 (512 zeros) -> (B,32,32,32)

    Stage 2: (B,32,32,32) -> OPE -> 256 tokens -> Blocks -> Norm -> (B,64,16,16)
             (no removal -- some tokens have zero-ish values from masked Stage 1 positions,
              but the stride-2 conv pools over them, diluting the gaps)

    Stage 3: (B,64,16,16) -> OPE -> 64 tokens -> Blocks -> Norm -> (B,160,8,8)
    Stage 4: (B,160,8,8)  -> OPE -> 16 tokens -> Blocks -> Norm -> (B,256,4,4)

    Output: [F1, F2, F3, F4] at 4 spatial scales
"""

import torch
import torch.nn as nn

from app.foundation_models.components.overlap_patch_embedding import OverlapPatchEmbedding
from app.foundation_models.components.segformer_block import SegFormerBlock
from app.foundation_models.components.token_masking import TokenMasking


class SegFormerEncoder(nn.Module):
    """
    4-stage hierarchical transformer encoder with optional MAE token removal.

    When keep_mask is provided, Stage 1 removes masked tokens before processing.
    The encoder only sees visible + invalid tokens -- prediction targets are
    physically absent from the sequence. After Stage 1, tokens are scattered
    back to the full grid (zeros at masked positions) and Stages 2-4 process
    all tokens normally.

    Data flow per stage (without masking):
        (B, C_in, H, W)
            -> OverlapPatchEmbedding -> (B, N, C_out) + H', W'
            -> SegFormerBlock x num_blocks -> (B, N, C_out)
            -> LayerNorm -> (B, N, C_out)
            -> reshape to (B, C_out, H', W')

    Data flow for Stage 1 (with masking):
        (B, C_in, H, W)
            -> OverlapPatchEmbedding -> (B, 1024, C1)
            -> remove_tokens(keep_mask) -> (B, ~512, C1)    prediction targets removed
            -> SegFormerBlock x num_blocks -> (B, ~512, C1)  encoder never sees targets
            -> LayerNorm -> (B, ~512, C1)
            -> restore_tokens -> (B, 1024, C1)               zeros at masked positions
            -> reshape to (B, C1, 32, 32)

    SegFormer-B0 configuration:
        Stage | embed_dim | num_heads | reduction_ratio | num_blocks | spatial
        ------|-----------|-----------|-----------------|------------|--------
          1   |    32     |     1     |        8        |     2      | H/4 x W/4
          2   |    64     |     2     |        4        |     2      | H/8 x W/8
          3   |   160     |     5     |        2        |     2      | H/16 x W/16
          4   |   256     |     8     |        1        |     2      | H/32 x W/32

    Args:
        in_channels:      Input channels (1 for single-band thermal)
        embed_dims:       Embedding dimension per stage, e.g. [32, 64, 160, 256]
        num_heads:        Attention heads per stage, e.g. [1, 2, 5, 8]
        reduction_ratios: ESA spatial reduction per stage, e.g. [8, 4, 2, 1]
        num_blocks:       Transformer blocks per stage, e.g. [2, 2, 2, 2]
        expansion_ratio:  MixFFN expansion multiplier (default 4)
        drop_rate:        Dropout rate for transformer blocks (default 0.0)
    """

    def __init__(self, in_channels, embed_dims, num_heads,
                 reduction_ratios, num_blocks, expansion_ratio=4, drop_rate=0.0):
        super().__init__()

        self.num_stages = len(embed_dims)

        # --- Patch embeddings: one per stage ---
        # Stage 1: kernel=4, stride=4 (NON-OVERLAPPING — zero information leakage
        #          for MAE token removal. Each token sees exactly its own 4x4 block.)
        # Stages 2-4: kernel=3, stride=2 (overlapping, halves spatial dims.
        #             No token removal at these stages so overlap is beneficial.)
        # c_in for each stage: [in_channels, embed_dims[0], embed_dims[1], embed_dims[2]]
        self.patch_embeds = nn.ModuleList()
        for i in range(self.num_stages):
            c_in = in_channels if i == 0 else embed_dims[i - 1]
            patch_size = 4 if i == 0 else 3
            stride = 4 if i == 0 else 2
            # Stage 1: padding=0 for non-overlapping (kernel=stride=4)
            # Stages 2-4: padding=kernel//2 for overlapping (default)
            padding = 0 if i == 0 else None
            self.patch_embeds.append(
                OverlapPatchEmbedding(
                    c_in=c_in,
                    patch_size=patch_size,
                    desired_compression=stride,
                    c_out=embed_dims[i],
                    padding=padding,
                )
            )

        # --- Transformer blocks: num_blocks[i] blocks per stage ---
        # All blocks within a stage share the same embed_dim, num_heads, reduction_ratio
        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            stage_blocks = nn.ModuleList([
                SegFormerBlock(
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    reduction_ratio=reduction_ratios[i],
                    expansion_ratio=expansion_ratio,
                    drop_ratio=drop_rate,
                )
                for _ in range(num_blocks[i])
            ])
            self.blocks.append(stage_blocks)

        # --- Layer norms: one per stage, applied after all blocks ---
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims[i]) for i in range(self.num_stages)
        ])

    def forward(self, x, keep_mask=None):
        """
        Args:
            x:         (B, in_channels, H, W) -- spatial input (e.g. normalized thermal image)
            keep_mask: (B, N_stage1) or None -- token-level mask for Stage 1
                       1=keep (visible valid + invalid), 0=remove (prediction target)
                       N_stage1 = (H/4) * (W/4) = 1024 for 128x128 input
                       If None, no token removal is performed (full reconstruction).

        Returns:
            features: list of 4 feature maps, one per stage
                F1: (B, C1, H/4,  W/4)
                F2: (B, C2, H/8,  W/8)
                F3: (B, C3, H/16, W/16)
                F4: (B, C4, H/32, W/32)

        Example with masking (B=2, input 128x128, 50% masking):
            Stage 1:
                OPE:     (2, 1, 128, 128) -> (2, 1024, 32)   all 1024 tokens
                Remove:  keep_mask has ~512 ones -> (2, 512, 32) visible tokens only
                Blocks:  (2, 512, 32) -> (2, 512, 32)        encoder processes 512 tokens
                Norm:    (2, 512, 32) -> (2, 512, 32)
                Restore: (2, 512, 32) -> (2, 1024, 32)       scatter back, zeros at masked pos
                Reshape: (2, 1024, 32) -> (2, 32, 32, 32)    F1 with gaps

            Stage 2 (no removal):
                OPE:     (2, 32, 32, 32) -> (2, 256, 64)     stride-2 conv pools over gaps
                Blocks:  (2, 256, 64) -> (2, 256, 64)        all 256 tokens processed
                Reshape: -> (2, 64, 16, 16)                   F2

            Stages 3-4: same as Stage 2, all tokens processed normally
        """
        # x: (B, in_channels, H, W)
        features = []

        for i in range(self.num_stages):
            # --- Patch embedding ---
            # Converts spatial feature map to token sequence
            #
            # Stage 1: (B, 1, 128, 128) -> (B, 1024, 32) + H'=32, W'=32
            # Stage 2: (B, 32, 32, 32)  -> (B, 256, 64)  + H'=16, W'=16
            # Stage 3: (B, 64, 16, 16)  -> (B, 64, 160)  + H'=8,  W'=8
            # Stage 4: (B, 160, 8, 8)   -> (B, 16, 256)  + H'=4,  W'=4
            x, H, W = self.patch_embeds[i](x)
            # x: (B, N, C_i)  where N = H * W

            # Total number of tokens and embedding dim at this stage
            # (needed for restore_tokens after Stage 1)
            B, N, C = x.shape

            # --- MAE token removal (Stage 1 only) ---
            # Remove prediction target tokens so the encoder never sees them.
            # The encoder processes only visible valid + invalid tokens.
            #
            # Example: x is (2, 1024, 32), keep_mask has ~512 ones per row
            #          remove_tokens -> kept: (2, 512, 32), indices: (2, 512)
            #
            # ESA needs H, W for reshaping tokens to 2D (spatial reduction conv).
            # After removal, the tokens are a sparse subset -- H, W still refer to
            # the FULL grid dimensions. ESA's reshape will produce a grid with gaps,
            # which is fine: the reduction conv pools over them.
            gather_indices = None
            if i == 0 and keep_mask is not None:
                x, gather_indices = TokenMasking.remove_tokens(x, keep_mask)
                # x: (B, N_kept, C_i)  where N_kept ≈ N * 0.5
                # gather_indices: (B, N_kept) -- original positions for scatter-back

            # --- Transformer blocks ---
            # Each block: pre-norm ESA + residual, pre-norm MixFFN + residual
            #
            # Without masking: (B, N, C_i) -> (B, N, C_i)             all tokens
            # With masking:    (B, N_kept, C_i) -> (B, N_kept, C_i)   only visible tokens
            #
            # H, W are the full grid dims. ESA uses them to reshape tokens to 2D
            # for the spatial reduction conv. With token removal, the reshape produces
            # a grid that's smaller than H*W -- but ESA only needs H, W for the
            # reduction conv on the K/V path, which operates on the full x reshaped,
            # so we pass the actual token grid dims instead.
            for block in self.blocks[i]:
                x = block(x, H, W)
            # x: (B, N or N_kept, C_i)

            # --- Stage output norm ---
            x = self.norms[i](x)
            # x: (B, N or N_kept, C_i)

            # --- Restore masked tokens (Stage 1 only) ---
            # Scatter visible tokens back into the full N-token grid.
            # Prediction positions get zeros -- the decoder must reconstruct these.
            #
            # Example: x is (2, 512, 32), gather_indices is (2, 512)
            #          restore -> (2, 1024, 32) with 512 zero tokens
            if i == 0 and gather_indices is not None:
                x = TokenMasking.restore_tokens(x, gather_indices, N, C)
                # x: (B, N, C_i) -- full grid, zeros at prediction positions

            # --- Reshape tokens back to 2D spatial form ---
            # (B, N, C_i) -> transpose -> (B, C_i, N) -> reshape -> (B, C_i, H, W)
            #
            # This serves two purposes:
            #   1. Stored as this stage's feature map for the decoder
            #   2. Becomes the spatial input to the next stage's OverlapPatchEmbedding
            #
            # After restore_tokens, this always uses the FULL grid dims:
            #   Stage 1: (B, 1024, 32) -> (B, 32, 32, 32)   includes zero-filled gaps
            #   Stage 2: (B, 256, 64)  -> (B, 64, 16, 16)   no gaps, all tokens present
            B_cur, N_cur, C_cur = x.shape
            x = x.transpose(1, 2).reshape(B_cur, C_cur, H, W)
            # x: (B, C_i, H', W')

            features.append(x)

        # features: [F1, F2, F3, F4] at 4 spatial scales
        # F1 has zeros at prediction positions (if masking was applied)
        # F2-F4 are fully populated (Stage 2+ process all tokens)
        return features
