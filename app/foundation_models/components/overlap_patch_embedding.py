"""
Overlap Patch Embedding — the entry point for each SegFormer encoder stage.

Converts a 2D spatial feature map into a sequence of token embeddings using
an overlapping convolution. Unlike ViT's non-overlapping patches (kernel=stride,
zero overlap), SegFormer uses kernel > stride so neighboring tokens share
input pixels, preserving local continuity across token boundaries.

Example for Stage 1 (input -> first transformer stage):
    kernel=7, stride=4 -> overlap = 7 - 4 = 3 pixels shared between neighbors
    (B, 1, 128, 128) -> (B, 1024, 64)  -- 1024 tokens, each 64-dimensional

Example for Stage 2+ transitions (between transformer stages):
    kernel=3, stride=2 -> overlap = 3 - 2 = 1 token shared between neighbors
    (B, 64, 32, 32) -> (B, 256, 128) -- spatial halved, channels doubled
"""

import torch
import torch.nn as nn


class OverlapPatchEmbedding(nn.Module):
    """
    Patchify a 2D feature map into a token sequence via overlapping convolution.

    Data flow:
        (B, C_in, H, W)
            -> Conv2d(kernel=patch_size, stride=desired_compression, padding=patch_size//2)
        (B, C_out, H', W')          where H' = H // desired_compression
            -> flatten spatial dims, transpose to sequence form
        (B, H'*W', C_out)           N tokens, each C_out-dimensional
            -> LayerNorm over the embedding dimension
        (B, N, C_out)

    Also returns (H', W') since downstream modules (ESA, Mix-FFN) need to
    reshape tokens back into 2D spatial form for convolutions.

    Args:
        c_in:                 Input channels (1 for raw thermal, C_i for inter-stage)
        patch_size:           Convolution kernel size -- determines receptive field per token
        desired_compression:  Convolution stride -- determines spatial reduction factor
                              H' = H // desired_compression, W' = W // desired_compression
        c_out:                Output embedding dimension -- token feature size

    Spatial arithmetic (with padding = patch_size // 2):
        H_out = H_in // stride     (padding absorbs the kernel size)
        overlap = patch_size - desired_compression
    """

    def __init__(self, c_in, patch_size, desired_compression, c_out, padding=None):
        super().__init__()

        # Default padding = patch_size // 2 gives H_out = H_in / stride for overlapping patches.
        # For non-overlapping patches (kernel=stride), use padding=0 for exact H_out = H_in / stride.
        if padding is None:
            padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=patch_size,
            stride=desired_compression,
            padding=padding
        )

        # LayerNorm over the embedding dimension (c_out), applied per-token.
        # Each of the N tokens is independently normalized to zero-mean, unit-variance
        # across its c_out features. This stabilizes the input to attention layers.
        # (contrast: BatchNorm normalizes across the batch per-channel -- wrong for sequences)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x):
        # x: (B, C_in, H, W)

        x = self.proj(x)
        # x: (B, c_out, H', W')  where H' = H // stride, W' = W // stride

        B, _ , H_dash, W_dash = x.shape

        # Reshape from spatial (B, C, H', W') to sequence (B, N, C) form.
        # Cannot use view/reshape directly because the axes need to swap:
        #   Memory layout after Conv2d: [B][C][H'][W']
        #   Desired layout for attention: [B][N][C]  where N = H'*W'
        # Step 1: merge H' and W' -> (B, C, H'*W')
        x = x.flatten(2)
        # Step 2: swap channel and sequence dims -> (B, H'*W', C)
        x = x.transpose(1, 2)

        # Normalize each token's embedding independently
        out = self.norm(x)
        # out: (B, N, c_out)  where N = H' * W'

        # Return tokens plus spatial dims (needed to reshape back to 2D later)
        return out, H_dash, W_dash
