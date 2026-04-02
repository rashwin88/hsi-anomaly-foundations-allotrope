"""
MLP Reconstruction Decoder -- fuses multi-scale encoder features into full-resolution output.

The SegFormer encoder produces 4 feature maps at different spatial scales:
    F1: (B, C1, H/4,  W/4)   -- fine-grained local detail
    F2: (B, C2, H/8,  W/8)   -- medium-scale patterns
    F3: (B, C3, H/16, W/16)  -- coarse spatial structure
    F4: (B, C4, H/32, W/32)  -- global context

The decoder fuses these into a single full-resolution reconstruction:

    Step 1: Unify channels
        Each F_i has a different channel dim (C1, C2, C3, C4).
        A 1x1 Conv projects each to a common dimension C_embed.
        F_i: (B, C_i, H_i, W_i) -> (B, C_embed, H_i, W_i)

    Step 2: Upsample to common resolution
        All feature maps are bilinearly upsampled to Stage 1's resolution (H/4, W/4).
        This is the coarsest resolution that still preserves fine detail from F1.
        F_i: (B, C_embed, H_i, W_i) -> (B, C_embed, H/4, W/4)

    Step 3: Concatenate
        Stack all 4 unified feature maps along the channel dimension.
        (B, 4 * C_embed, H/4, W/4)

    Step 4: Fuse
        A 1x1 Conv reduces the concatenated channels back to C_embed.
        (B, 4 * C_embed, H/4, W/4) -> (B, C_embed, H/4, W/4)

    Step 5: Upsample to full resolution
        4x bilinear upsampling to match the original input size.
        (B, C_embed, H/4, W/4) -> (B, C_embed, H, W)

    Step 6: Final projection
        A 3x3 Conv projects to the output channel count (1 for thermal).
        NO activation function, NO BatchNorm on this final layer.
        The output must be free to represent any value in the data range,
        including negative values (temperatures below the dataset mean in
        normalized space). Any nonlinearity would clip the output range.
        (B, C_embed, H, W) -> (B, out_channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerDecoder(nn.Module):
    """
    Multi-scale feature fusion decoder for reconstruction.

    Takes the list of encoder feature maps [F1, F2, F3, F4] and produces
    a full-resolution reconstruction of the input.

    Data flow:
        [F1, F2, F3, F4]                                   4 feature maps at different scales
            -> Conv1x1 each                                 unify to C_embed channels
        [F1', F2', F3', F4']                                all (B, C_embed, H_i, W_i)
            -> bilinear upsample each to F1's resolution
        [F1', F2', F3', F4']                                all (B, C_embed, H/4, W/4)
            -> concatenate along channels                   (B, 4*C_embed, H/4, W/4)
            -> Conv1x1 fuse                                 (B, C_embed, H/4, W/4)
            -> 4x bilinear upsample                         (B, C_embed, H, W)
            -> Conv 3x3 project (no activation)             (B, out_channels, H, W)

    Args:
        embed_dims:    List of encoder channel dims per stage, e.g. [32, 64, 160, 256]
        decoder_dim:   Common channel dimension for fusion (C_embed), e.g. 256
        out_channels:  Output channels (1 for single-band thermal reconstruction)
    """

    def __init__(self, embed_dims, decoder_dim=256, out_channels=1):
        super().__init__()

        # --- Step 1: Channel unification ---
        # One 1x1 Conv per stage to project from C_i to decoder_dim
        # 1x1 Conv is equivalent to a per-pixel Linear: no spatial mixing, only channel mixing
        self.linear_projections = nn.ModuleList([
            nn.Conv2d(embed_dim, decoder_dim, kernel_size=1)
            for embed_dim in embed_dims
        ])

        # --- Step 4: Fuse concatenated features ---
        # After concatenation: 4 * decoder_dim channels -> decoder_dim
        self.fuse_conv = nn.Conv2d(decoder_dim * len(embed_dims), decoder_dim, kernel_size=1)
        self.fuse_norm = nn.BatchNorm2d(decoder_dim)
        self.fuse_act = nn.GELU()

        # --- Step 6: Final projection to output channels ---
        # 3x3 Conv with padding=1 preserves spatial dims
        # NO activation, NO BatchNorm -- output must be unconstrained
        self.final_proj = nn.Conv2d(decoder_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        """
        Args:
            features: list of 4 feature maps from the encoder
                F1: (B, C1, H/4,  W/4)
                F2: (B, C2, H/8,  W/8)
                F3: (B, C3, H/16, W/16)
                F4: (B, C4, H/32, W/32)

        Returns:
            out: (B, out_channels, H, W) -- full-resolution reconstruction
        """
        # Target spatial size: Stage 1's resolution (the finest encoder output)
        # F1 is features[0]: (B, C1, H/4, W/4)
        target_h, target_w = features[0].shape[2], features[0].shape[3]

        # --- Steps 1 & 2: Unify channels and upsample to common resolution ---
        unified = []
        for i, feat in enumerate(features):
            # feat: (B, C_i, H_i, W_i)

            # 1x1 Conv: project to common decoder_dim
            proj = self.linear_projections[i](feat)
            # proj: (B, decoder_dim, H_i, W_i)

            # Bilinear upsample to Stage 1's spatial size
            # F1 is already at target size (no-op), F2 is 2x, F3 is 4x, F4 is 8x
            if proj.shape[2] != target_h or proj.shape[3] != target_w:
                proj = F.interpolate(proj, size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)
            # proj: (B, decoder_dim, H/4, W/4)

            unified.append(proj)

        # --- Step 3: Concatenate along channel dimension ---
        fused = torch.cat(unified, dim=1)
        # fused: (B, 4 * decoder_dim, H/4, W/4)

        # --- Step 4: Fuse with 1x1 Conv ---
        fused = self.fuse_conv(fused)
        fused = self.fuse_norm(fused)
        fused = self.fuse_act(fused)
        # fused: (B, decoder_dim, H/4, W/4)

        # --- Step 5: Upsample to full input resolution ---
        # 4x bilinear upsample (Stage 1 is at H/4, so 4x gets back to H)
        fused = F.interpolate(fused, scale_factor=4, mode='bilinear', align_corners=False)
        # fused: (B, decoder_dim, H, W)

        # --- Step 6: Final projection ---
        # 3x3 Conv to output channels, NO activation (output must be unconstrained)
        out = self.final_proj(fused)
        # out: (B, out_channels, H, W)

        return out
