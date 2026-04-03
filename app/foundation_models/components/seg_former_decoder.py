"""
MLP Reconstruction Decoder -- fuses multi-scale encoder features into full-resolution output
with learned pixel-level upsampling via PixelShuffle.

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
        F_i: (B, C_embed, H_i, W_i) -> (B, C_embed, H/4, W/4)

    Step 3: Concatenate
        Stack all 4 unified feature maps along the channel dimension.
        (B, 4 * C_embed, H/4, W/4)

    Step 4: Fuse
        A 1x1 Conv reduces the concatenated channels back to C_embed.
        (B, 4 * C_embed, H/4, W/4) -> (B, C_embed, H/4, W/4)

    Step 5: Refine + PixelShuffle
        Two 3x3 convs refine spatial features at H/4 resolution (cheap, large
        effective receptive field). Then a final conv produces out_channels * 16
        channels — one for each sub-pixel in a 4x4 block. PixelShuffle(4)
        rearranges these into full-resolution pixel predictions.

        (B, C_embed, H/4, W/4)
            -> Conv3x3 + GELU           refine at coarse resolution
            -> Conv3x3                   produce out_channels * 16 channels
        (B, out_channels * 16, H/4, W/4)
            -> PixelShuffle(4)           rearrange to full resolution
        (B, out_channels, H, W)

        Unlike bilinear upsampling (which blurs by averaging neighbors),
        PixelShuffle produces each output pixel independently via learned
        weights. This is critical for point anomaly detection: a single
        anomalous pixel can get its own predicted value without being
        diluted by its 4x4 neighbors.

        No activation on the final conv — output must be unconstrained to
        represent any temperature value, including below the dataset mean
        (negative in normalized space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerDecoder(nn.Module):
    """
    Multi-scale feature fusion decoder with learned pixel-level upsampling.

    Takes the list of encoder feature maps [F1, F2, F3, F4] and produces
    a full-resolution reconstruction via PixelShuffle.

    Data flow:
        [F1, F2, F3, F4]                                  4 feature maps at different scales
            -> Conv1x1 each                                unify to C_embed channels
        [F1', F2', F3', F4']                               all (B, C_embed, H_i, W_i)
            -> bilinear upsample each to F1's resolution
        [F1', F2', F3', F4']                               all (B, C_embed, H/4, W/4)
            -> concatenate along channels                  (B, 4*C_embed, H/4, W/4)
            -> Conv1x1 fuse + GELU                         (B, C_embed, H/4, W/4)
            -> Conv3x3 refine + GELU                       (B, C_embed, H/4, W/4)
            -> Conv3x3 sub-pixel predict (no activation)   (B, out_channels*16, H/4, W/4)
            -> PixelShuffle(4)                             (B, out_channels, H, W)

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
        self.fuse_act = nn.GELU()

        # --- Step 5: Refine + learned pixel-level upsampling ---
        # Two convs at H/4 resolution: first refines features, second produces
        # out_channels * 16 sub-pixel values per spatial position.
        # PixelShuffle(4) then rearranges each group of 16 channels into a 4x4
        # spatial block, yielding full-resolution output.
        #
        # Each output pixel is independently predicted (no interpolation blurring).
        # This is critical for point anomaly detection: a 1-pixel anomaly at 45°C
        # surrounded by 30°C background gets its own predicted value, not a
        # smoothed average of the 4x4 block.
        #
        # No activation on the final conv — output represents raw reconstruction
        # values (temperatures in denormalized space).
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_dim, out_channels * 4 * 4, kernel_size=3, padding=1)
        )

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
                 Each pixel independently predicted via PixelShuffle, not interpolated.
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
        fused = self.fuse_act(fused)
        # fused: (B, decoder_dim, H/4, W/4)

        # --- Step 5: Refine at H/4 resolution + PixelShuffle to full resolution ---
        # Conv3x3 at H/4 has effective receptive field of 12x12 original pixels
        # — fine enough to learn spatial patterns around point anomalies
        refined = self.refine(fused)
        # refined: (B, out_channels * 16, H/4, W/4)
        # Each spatial position has 16 values — one for each pixel in the 4x4 block

        # PixelShuffle: rearrange channels into spatial dimensions
        # Groups of 4x4=16 channels become a 4x4 spatial block
        # (B, out_channels * 16, H/4, W/4) -> (B, out_channels, H, W)
        out = F.pixel_shuffle(refined, 4)
        # out: (B, out_channels, H, W) — each pixel independently predicted

        return out
