"""
ProgressiveDecoder: SegFormer's all-MLP fusion at 1/4 resolution, then two learned x2 upsampling stages that each
see a skip from the input at their own resolution - instead of emitting every 4 x 4 pixel block at once from one
token-level vector (SegFormerDecoder's PixelShuffle(4)).

Why (research/experiments/decoder-capacity/): FT04 reproduces pixels it can see barely better than pixels it must
predict (background error 0.056 visible vs 0.057 hidden on PRISMA set1) - its error is what the architecture cannot
render, not what the context cannot predict. Each stage-1 token squeezes 16 pixels x 32 channels into a 32-wide
vector and the SegFormer decoder has no full-resolution path back.

    features (encoder stages)  -> 1x1 projections + bilinear up + 1x1 fuse      (B, Cd, H/4, W/4)  [SegFormer's]
    -> x2 up, concat skip at H/2 -> 3x3 conv, GELU, 3x3 conv, GELU             (B, C2, H/2, W/2)
    -> x2 up, concat skip at H   -> 3x3 conv, GELU, 3x3 conv, GELU             (B, C1, H,   W)
    -> 1x1 conv                                                                (B, out_channels, H, W)

**The skip must not carry hidden pixels.** The caller passes the stem output with every pixel of a removed (hidden)
token set to 0; the H/2 skip is its 2 x 2 average, which stays inside a 4 px token, so hidden tokens are 0 at both
resolutions. Neighbouring visible pixels still reach a hidden pixel through the 3 x 3 convolutions - that is context,
as in the encoder. The layer names `linear_projections` and `fuse_conv` match SegFormerDecoder's, so an FT04
checkpoint warm-starts the 1/4-resolution part (`refine.*` is dropped).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GELU(), nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU())


class ProgressiveDecoder(nn.Module):
    def __init__(self, embed_dims, decoder_dim: int = 256, out_channels: int = 32, skip_channels: int = 32,
                 stage_dims: tuple[int, int] = (192, 96)):          # 1.50 M with FT04 widths (FT04's decoder: 2.16 M)
        super().__init__()
        self.linear_projections = nn.ModuleList([nn.Conv2d(d, decoder_dim, kernel_size=1) for d in embed_dims])
        self.fuse_conv = nn.Conv2d(decoder_dim * len(embed_dims), decoder_dim, kernel_size=1)
        self.fuse_act = nn.GELU()
        c2, c1 = stage_dims
        self.up_half = _block(decoder_dim + skip_channels, c2)
        self.up_full = _block(c2 + skip_channels, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, features: list, skip: torch.Tensor) -> torch.Tensor:
        """features: the encoder's four stage maps; skip: (B, skip_channels, H, W), hidden pixels already 0.
        Returns (B, out_channels, H, W)."""
        h, w = features[0].shape[2:]
        unified = [F.interpolate(p(f), size=(h, w), mode="bilinear", align_corners=False) if f.shape[2:] != (h, w)
                   else p(f) for p, f in zip(self.linear_projections, features)]
        x = self.fuse_act(self.fuse_conv(torch.cat(unified, dim=1)))                       # (B, Cd, H/4, W/4)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up_half(torch.cat([x, F.avg_pool2d(skip, 2)], dim=1))                     # (B, C2, H/2, W/2)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up_full(torch.cat([x, skip], dim=1))                                      # (B, C1, H, W)
        return self.head(x)
