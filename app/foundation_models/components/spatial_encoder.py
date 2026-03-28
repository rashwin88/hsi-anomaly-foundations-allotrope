import torch
import torch.nn as nn


class SpatialEncoderBlock(nn.Module):
    """
    Single spatial downsample step.

    Uses Conv2d with kernel_size=4, stride=2, padding=1 to exactly halve
    spatial dimensions. The formula:

        H_out = (H_in + 2*P - K) / S + 1
              = (H_in + 2*1 - 4) / 2 + 1
              = H_in / 2

    This (K=4, S=2, P=1) combination is chosen specifically because it
    produces clean 2x downsampling without fractional sizes. Other combos
    like (K=3, S=2, P=1) would require output_padding on the transpose side.

    BatchNorm normalizes per-channel statistics across the batch, stabilizing
    training by keeping activations in a well-behaved range.

    GELU provides nonlinearity — necessary because spatial patterns (edges,
    thermal gradients, textures) are inherently nonlinear and cannot be
    captured by purely linear transforms.

    Input patches must have H and W divisible by 2. When stacking multiple
    blocks (e.g. 3 stages), H and W must be divisible by 2^3 = 8.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4):
        super().__init__()
        # padding = (K - 2) // 2 ensures exact 2x spatial reduction for any even kernel size
        padding = (kernel_size - 2) // 2
        self.block = nn.Sequential(
            # (B, in_channels, H, W) → (B, out_channels, H/2, W/2)
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        return self.block(x)
    

class SpatialEncoder(nn.Module):
    """
    Stack of SpatialEncoderBlocks. Each stage halves spatial dims while
    doubling channels, so total information capacity per layer stays
    roughly constant (C * H * W ≈ const).

    The channel progression starts from in_channels and doubles the base
    at each stage. For base_channels=32, num_stages=3:

      channels = [1, 64, 128, 256]
                  │    │     │     └─ stage 2 output: (B, 256, H/8,  W/8)
                  │    │     └─────── stage 1 output: (B, 128, H/4,  W/4)
                  │    └───────────── stage 0 output: (B, 64,  H/2,  W/2)
                  └────────────────── input:          (B, 1,   H,    W)

    The first stage maps from in_channels (e.g. 1 for thermal) into the
    base channel space. Subsequent stages double from there using
    base_channels * 2^i for i in [1, num_stages].

    num_stages controls total spatial reduction = 2^num_stages:
      num_stages=3 → /8  downsampling (128 → 16)
      num_stages=4 → /16 downsampling (128 → 8)

    H and W must be divisible by 2^num_stages.
    """

    def __init__(self, in_channels=1, base_channels=32, num_stages=3, kernel_size=4):
        super().__init__()
        # Build channel progression: [in_channels, bc, bc*2, bc*4, ...]
        channels = [in_channels] + [base_channels * (2 ** i) for i in range( num_stages)]
        self.stages = nn.Sequential(
            *[SpatialEncoderBlock(channels[i], channels[i + 1], kernel_size=kernel_size) for i in range(num_stages)]
        )

    def forward(self, x):
        return self.stages(x)
