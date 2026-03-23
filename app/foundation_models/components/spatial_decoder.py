import torch
import torch.nn as nn


class SpatialDecoderBlock(nn.Module):
    """
    Single spatial upsample step using ConvTranspose2d.

    ConvTranspose2d is the learnable inverse of a strided Conv2d. It works by:
      1. Inserting zeros between input pixels (spacing them out by stride)
      2. Applying a learned kernel over this expanded grid
    Each output pixel becomes a learned blend of nearby input pixels.

    With (K=4, S=2, P=1) the output exactly doubles:

        H_out = (H_in - 1) * S - 2*P + K
              = (H_in - 1) * 2 - 2*1 + 4
              = 2 * H_in

    This mirrors the encoder's (K=4, S=2, P=1) Conv2d which exactly halves,
    making the encoder-decoder pair symmetric.

    BatchNorm in intermediate blocks:
        BatchNorm normalizes each channel to zero-mean unit-variance across
        the batch, then applies a learned affine transform (scale + shift).
        This keeps activations in a stable range between decoder stages,
        preventing gradient vanishing/explosion as signals pass through
        multiple layers. It also acts as a mild regularizer.

    Why BatchNorm is SKIPPED on the final block:
        The final layer's job is to reconstruct the original input values
        (e.g. surface temperatures in Kelvin). BatchNorm would force the
        output to zero-mean unit-variance per channel — but real thermal
        data is NOT zero-mean or unit-variance. For example, if all pixels
        in a scene are ~300K, BN would center them around 0, making it
        impossible to reconstruct the true values. Similarly, GELU would
        clip negative values, but temperature anomalies can be negative.
        Skipping both lets the final ConvTranspose2d output any value range
        the data requires.
    """

    def __init__(self, in_channels, out_channels, final=False):
        super().__init__()
        # Constructed slightly differently from the encode block but the principle remains the same.
        layers = [
            # (B, in_channels, H, W) → (B, out_channels, 2*H, 2*W)
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if not final:
            # Intermediate block: stabilize activations before next stage
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
        # Final block: raw output, free to match any value distribution
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SpatialDecoder(nn.Module):
    """
    Stack of SpatialDecoderBlocks that mirrors the SpatialEncoder back up
    to the original resolution.

    The channel progression is the reverse of the encoder. We build it by
    generating the encoder's channel list, reversing it, and appending
    out_channels at the end.

    For base_channels=32, num_stages=3:

      Encoder channels: [1, 32, 64, 128]   (ascending)
      Decoder channels: [128, 64, 32, 1]   (descending — mirror)

      channels = [bc * 2^i for i in range(num_stages)][::-1] + [out_channels]
               = [32*1, 32*2, 32*4][::-1] + [1]
               = [128, 64, 32] + [1]
               = [128, 64, 32, 1]

      Stage 0: (B, 128, H/8, W/8) → (B, 64, H/4, W/4)   BN + GELU
      Stage 1: (B, 64,  H/4, W/4) → (B, 32, H/2, W/2)   BN + GELU
      Stage 2: (B, 32,  H/2, W/2) → (B, 1,  H,   W)     final=True, no BN/GELU

    The last stage is marked final=True so the output is unconstrained
    and can match the original input distribution.
    """

    def __init__(self, out_channels=1, base_channels=32, num_stages=3):
        super().__init__()
        # Reverse the encoder's channel progression and append output channels
        channels = [base_channels * (2 ** i) for i in range(num_stages)][::-1] + [out_channels]
        self.stages = nn.Sequential(
            *[SpatialDecoderBlock(channels[i], channels[i + 1], final=(i == num_stages - 1))
              for i in range(num_stages)]
        )

    def forward(self, x):
        return self.stages(x)
