"""
Nearest-valid pixel fill for boundary artifact elimination.

The SegFormer's Overlap Patch Embedding (OPE) uses a 7x7 kernel that straddles
valid/invalid pixel boundaries. Invalid pixels at 0 degrees normalise to ~-2.3,
creating a thermal cliff against valid pixels in the [-0.7, +1.7] range. The OPE
produces contaminated tokens at boundaries, and attention broadcasts this
contamination globally, producing elevated reconstruction error along all
boundaries. These boundary artefacts dominate the anomaly ranking, crowding
out real detections.

Solution: replace every invalid pixel with the value of its nearest valid pixel
using Euclidean distance. The OPE's receptive field at boundaries sees a smooth
continuation of the nearby surface rather than a cliff.

Properties:
    - Valid pixels are guaranteed unchanged (nearest valid = self, distance = 0)
    - Filled values are always physically plausible (they come from real
      measurements, not interpolation or averaging)
    - Preserves local temperature gradients at boundaries (unlike global mean fill)
    - Runs on CPU via scipy — suitable for DataLoader workers with no GPU overhead

What changes:
    - Invalid pixels contain physically plausible fill values instead of zeros
    - The OPE produces clean tokens at all boundaries
    - Boundary artefacts in the residual map are eliminated

What doesn't change:
    - The validity masks are preserved — the loss still excludes filled pixels
    - The model architecture — no changes to encoder, decoder, or pixel predictor
    - The shards on disk — fill is applied on-the-fly
    - GPU memory or compute
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def nearest_valid_fill(pixels, mask):
    """
    Fill invalid pixels with the value of their nearest valid pixel
    using Euclidean distance.

    Args:
        pixels: (C, H, W) numpy array — temperature values (or any per-pixel data)
        mask:   (1, H, W) numpy array — validity (1=valid, 0=invalid)

    Returns:
        filled: (C, H, W) numpy array — invalid pixels replaced with
                nearest valid value. Valid pixels unchanged.

    Example (1D, single channel):
        pixels = [30, 32, 0,  0,  0,  25, 28]
        mask   = [ 1,  1, 0,  0,  0,   1,  1]

        Nearest valid for position 2 → position 1 (value 32)
        Nearest valid for position 3 → position 1 or 5 (equidistant, impl picks one)
        Nearest valid for position 4 → position 5 (value 25)

        filled = [30, 32, 32, 32, 25, 25, 28]

        The OPE now sees a smooth transition 32→32→25→25 instead of 32→0→0→25.

    Example (2D boundary):
        A cloud boundary at row 100. Rows 100+ are invalid (mask=0).
        Without fill: OPE tokens at rows 97-103 mix 30°C land with -2.3 normalised zeros.
        With fill: OPE tokens at rows 97-103 mix 30°C land with ~30°C fill (nearest valid).
        The OPE produces clean, uncontaminated tokens at the boundary.
    """
    mask_2d = mask.squeeze()  # (H, W)

    # distance_transform_edt operates on the invalid region (mask==0).
    # For each pixel where mask==0, it finds the nearest pixel where mask==1.
    #
    # Returns:
    #   distances: (H, W) — Euclidean distance to nearest valid pixel
    #   indices:   (2, H, W) — row and col coordinates of nearest valid pixel
    #
    # For valid pixels (mask==1): distance=0, indices point to self.
    # For invalid pixels (mask==0): distance>0, indices point to nearest valid.
    _, indices = distance_transform_edt(
        mask_2d == 0,
        return_distances=True,
        return_indices=True
    )
    # indices[0]: (H, W) — row of nearest valid pixel for each position
    # indices[1]: (H, W) — col of nearest valid pixel for each position

    # Look up nearest valid pixel's value for each channel
    # Valid pixels index into themselves (no change)
    # Invalid pixels index into their nearest valid neighbor
    filled = pixels.copy()
    for c in range(pixels.shape[0]):
        filled[c] = pixels[c][indices[0], indices[1]]

    return filled
