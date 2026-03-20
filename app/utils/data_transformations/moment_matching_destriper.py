"""
Moment-matching destriper for hyperspectral cubes in BSQ format.

Corrects per-detector-element gain/offset artifacts by matching each
stripe-aligned column's mean and std to the scene-wide statistics.

Operates along an arbitrary stripe angle — each pixel is binned by its
perpendicular distance to a reference line at the stripe angle. This
maps each pixel to its "detector column" in the original pushbroom
geometry, regardless of orthorectification rotation.

Uses one iteration of 2-sigma outlier rejection so real scene features
don't contaminate the column statistics.
"""

import logging

import numpy as np

from app.abstract_classes.data_transformer import DataTransformer
from app.models.dataset.transformations import Transformation

logger = logging.getLogger(__name__)

MIN_BIN_PIXELS = 20
DEGENERACY_THRESHOLD = 1e-10


class MomentMatchingDestriper(DataTransformer):
    """
    Per-band, per-stripe-column moment-matching destriper with robust
    outlier rejection. Operates along an arbitrary stripe angle.
    """

    def __init__(self):
        super().__init__(
            transformation_category=Transformation.MOMENT_MATCHING_DESTRIPE
        )

    def transform(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Destripe a BSQ reflectance cube via moment-matching along the
        stripe direction.

        Args:
            input_data:     (B, H, W) reflectance cube.
            validity_mask:  (B, H, W) binary mask, 1 = valid. Required.
            stripe_angle:   float, stripe angle in degrees. Required.

        Returns:
            Destriped cube, same shape and dtype as input_data.
        """
        validity_mask = kwargs.get("validity_mask")
        if validity_mask is None:
            raise ValueError("validity_mask is required.")

        stripe_angle = kwargs.get("stripe_angle")
        if stripe_angle is None:
            raise ValueError("stripe_angle is required.")

        B, H, W = input_data.shape
        output = input_data.copy()

        # Compute perpendicular distance of each pixel to a line at
        # stripe_angle through the image center → "detector column" index
        bin_map = self._compute_bin_map(H, W, stripe_angle)

        # Precompute bin → flat pixel indices once (avoids O(H*W*num_bins))
        bin_to_pixels = self._precompute_bin_indices(bin_map)

        for b in range(B):
            self._destripe_band(output[b], validity_mask[b], bin_to_pixels)

        logger.info(
            "Moment-matching destriped %d bands at %.1f° (%d bins).",
            B, stripe_angle, len(bin_to_pixels),
        )
        return output.astype(input_data.dtype)

    @staticmethod
    def _compute_bin_map(H: int, W: int, stripe_angle: float) -> np.ndarray:
        """
        For each pixel, compute its integer bin index based on
        perpendicular distance to a line at stripe_angle through center.
        """
        cy, cx = H / 2.0, W / 2.0
        angle_rad = np.deg2rad(stripe_angle)

        # Normal vector to the stripe direction
        nx = -np.sin(angle_rad)
        ny = np.cos(angle_rad)

        rows, cols = np.mgrid[0:H, 0:W]
        # Perpendicular distance (continuous)
        dist = (cols - cx) * nx + (rows - cy) * ny

        # Quantize to integer bins
        return np.round(dist).astype(int)

    @staticmethod
    def _precompute_bin_indices(bin_map: np.ndarray) -> dict[int, np.ndarray]:
        """Map each bin index to the flat pixel indices that belong to it."""
        flat = bin_map.ravel()
        bin_to_pixels = {}
        for idx in np.unique(flat):
            bin_to_pixels[idx] = np.where(flat == idx)[0]
        return bin_to_pixels

    @staticmethod
    def _destripe_band(
        band: np.ndarray,
        validity: np.ndarray,
        bin_to_pixels: dict[int, np.ndarray],
    ) -> None:
        """In-place moment-match a single (H, W) band slice per bin."""
        flat_band = band.ravel()
        flat_valid = validity.ravel().astype(bool)

        scene_values = flat_band[flat_valid]
        if scene_values.size == 0:
            return

        mu_scene = scene_values.mean()
        sigma_scene = scene_values.std()

        if sigma_scene < DEGENERACY_THRESHOLD:
            return

        for pixel_indices in bin_to_pixels.values():
            valid_in_bin = pixel_indices[flat_valid[pixel_indices]]

            if valid_in_bin.size < MIN_BIN_PIXELS:
                continue

            col_values = flat_band[valid_in_bin]

            # First pass: raw bin stats
            mu_j = col_values.mean()
            sigma_j = col_values.std()

            if sigma_j < DEGENERACY_THRESHOLD:
                continue

            # Outlier rejection: exclude pixels > 2σ from bin mean
            inlier_mask = np.abs(col_values - mu_j) <= 2.0 * sigma_j

            if inlier_mask.sum() < MIN_BIN_PIXELS:
                continue

            # Second pass: robust bin stats from inliers
            mu_j = col_values[inlier_mask].mean()
            sigma_j = col_values[inlier_mask].std()

            if sigma_j < DEGENERACY_THRESHOLD:
                continue

            # Apply correction to ALL valid pixels in the bin
            flat_band[valid_in_bin] = (
                (col_values - mu_j) * (sigma_scene / sigma_j) + mu_scene
            )
