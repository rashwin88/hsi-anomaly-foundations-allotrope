"""
Given a patching plan and a vendable from PRISMA or EnMAP,
produces patches for webdataset sharding.

Sensor-agnostic: works with both VendableHyperspectralDataset (PRISMA)
and VendableEnmapHyperspectralDataset (EnMAP).

When the vendable has been resampled to a common wavelength grid
(via BandFilterConfig.common_wavelength_grid), wavelengths are already
ascending and identical across sensors. If not resampled, the patcher
sorts to ascending order as a fallback.
"""

from typing import Dict, Generator, Literal, Union

import numpy as np

from app.models.dataset.vendables import (
    VendableHyperspectralDataset,
    VendableEnmapHyperspectralDataset,
)
from app.models.patches.patching_response import PatchingPlan


def patch_hyperspectral_vendable(
    vendable: Union[VendableHyperspectralDataset, VendableEnmapHyperspectralDataset],
    patching_plan: PatchingPlan,
    scene_id: str,
    sensor: Literal["prisma", "enmap"],
) -> Generator[Dict, None, None]:
    """
    Yields patches from a hyperspectral vendable dataset.

    Each patch contains the reflectance cube, validity cube, wavelengths,
    and metadata.
    """
    wavelengths = np.array(vendable.band_cw_order, dtype=np.float64)

    # Ensure ascending wavelength order. When the vendable was resampled
    # to a common grid this is already the case; otherwise sort as fallback.
    if not np.all(np.diff(wavelengths) > 0):
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        cube = vendable.normalized_hyperspectral_cube[sort_idx]
        validity = vendable.validity_cube[sort_idx]
        families = [vendable.spectral_family_order[i] for i in sort_idx]
    else:
        cube = vendable.normalized_hyperspectral_cube
        validity = vendable.validity_cube
        families = vendable.spectral_family_order

    height = patching_plan.originating_request.height
    width = patching_plan.originating_request.width
    band_count = len(wavelengths)

    family_values = [f.value for f in families]

    for row_coords, col_coords in patching_plan.patch_coordinates:
        patch_key = f"{scene_id}#row_coord:{row_coords}#col_coord:{col_coords}"

        r_end = row_coords + height
        c_end = col_coords + width

        output_object = {
            "__key__": patch_key,
            "meta.json": {
                "scene_id": scene_id,
                "row_coords": row_coords,
                "col_coords": col_coords,
                "patch_height": height,
                "patch_width": width,
                "patch_stride": patching_plan.originating_request.stride,
                "sensor": sensor,
                "spectral_family_order": family_values,
                "band_count": band_count,
            },
            "pixels.npy": cube[:, row_coords:r_end, col_coords:c_end].astype(
                np.float32
            ),
            "validity_cube.npy": validity[
                :, row_coords:r_end, col_coords:c_end
            ].astype(np.int8),
            "wavelengths.npy": wavelengths,
        }

        yield output_object
