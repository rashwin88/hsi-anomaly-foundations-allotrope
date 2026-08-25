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


# Provider quality layers -> shard key, emitted only when include_labels=True.
# EnMAP-only: any vendable lacking the attribute (PRISMA) simply skips it, so
# no caller has to know which sensor carries which layers.
_LABEL_LAYERS: tuple[tuple[str, str], ...] = (
    ("cloud_mask", "label_cloud.npy"),
    ("cirrus_mask", "label_cirrus.npy"),
    ("haze_mask", "label_haze.npy"),
    ("cloud_shadow_mask", "label_cloud_shadow.npy"),
    ("snow_mask", "label_snow.npy"),
    # Categorical, NOT binary like the five above. Per the EnMAP product
    # spec: 0 = no-data (error), 1 = land, 2 = water, 3 = no-data (outside
    # the imaged swath, ~25% of every raster). Treating 0 or 3 as a class
    # would swamp training - the swath padding alone outnumbers cloud
    # pixels roughly 35:1. Consumers must map {0, 3} -> excluded.
    ("quality_classes_mask", "label_classes.npy"),
)


def patch_hyperspectral_vendable(
    vendable: Union[VendableHyperspectralDataset, VendableEnmapHyperspectralDataset],
    patching_plan: PatchingPlan,
    scene_id: str,
    sensor: Literal["prisma", "enmap"],
    include_labels: bool = False,
    pixel_dtype: type = np.float32,
) -> Generator[Dict, None, None]:
    """
    Yields patches from a hyperspectral vendable dataset.

    Each patch contains the reflectance cube, validity cube, wavelengths,
    and metadata.

    When `include_labels` is set, EnMAP's provider quality layers are
    emitted alongside as `label_*.npy` keys — the training targets for
    the segmentation model. Off by default, so the reconstruction
    sharding path (Indradhanu's training data) is unaffected.

    The `label_` prefix is deliberate: the thermal shards already carry
    `predicted_cloud_mask.npy`, which is a *model output*. These are
    provider ground truth. Do not conflate them.

    Only EnMAP carries these layers; PRISMA has no provider masks at all,
    so `include_labels=True` on a PRISMA vendable emits nothing extra.

    `pixel_dtype` defaults to float32 — the reconstruction lane's format, and
    what Indradhanu trained on. Segmentation sharding passes float16, halving
    the dominant term: reflectance sits in ~0-1, where float16 resolves to
    about 5e-4 relative, far below sensor noise. **A trainer reading float16
    shards must cast to float32 before the model**, which is fp32.
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
                pixel_dtype
            ),
            "validity_cube.npy": validity[
                :, row_coords:r_end, col_coords:c_end
            ].astype(np.int8),
            "wavelengths.npy": wavelengths,
        }

        if include_labels:
            # Stored (H, W) on the vendable; every other shard array is
            # channel-first, so add the leading axis here.
            for attr, key in _LABEL_LAYERS:
                mask = getattr(vendable, attr, None)
                if mask is not None:
                    output_object[key] = mask[
                        None, row_coords:r_end, col_coords:c_end
                    ].astype(np.uint8)

        yield output_object
