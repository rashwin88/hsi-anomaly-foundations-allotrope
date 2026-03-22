"""
Given a patching plan and a vendable from landsat,
produces patches. Defined as a function
"""

from typing import Dict

from pystac import Item

from app.models.dataset.vendables import VendableThermalDataset
from app.models.patches.patching_response import PatchingPlan


def patch_landsat_vendable(
    vendable: VendableThermalDataset, patching_plan: PatchingPlan, stac_item: Item
) -> Dict:
    """
    Yields a single patch from the landsat data.
    """

    # Loop through each of the patching coordinates
    for patch_coords in patching_plan.patch_coordinates:
        row_coords = patch_coords[0]
        col_coords = patch_coords[1]
        # Unique patch key
        patch_key = f"{stac_item.id}#row_coord:{row_coords}#col_coord:{col_coords}"
        # Create the basic object and store metadata
        output_object = {}
        output_object["__key__"] = patch_key

        patch_metadata = {
            "scene_id": stac_item.id,
            "row_coords": row_coords,
            "col_coords": col_coords,
            "patch_height": patching_plan.originating_request.height,
            "patch_width": patching_plan.originating_request.width,
            "patch_stride": patching_plan.originating_request.stride,
            "bands": patching_plan.originating_request.input_cube[0],
        }
        output_object["meta.json"] = patch_metadata
        # Now add the actual numpy files
        output_object["pixels.npy"] = vendable.normalized_thermal_cube[
            :,
            row_coords : row_coords + patching_plan.originating_request.height,
            col_coords : col_coords + patching_plan.originating_request.width,
        ]
        output_object["pixels.npy"] = vendable.normalized_thermal_cube[
            :,
            row_coords : row_coords + patching_plan.originating_request.height,
            col_coords : col_coords + patching_plan.originating_request.width,
        ]
        output_object["validity_cube.npy"] = vendable.validity_cube[
            :,
            row_coords : row_coords + patching_plan.originating_request.height,
            col_coords : col_coords + patching_plan.originating_request.width,
        ]
        output_object["predicted_cloud_mask.npy"] = vendable.cloud_mask[
            :,
            row_coords : row_coords + patching_plan.originating_request.height,
            col_coords : col_coords + patching_plan.originating_request.width,
        ]
        output_object["pure_validity_mask.npy"] = vendable.pure_validity_mask[
            :,
            row_coords : row_coords + patching_plan.originating_request.height,
            col_coords : col_coords + patching_plan.originating_request.width,
        ]

        if vendable.custom_quality_mask is not None:
            # custom_quality_mask: 0 = invalid, 1 = valid.
            # Must be multiplied with pure_validity_mask before use.
            output_object["custom_quality_mask.npy"] = (
                vendable.custom_quality_mask[
                    :,
                    row_coords : row_coords + patching_plan.originating_request.height,
                    col_coords : col_coords + patching_plan.originating_request.width,
                ]
            )
        if vendable.provider_cloud_presence is not None:
            output_object["provider_cloud_presence.npy"] = (
                vendable.provider_cloud_presence[
                    :,
                    row_coords : row_coords + patching_plan.originating_request.height,
                    col_coords : col_coords + patching_plan.originating_request.width,
                ]
            )
        if vendable.provider_water_presence is not None:
            output_object["provider_water_presence.npy"] = (
                vendable.provider_water_presence[
                    :,
                    row_coords : row_coords + patching_plan.originating_request.height,
                    col_coords : col_coords + patching_plan.originating_request.width,
                ]
            )
        if vendable.provider_snow_presence is not None:
            output_object["provider_snow_presence.npy"] = (
                vendable.provider_snow_presence[
                    :,
                    row_coords : row_coords + patching_plan.originating_request.height,
                    col_coords : col_coords + patching_plan.originating_request.width,
                ]
            )

        # Just yield the output object
        yield output_object
