"""
Decides WHERE to cut a cube into patches. Coordinates only, no pixels.

Given a cube shape, a patch size and a stride, returns the top-left corner of
every patch. The per-sensor cutters (landsat_patcher, hyperspectral_patcher) do
the actual slicing; keeping the plan separate means the geometry is testable
without loading a scene, and the same plan can be replayed over any cube of
matching shape.

Also used at INFERENCE time, not just for training data: full-scene inference
tiles a scene the same way, reconstructs each tile, and overlap-averages the
results.

Stride controls overlap. The convention throughout the training pipeline is
stride = size // 2, i.e. 50% overlap, which means every interior pixel is
covered by several patches and boundary artefacts get averaged down.

Edge handling worth knowing: when the last stride step would run off the edge,
the final corner is SNAPPED back so the patch ends flush with the boundary. That
guarantees full coverage with no partial patches, at the cost of a final row and
column that overlap their neighbours more than the stride implies. So the patch
count is not simply ceil(extent / stride).
"""

import logging

from typing import List
import numpy as np

from app.models.patches.patching_request import PatchRequest
from app.models.patches.patching_response import PatchingPlan


class PatchPlanGenerator:
    """
    A class to generate patching plans given an input BSQ cube
    """

    def __init__(self):
        pass

    def generate_patching_plan(self, request: PatchRequest) -> PatchingPlan:
        """
        Generates a patching plan for a patch request
        """

        # Store variables
        cube_height = request.input_cube[1]
        cube_width = request.input_cube[2]

        if request.stride <= 0:
            raise ValueError("Stride must be greater than 0 to avoid an infinite loop.")
        # Perform basic sanity checks
        if cube_height < request.height:
            raise ValueError(
                f"The input cube has a height : {cube_height} while the patch requested is larger : {request.height}"
            )
        if cube_width < request.width:
            raise ValueError(
                f"The input cube has a height : {cube_width} while the patch requested is larger : {request.width}"
            )

        # the idea is to first generate the x coords
        row_coords = []
        row = 0
        while True:
            # check if row is outside the height bounds
            if row >= cube_height:
                break
            # Check if the patch lies outside the cube
            if row + request.height > cube_height:
                # In the even that the patch lies outside the cube
                # Set the patch row coord to be such that row + patch_height  = height. So row = Height - patch_height
                # this is the last patch basically
                row_coords.append(cube_height - request.height)
                # Then break
                break
            # Otherwise we are good and we can just move on
            row_coords.append(row)
            row += request.stride

        # the same logic applies for the y_coords too
        col_coords = []
        col = 0
        while True:
            # check if col is outside the width bounds
            if col >= cube_width:
                break
            # Check if the patch lies outside the cube
            if col + request.width > cube_width:
                # In the even that the patch lies outside the cube
                # Set the patch x coord to be such that col + patch_width  = width. So x = width - patch_width
                # this is the last patch basically
                col_coords.append(cube_width - request.width)
                # Then break
                break
            # Otherwise we are good and we can just move on
            col_coords.append(col)
            col += request.stride

        # Now we have x and y we can basically permute
        final_coords = [(r, c) for r in row_coords for c in col_coords]

        return PatchingPlan(originating_request=request, patch_coordinates=final_coords)
