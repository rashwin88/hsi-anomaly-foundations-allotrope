"""
Generating a patch plan for a cube
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

    def generate_patching_plan(
        self, input_cube: np.ndarray, request: PatchRequest
    ) -> PatchingPlan:
        """
        Generates a patching plan for a patch request
        """

        # Store variables
        cube_height = input_cube.shape[1]
        cube_width = input_cube.shape[2]

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
            if row + request.height >= cube_height:
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
            if col + request.width >= cube_width:
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
