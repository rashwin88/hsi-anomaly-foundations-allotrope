"""
Perform transformation operations on image cubes
"""

from typing import Dict, List, Literal, Union
import torch
import numpy as np

from app.models.images.cube_representation import (
    CubeRepresentation,
    DIMENSION_MAPPING,
    DIMENSIONAL_ARRANGEMENTS,
)
from app.utils.torch_helpers.device_selection import get_device


class ImageCubeOperations:
    """
    Operations on an image cube to transform it into multiple
    representations.
    """

    def __init__(self):
        self.dimension_map: Dict[CubeRepresentation, Dict[str, int]] = DIMENSION_MAPPING
        self.arrangements: Dict[CubeRepresentation, List[str]] = (
            DIMENSIONAL_ARRANGEMENTS
        )
        self.device = get_device()
        print(f"Using device: {self.device}")

    def convert_cube(
        self,
        cube: Union[np.ndarray, np.ma.MaskedArray, torch.Tensor],
        from_format: CubeRepresentation,
        to_format: CubeRepresentation,
        output_form: Literal["tensor", "numpy"] = "numpy",
    ) -> Union[torch.Tensor, np.ndarray, np.ma.MaskedArray]:
        """
        Converts a cube from one format to another
        """
        # We will need to handled masked cubes in a special manner
        # The idea is that if the input is a masked array and the output from is numpy
        # Then, we must return as masked array as default behavior/
        # But when we take the array into a Tensor form,
        # the masking information is lost, so we must also Tensorify the mask and permute it
        # And then the permuted mask must be applied to the permuted output
        if isinstance(cube, np.ma.MaskedArray):
            is_masked_input = True
        else:
            is_masked_input = False
        mask_tensor = None
        # If the input is a numpy array, convert it to a tensor
        if is_masked_input:
            mask_array = np.ma.getmaskarray(cube)
            # Create a mask_tensor
            mask_tensor = torch.from_numpy(mask_array).to(self.device)
            cube_data = cube.data
            if cube_data.dtype == np.float64:
                cube_data = cube_data.astype(np.float32)
            cube = torch.from_numpy(cube_data).to(self.device)
        elif isinstance(cube, np.ndarray):
            if cube.dtype == np.float64:
                cube = cube.astype(np.float32)
            cube = torch.from_numpy(cube).to(self.device)
        else:
            if cube.dtype == torch.float64:
                cube = cube.float()
            cube = cube.to(self.device)
        # For the from format, get the dimension_map
        from_dim_map = self.dimension_map.get(from_format)
        to_dim_arrangement = self.arrangements.get(to_format)
        final_permutation_arrangement = tuple(
            from_dim_map.get(dim)
            for dim in to_dim_arrangement  # pyright: ignore[reportOptionalIterable]
        )
        transformed = cube.permute(*final_permutation_arrangement)
        transformed_mask = None
        if is_masked_input:
            # Transform the mask too.
            transformed_mask = mask_tensor.permute(*final_permutation_arrangement)
        if output_form == "tensor":
            return transformed
        elif output_form == "numpy":
            if is_masked_input:
                numpy_data = transformed.detach().cpu().numpy()
                numpy_mask = transformed_mask.detach().cpu().numpy().astype(bool)
                # Reapply the mask and then move on
                return np.ma.MaskedArray(data=numpy_data, mask=numpy_mask)
            # Detach the tensor from the computational graph and move it to the CPU
            return transformed.detach().cpu().numpy()
        else:
            raise ValueError(f"Invalid output form: {output_form}")
