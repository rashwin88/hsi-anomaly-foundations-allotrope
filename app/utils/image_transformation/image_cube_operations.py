"""
Perform transformation operations on image cubes
"""

from typing import Dict, List, Literal
import torch
import numpy as np
import copy

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
        cube: np.ndarray | torch.Tensor,
        from_format: CubeRepresentation,
        to_format: CubeRepresentation,
        output_form: Literal["tensor", "numpy"] = "numpy",
    ) -> torch.Tensor | np.ndarray:
        """
        Converts a cube from one format to another
        """
        # If the input is a numpy array, convert it to a tensor
        if isinstance(cube, np.ndarray):
            cube = torch.from_numpy(cube).float().to(self.device)
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
        if output_form == "tensor":
            return transformed
        elif output_form == "numpy":
            # Detach the tensor from the computational graph and move it to the CPU
            return transformed.detach().cpu().numpy()
        else:
            raise ValueError(f"Invalid output form: {output_form}")
