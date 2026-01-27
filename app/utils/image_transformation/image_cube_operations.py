import torch
import numpy as np
from app.models.images.cube_representation import CubeRepresentation
from typing import Dict, List, Literal
from app.utils.torch_helpers.device_selection import get_device


class ImageCubeOperations:
    def __init__(self):
        self.dimension_map: Dict[CubeRepresentation, Dict[str, int]] = (
            self._build_dimension_map()
        )
        self.arrangements: Dict[CubeRepresentation, List[str]] = (
            self._build_dimensional_arrangements()
        )
        self.device = get_device()
        print(f"Using device: {self.device}")

    def _build_dimension_map(self) -> Dict[CubeRepresentation, Dict[str, int]]:
        """
        Builds a dimension map for the cube representations
        """
        dimension_map = {
            CubeRepresentation.BIL: {
                "H": 0,
                "W": 2,
                "C": 1,
            },
            CubeRepresentation.BIP: {
                "H": 0,
                "W": 1,
                "C": 2,
            },
            CubeRepresentation.BSQ: {
                "H": 1,
                "W": 2,
                "C": 0,
            },
        }
        return dimension_map

    def _build_dimensional_arrangements(self) -> Dict[CubeRepresentation, List[str]]:
        """
        Builds a dimensional arrangement for the cube representations
        """
        dimensional_arrangements = {
            CubeRepresentation.BIL: ["H", "C", "W"],
            CubeRepresentation.BIP: ["H", "W", "C"],
            CubeRepresentation.BSQ: ["C", "H", "W"],
        }
        return dimensional_arrangements

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


if __name__ == "__main__":
    image_cube_operations = ImageCubeOperations()
    cube = np.random.rand(30, 2000, 1000)
    print(cube.shape)
    transformed = image_cube_operations.convert_cube(
        cube, CubeRepresentation.BSQ, CubeRepresentation.BIL
    )
    print(transformed.shape)
