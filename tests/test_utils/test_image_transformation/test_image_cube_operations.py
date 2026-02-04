"""
Critical Test: Checks whether image cube operations
are performed properly
"""

import pytest
import numpy as np
import torch
from app.utils.image_transformation.image_cube_operations import ImageCubeOperations
from app.models.images.cube_representation import CubeRepresentation


@pytest.fixture
def numpy_cubes():
    """
    Creates a set of numpy cubes for testing
    """
    return {
        CubeRepresentation.BIL: np.random.rand(2000, 3, 2000),
        CubeRepresentation.BIP: np.random.rand(2000, 2000, 3),
        CubeRepresentation.BSQ: np.random.rand(3, 2000, 2000),
    }


@pytest.fixture
def tensor_cubes():
    """
    Creates a set of tensor cubes for testing
    """
    return {
        CubeRepresentation.BIL: torch.from_numpy(np.random.rand(2000, 3, 2000)).to(
            torch.float64
        ),
        CubeRepresentation.BIP: torch.from_numpy(np.random.rand(2000, 2000, 3)).to(
            torch.float64
        ),
        CubeRepresentation.BSQ: torch.from_numpy(np.random.rand(3, 2000, 2000)).to(
            torch.float64
        ),
    }


@pytest.fixture
def masked_numpy_array() -> np.ma.MaskedArray:
    """
    Fixture for a masked numpy array
    """
    # The shape is BSQ
    shape = (3, 2000, 2000)
    cube = np.random.randint(0, 100, size=shape)
    mask = np.random.rand(*shape) < 0.20
    return np.ma.masked_array(cube, mask=mask)


def test_array_inputs_array_outputs(numpy_cubes):
    """
    Tests to check if array inputs under various configurations
    Produce outputs in the desired configurations if specified as numpy
    arrays
    """
    transformer = ImageCubeOperations()
    # BIL to other formats
    input_cube = numpy_cubes[CubeRepresentation.BIL]

    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIP,
        output_form="numpy",
    )
    assert bip_output.shape == (2000, 2000, 3)
    # Make sure that the input cube is not modified
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIL])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BSQ,
        output_form="numpy",
    )
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIL])

    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIL,
        output_form="numpy",
    )
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIL])

    # Perform the same tests for BIP to other formats
    input_cube = numpy_cubes[CubeRepresentation.BIP]
    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BIL,
        output_form="numpy",
    )
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIP])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BSQ,
        output_form="numpy",
    )
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIP])

    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BIP,
        output_form="numpy",
    )
    assert bip_output.shape == (2000, 2000, 3)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIP])

    # Perform the same tests for BSQ to other formats
    input_cube = numpy_cubes[CubeRepresentation.BSQ]
    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIL,
        output_form="numpy",
    )
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BSQ])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BSQ,
        output_form="numpy",
    )
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BSQ])

    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIP,
        output_form="numpy",
    )
    assert bip_output.shape == (2000, 2000, 3)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BSQ])


def test_array_inputs_tensor_outputs(numpy_cubes):
    """
    Tests to check if array inputs under various configurations
    Produce outputs in the desired configurations if specified as numpy
    arrays
    """
    transformer = ImageCubeOperations()
    # BIL to other formats
    input_cube = numpy_cubes[CubeRepresentation.BIL]
    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(bip_output, torch.Tensor)
    assert bip_output.shape == (2000, 2000, 3)
    # Make sure that the input cube is not modified
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIL])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BSQ,
        output_form="tensor",
    )
    assert isinstance(bsq_output, torch.Tensor)
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIL])

    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIL,
        output_form="tensor",
    )
    assert isinstance(bil_output, torch.Tensor)
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIL])

    # Perform the same tests for BIP to other formats
    input_cube = numpy_cubes[CubeRepresentation.BIP]
    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BIL,
        output_form="tensor",
    )
    assert isinstance(bil_output, torch.Tensor)
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIP])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BSQ,
        output_form="tensor",
    )
    assert isinstance(bsq_output, torch.Tensor)
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIP])

    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(bip_output, torch.Tensor)
    assert bip_output.shape == (2000, 2000, 3)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BIP])

    # Perform the same tests for BSQ to other formats
    input_cube = numpy_cubes[CubeRepresentation.BSQ]
    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIL,
        output_form="tensor",
    )
    assert isinstance(bil_output, torch.Tensor)
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BSQ])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BSQ,
        output_form="tensor",
    )
    assert isinstance(bsq_output, torch.Tensor)
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BSQ])

    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(bip_output, torch.Tensor)
    assert bip_output.shape == (2000, 2000, 3)
    assert np.allclose(input_cube, numpy_cubes[CubeRepresentation.BSQ])


def test_tensor_inputs_tensor_outputs(tensor_cubes):
    """
    Tests to check if tensor inputs under various configurations
    Produce outputs in the desired configurations if specified as numpy
    arrays
    """
    transformer = ImageCubeOperations()
    # BIL to other formats
    input_cube = tensor_cubes[CubeRepresentation.BIL]
    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(bip_output, torch.Tensor)
    assert bip_output.shape == (2000, 2000, 3)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BIL])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BSQ,
        output_form="tensor",
    )
    assert isinstance(bsq_output, torch.Tensor)
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BIL])

    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIL,
        output_form="tensor",
    )
    assert isinstance(bil_output, torch.Tensor)
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BIL])

    # Perform the same tests for BIP to other formats
    input_cube = tensor_cubes[CubeRepresentation.BIP]
    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BIL,
        output_form="tensor",
    )
    assert isinstance(bil_output, torch.Tensor)
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BIP])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BSQ,
        output_form="tensor",
    )
    assert isinstance(bsq_output, torch.Tensor)
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BIP])

    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BIP,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(bip_output, torch.Tensor)
    assert bip_output.shape == (2000, 2000, 3)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BIP])

    # Perform the same tests for BSQ to other formats
    input_cube = tensor_cubes[CubeRepresentation.BSQ]
    bil_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIL,
        output_form="tensor",
    )
    assert isinstance(bil_output, torch.Tensor)
    assert bil_output.shape == (2000, 3, 2000)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BSQ])

    bsq_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BSQ,
        output_form="tensor",
    )
    assert isinstance(bsq_output, torch.Tensor)
    assert bsq_output.shape == (3, 2000, 2000)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BSQ])

    bip_output = transformer.convert_cube(
        cube=input_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(bip_output, torch.Tensor)
    assert bip_output.shape == (2000, 2000, 3)
    assert np.allclose(input_cube, tensor_cubes[CubeRepresentation.BSQ])


def test_invalid_output_form(numpy_cubes):
    """
    Tests to check if an error is raised when an invalid output form is specified
    """
    transformer = ImageCubeOperations()
    input_cube = numpy_cubes[CubeRepresentation.BIL]
    with pytest.raises(ValueError):
        transformer.convert_cube(
            cube=input_cube,
            from_format=CubeRepresentation.BIL,
            to_format=CubeRepresentation.BIP,
            output_form="invalid",
        )


def test_cube_manipulations_on_masked_array(masked_numpy_array):
    """
    Tests to see if cube operations work on masked arrays
    """
    transformer = ImageCubeOperations()

    output_cube = transformer.convert_cube(
        cube=masked_numpy_array,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIP,
        output_form="numpy",
    )

    assert output_cube.shape == (2000, 2000, 3)

    tensor_output = transformer.convert_cube(
        cube=masked_numpy_array,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )

    assert isinstance(tensor_output, torch.Tensor)
    assert tensor_output.shape == (2000, 2000, 3)
