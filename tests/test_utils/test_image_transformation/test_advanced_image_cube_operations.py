"""
Performs more advanced tests on image cube operations
using real datasets from prisma and landsat
"""

import pytest
import numpy as np
import torch
from app.utils.files.he5_helper import HE5Helper
from app.utils.files.tif_helper import TIFHelper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier
from app.models.images.cube_representation import CubeRepresentation
from app.utils.image_transformation.image_cube_operations import ImageCubeOperations
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

HYPERSPECTRAL = "phase_2_hyperspectral_1"
THERMAL = "phase_2_thermal_1"


@pytest.mark.large_files
def test_masked_operations_hyperpsectral(live_source_data, hyperspectral_band_numbers):
    """
    Tests whether operations on masked cubes works as expected
    """
    # intialize the transformer
    transformer = ImageCubeOperations()

    # initialize the source
    source = live_source_data.get(HYPERSPECTRAL)

    # Initialize the helper
    helper = HE5Helper(
        file_source_config=source,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.PRISMA_HYPERSPECTRAL),
    )

    # Pull a cube of data from the dataset
    raw_cube = helper.extract_specific_bands(
        bands=hyperspectral_band_numbers,
        masking_needed=True,
        spectral_family=SpectralFamily.SWIR,
        mode="specific",
    )

    # perform some operations on the raw cube
    transformed_cube = transformer.convert_cube(
        cube=raw_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIP,
        output_form="numpy",
    )

    # assertions
    assert isinstance(transformed_cube, np.ma.MaskedArray)
    assert transformed_cube.shape[2] == len(hyperspectral_band_numbers)
    assert transformed_cube.shape[0] == raw_cube.shape[0]
    assert transformed_cube.shape[1] == raw_cube.shape[2]

    # Do a test for tensor operations also
    # perform some operations on the raw cube
    transformed_cube = transformer.convert_cube(
        cube=raw_cube,
        from_format=CubeRepresentation.BIL,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(transformed_cube, torch.Tensor)
    assert transformed_cube.shape[2] == len(hyperspectral_band_numbers)
    assert transformed_cube.shape[0] == raw_cube.shape[0]
    assert transformed_cube.shape[1] == raw_cube.shape[2]


@pytest.mark.large_files
def test_masked_operations_thermal(live_source_data):
    """
    Tests whether operations on masked cubes works as expected
    """
    # intialize the transformer
    transformer = ImageCubeOperations()

    # initialize the source
    source = live_source_data.get(THERMAL)

    # Initialize the helper
    helper = TIFHelper(
        file_source_config=source,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    # Pull a cube of data from the dataset
    raw_cube = helper.extract_specific_bands(
        bands=[1],
        masking_needed=True,
        mode="specific",
    )

    # perform some operations on the raw cube
    transformed_cube = transformer.convert_cube(
        cube=raw_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIP,
        output_form="numpy",
    )

    # assertions
    assert isinstance(transformed_cube, np.ma.MaskedArray)
    assert transformed_cube.shape[2] == raw_cube.shape[0]
    assert transformed_cube.shape[0] == raw_cube.shape[1]
    assert transformed_cube.shape[1] == raw_cube.shape[2]

    # Do a test for tensor operations also
    # perform some operations on the raw cube
    transformed_cube = transformer.convert_cube(
        cube=raw_cube,
        from_format=CubeRepresentation.BSQ,
        to_format=CubeRepresentation.BIP,
        output_form="tensor",
    )
    assert isinstance(transformed_cube, torch.Tensor)
    assert transformed_cube.shape[2] == raw_cube.shape[0]
    assert transformed_cube.shape[0] == raw_cube.shape[1]
    assert transformed_cube.shape[1] == raw_cube.shape[2]
