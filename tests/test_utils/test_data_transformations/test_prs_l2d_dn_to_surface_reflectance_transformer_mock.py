"""
Tests the L2D DN to surface reflectance transformer on mock data
"""

from typing import Tuple, Dict
import pytest
import numpy as np
from app.models.file_processing.file_metadata_models import (
    He5Metadata,
    He5ComponentMetadata,
)
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.utils.image_transformation.image_cube_operations import (
    CubeRepresentation,
)
from app.utils.data_transformations.prs_l2d_dn_to_surface_reflectance_transformer import (
    PrsL2dDnToSurfaceReflectanceTransformer,
)


@pytest.fixture
def file_attributes() -> Dict:
    """
    Sample file attributes for testing
    """
    return {
        "L2ScaleSwirMax": 0.9,
        "L2ScaleSwirMin": 0.5,
        "L2ScaleVnirMax": 0.8,
        "L2ScaleVnirMin": 0.4,
    }


@pytest.fixture
def file_metadata_mock(file_attributes) -> He5Metadata:
    """
    Mocks out a simple metadata object for testing
    """

    root_mock = He5ComponentMetadata(file_attributes=file_attributes)

    final_mock = He5Metadata(root_metadata=root_mock)

    return final_mock


@pytest.fixture
def sample_input_data() -> Tuple:
    """
    Mocks Sample Input Data
    """

    # Makes up mock data in the BIP form
    # Will just be 1 0 1 0 and so on
    full_template = np.array(
        [
            [5000, 0, 5000, 0],
            [0, 25000, 0, 25000],
            [5000, 0, 5000, 0],
            [0, 25000, 0, 25000],
            [5000, 0, 5000, 0],
        ],
        dtype=np.uint16,
    )

    # Create an empty input array and the band arrangements
    input_array = np.empty(shape=(5, 12, 4), dtype=np.uint16)
    band_arrangements = [SpectralFamily.SWIR] * 6 + [SpectralFamily.VNIR] * 6
    for i in range(12):
        if i <= 9:
            input_array[:, i, :] = full_template.copy()
        else:
            input_array[:, i, :] = np.zeros_like(full_template)
    return band_arrangements, input_array


def test_conversions_on_mock_data(
    file_metadata_mock, sample_input_data, file_attributes
):
    """
    Tests the actual conversions on mock data
    """

    # First test on standard numpy arrays
    input_cube = sample_input_data[1]
    input_band_mapping = sample_input_data[0]

    # Initialize the converter
    converter = PrsL2dDnToSurfaceReflectanceTransformer()

    # Perform a single conversion
    print(file_metadata_mock)
    converted_output = converter.transform(
        input_data=input_cube,
        cube_representation=CubeRepresentation.BIL,
        band_mapping=input_band_mapping,
        file_metadata=file_metadata_mock,
    )

    # Now check the converted output
    assert isinstance(converted_output, np.ndarray)
    assert converted_output.shape == input_cube.shape
    print(converted_output[:, 0, :])
    assert round(converted_output[0, 0, 0], 4) == round(
        file_attributes.get("L2ScaleSwirMin")
        + 5000
        * (
            file_attributes.get("L2ScaleSwirMax")
            - file_attributes.get("L2ScaleSwirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output[0, 0, 1], 4) == round(
        file_attributes.get("L2ScaleSwirMin")
        + 0
        * (
            file_attributes.get("L2ScaleSwirMax")
            - file_attributes.get("L2ScaleSwirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output[1, 0, 1], 4) == round(
        file_attributes.get("L2ScaleSwirMin")
        + 25000
        * (
            file_attributes.get("L2ScaleSwirMax")
            - file_attributes.get("L2ScaleSwirMin")
        )
        / 65535,
        4,
    )
    # Tests on VNIR conversions
    assert round(converted_output[0, 6, 0], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 5000
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output[0, 6, 1], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 0
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output[1, 6, 1], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 25000
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )

    # Tests on null bands
    assert round(converted_output[0, 10, 0], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 0
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )


def test_conversions_on_mock_masked_data(
    file_metadata_mock, sample_input_data, file_attributes
):
    """
    Tests the actual conversions on mock data
    """

    # First test on standard numpy arrays
    input_cube = sample_input_data[1]
    input_band_mapping = sample_input_data[0]
    input_cube = np.ma.masked_where(input_cube == 0, input_cube)

    # Initialize the converter
    converter = PrsL2dDnToSurfaceReflectanceTransformer()

    # Perform a single conversion
    print(file_metadata_mock)
    converted_output = converter.transform(
        input_data=input_cube,
        cube_representation=CubeRepresentation.BIL,
        band_mapping=input_band_mapping,
        file_metadata=file_metadata_mock,
    )

    # Now check the converted output
    assert isinstance(converted_output, np.ma.MaskedArray)
    assert converted_output.data.shape == input_cube.shape
    assert converted_output.mask.shape == input_cube.shape
    print(converted_output[:, 0, :])
    assert round(converted_output[0, 0, 0], 4) == round(
        file_attributes.get("L2ScaleSwirMin")
        + 5000
        * (
            file_attributes.get("L2ScaleSwirMax")
            - file_attributes.get("L2ScaleSwirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output.data[0, 0, 1], 4) == round(
        file_attributes.get("L2ScaleSwirMin")
        + 0
        * (
            file_attributes.get("L2ScaleSwirMax")
            - file_attributes.get("L2ScaleSwirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output[1, 0, 1], 4) == round(
        file_attributes.get("L2ScaleSwirMin")
        + 25000
        * (
            file_attributes.get("L2ScaleSwirMax")
            - file_attributes.get("L2ScaleSwirMin")
        )
        / 65535,
        4,
    )
    # Tests on VNIR conversions
    assert round(converted_output[0, 6, 0], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 5000
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output.data[0, 6, 1], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 0
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )
    assert round(converted_output[1, 6, 1], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 25000
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )

    # Tests on null bands
    assert round(converted_output.data[0, 10, 0], 4) == round(
        file_attributes.get("L2ScaleVnirMin")
        + 0
        * (
            file_attributes.get("L2ScaleVnirMax")
            - file_attributes.get("L2ScaleVnirMin")
        )
        / 65535,
        4,
    )
