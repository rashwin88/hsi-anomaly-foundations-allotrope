"""
Tests the he5 helper implementation of the file helper abstraction
"""

import pytest
import numpy as np
from app.utils.files.he5_helper import HE5Helper
from app.models.products.products import Product
from app.models.file_processing.file_metadata_models import (
    He5Metadata,
    He5ComponentMetadata,
)

from app.models.file_processing.file_categories import FileCategory
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.hyperspectral_concepts.file_components import (
    HyperspectralFileComponents,
)

PHASE1 = "hyperspectral_3"
PHASE2 = "phase_2_hyperspectral_1"


@pytest.mark.large_files
def test_class_initialization(live_source_data):
    """
    Uses an actual file to see if the HE5Helper initializes correctly
    """

    # Using hyperspectral from phase 2 for this
    source = live_source_data.get(PHASE2)

    # Initialize the HE5Helper1
    helper = HE5Helper(
        file_source_config=source, product=Product.PRISMA
    )  # pylint: disable=abstract-class-instantiated

    # Run tests to make sure that the objects that need to be populated are present
    assert helper.file_source_config.file_category == FileCategory.HDFS
    assert isinstance(helper.file_metadata, He5Metadata)
    assert len(helper.file_metadata.components) > 0
    assert isinstance(helper.file_metadata.root_metadata, He5ComponentMetadata)

    # Check out the root metdata
    assert len(helper.file_metadata.root_metadata.file_attributes.keys()) > 0

    # Check out the component metadata
    assert (
        len(helper.file_metadata.component_metadata.keys()) > 0
    )  # Pylint: disable=no-member

    # make sure templates are loaded
    assert len(helper.template.keys()) > 0


@pytest.mark.large_files
def test_access_dataset_path_invalidity(live_source_data):
    """
    Ensures that sending invalid paths to access datasets returns the right errors
    """
    # Using hyperspectral from phase 2 for this
    source = live_source_data.get(PHASE2)

    # Initialize the HE5Helper1
    helper = HE5Helper(
        file_source_config=source, product=Product.PRISMA
    )  # pylint: disable=abstract-class-instantiated

    with pytest.raises(TypeError):
        helper.access_dataset("xxxxxx")

    with pytest.raises(KeyError):
        helper.access_dataset("HDFEOS/SWATHS/PRS_L2D_PCO")


@pytest.mark.large_files
def test_full_band_extraction(live_source_data):
    """
    Tests band extraction in different scenarios from the proviced datasets
    """

    # Using hyperspectral from phase 2 for this
    source = live_source_data.get(PHASE2)

    # Initialize the HE5Helper1
    helper = HE5Helper(file_source_config=source, product=Product.PRISMA)

    # Collect all the bands from the SWIR Cube
    swir_cube = helper.extract_specific_bands(
        bands=["*"],
        masking_needed=False,
        spectral_family=SpectralFamily.SWIR,
        mode="all",
    )

    # collect the shape attreibute from the actual component properties
    swir_cube_path = helper.template.get(HyperspectralFileComponents.SWIR_CUBE_DATA)
    # Check the shape of the metadata
    swir_cube_metadata = helper.file_metadata.component_metadata.get(
        swir_cube_path.file_name
    )

    assert isinstance(swir_cube, np.ndarray)
    assert swir_cube.shape == swir_cube_metadata.shape

    # Collect all the bands from the VNIR Cube
    vnir_cube = helper.extract_specific_bands(
        bands=["*"],
        masking_needed=False,
        spectral_family=SpectralFamily.VNIR,
        mode="all",
    )

    # collect the shape attreibute from the actual component properties
    vnir_cube_path = helper.template.get(HyperspectralFileComponents.VNIR_CUBE_DATA)
    # Check the shape of the metadata
    vnir_cube_metadata = helper.file_metadata.component_metadata.get(
        vnir_cube_path.file_name
    )

    assert isinstance(vnir_cube, np.ndarray)
    assert vnir_cube.shape == vnir_cube_metadata.shape

    with pytest.raises(KeyError):
        panchromatic_cube = helper.extract_specific_bands(
            bands=["*"],
            masking_needed=False,
            spectral_family=SpectralFamily.PANCHROMATIC,
            mode="all",
        )
        del panchromatic_cube


@pytest.mark.large_files
def test_specific_band_pulling(live_source_data):
    """
    Tests whether specific bands can be pulled from the source data
    """

    # Using hyperspectral from phase 2 for this
    source = live_source_data.get(PHASE2)

    # Initialize the HE5Helper1
    helper = HE5Helper(file_source_config=source, product=Product.PRISMA)

    # collect the shape attreibute from the actual component properties
    swir_cube_path = helper.template.get(HyperspectralFileComponents.SWIR_CUBE_DATA)
    # Check the shape of the metadata
    swir_cube_metadata = helper.file_metadata.component_metadata.get(
        swir_cube_path.file_name
    )

    # Collect all the bands from the SWIR Cube
    swir_cube = helper.extract_specific_bands(
        bands=[4, 5, 10],
        masking_needed=False,
        spectral_family=SpectralFamily.SWIR,
        mode="specific",
    )

    assert swir_cube.shape[0] == swir_cube_metadata.shape[0]
    assert swir_cube.shape[2] == swir_cube_metadata.shape[2]
    assert swir_cube.shape[1] == 3


@pytest.mark.large_files
def test_masking_application(live_source_data):
    """
    Check whether the masking operation is being performed properly
    """

    # Using hyperspectral from phase 2 for this
    source = live_source_data.get(PHASE2)

    # Initialize the HE5Helper1
    helper = HE5Helper(file_source_config=source, product=Product.PRISMA)

    # Collect all the bands from the SWIR Cube
    swir_cube = helper.extract_specific_bands(
        bands=[4, 5, 10],
        masking_needed=True,
        spectral_family=SpectralFamily.SWIR,
        mode="specific",
    )

    assert isinstance(swir_cube, np.ma.MaskedArray)
    assert swir_cube.mask.sum() > 0

    # Collect all the bands from the SWIR Cube
    swir_cube = helper.extract_specific_bands(
        bands=[4, 5, 10],
        masking_needed=False,
        spectral_family=SpectralFamily.SWIR,
        mode="specific",
    )

    assert isinstance(swir_cube, np.ndarray)
