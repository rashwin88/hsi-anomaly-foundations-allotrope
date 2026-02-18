"""
Full suite of tests for Prisma Dataset Building
"""

import logging
from typing import Dict
import pytest

from pystac import Item

from app.models.dataset.vendables import VendableHyperspectralDataset
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.images.cube_representation import CubeRepresentation
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.utils.files.he5_helper import HE5Helper

logger = logging.getLogger("PrismaTester")
logger.setLevel(logging.INFO)


@pytest.mark.large_files
def test_initialization_of_dataset_builder(live_source_data):
    """
    Tests the initialization of the dataset builder
    """

    builder = PrismaDatasetBuilder(
        file_source_configuration=live_source_data.get("hyperspectral_1")
    )

    assert isinstance(builder.stac_item, Item)
    assert isinstance(builder.file_helper, HE5Helper)
    assert isinstance(builder.band_information, Dict)
    assert SpectralFamily.SWIR in builder.band_information.keys()
    assert SpectralFamily.VNIR in builder.band_information.keys()
    assert CubeRepresentation.BIL == builder.default_cube_representation

    # From the dataset builder create a full vendable
    vendable = builder.vend_dataset()

    assert isinstance(vendable, VendableHyperspectralDataset)
    assert len(vendable.band_cw_order) == 239
    assert len(vendable.spectral_family_order) == 239
    assert vendable.normalized_hyperspectral_cube.shape[0] == 239
    assert vendable.normalized_hyperspectral_cube.min() == 0.0
    assert vendable.normalized_hyperspectral_cube.max() > 0
    assert vendable.normalized_hyperspectral_cube.max() <= 1
    assert vendable.validity_cube.sum() > 0
    assert vendable.validity_cube.shape[0] == 239
