"""
Full suite of tests for EnMAP L2A Dataset Building.
"""

import logging
from typing import Dict
import pytest
import numpy as np

from pystac import Item

from app.models.dataset.vendables import VendableEnmapHyperspectralDataset
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.images.cube_representation import CubeRepresentation
from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
from app.utils.files.enmap_helper import EnmapHelper


logger = logging.getLogger("EnmapTester")
logger.setLevel(logging.INFO)

SCENE_FOLDER = "tests/test_payloads/ENMAP01-____L2A-DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z"


@pytest.fixture
def enmap_builder() -> EnmapDatasetBuilder:
    config = FileSourceConfig(source_path=SCENE_FOLDER)
    return EnmapDatasetBuilder(file_source_configuration=config)


@pytest.mark.large_files
def test_initialization_of_dataset_builder(enmap_builder):
    """Tests the initialization of the EnMAP dataset builder."""
    assert isinstance(enmap_builder.stac_item, Item)
    assert isinstance(enmap_builder.file_helper, EnmapHelper)
    assert isinstance(enmap_builder.band_information, Dict)
    assert SpectralFamily.VNIR in enmap_builder.band_information.keys()
    assert SpectralFamily.SWIR in enmap_builder.band_information.keys()
    assert CubeRepresentation.BSQ == enmap_builder.default_cube_representation


@pytest.mark.large_files
def test_band_information(enmap_builder):
    """Verify VNIR and SWIR band counts match detector boundary."""
    vnir_bands = enmap_builder.band_information[SpectralFamily.VNIR]
    swir_bands = enmap_builder.band_information[SpectralFamily.SWIR]
    total = len(vnir_bands.bands_by_index) + len(swir_bands.bands_by_index)
    assert total == 224


@pytest.mark.large_files
def test_vend_dataset(enmap_builder):
    """Full end-to-end test of vend_dataset."""
    vendable = enmap_builder.vend_dataset()

    assert isinstance(vendable, VendableEnmapHyperspectralDataset)

    # Cube shape and dtype
    assert vendable.normalized_hyperspectral_cube.shape[0] == 224
    assert vendable.normalized_hyperspectral_cube.dtype == np.float32

    # Reflectance range: L2A surface reflectance can have small negative values
    # from atmospheric correction artifacts, but should not be wildly negative
    assert vendable.normalized_hyperspectral_cube.min() >= -0.5

    # Validity cube
    assert vendable.validity_cube.shape[0] == 224
    assert vendable.validity_cube.sum() > 0

    # Per-band metadata
    assert len(vendable.spectral_family_order) == 224
    assert len(vendable.band_cw_order) == 224
    assert len(vendable.band_fwhm_order) == 224
    assert len(vendable.band_validity_by_position) == 224

    # Spectral families should include both VNIR and SWIR
    assert SpectralFamily.VNIR in vendable.spectral_family_order
    assert SpectralFamily.SWIR in vendable.spectral_family_order

    # First band CW should be ~418nm, last ~2445nm
    assert 415 < vendable.band_cw_order[0] < 425
    assert 2440 < vendable.band_cw_order[-1] < 2450

    # Quality masks should be present and 2D
    assert vendable.cloud_mask is not None
    assert vendable.cloud_mask.ndim == 2
    assert vendable.cirrus_mask is not None
    assert vendable.haze_mask is not None
    assert vendable.cloud_shadow_mask is not None
    assert vendable.snow_mask is not None

    # Channel indices
    assert len(vendable.vnir_channel_indices) == 91
    assert len(vendable.swir_channel_indices) == 133


@pytest.mark.large_files
def test_spectral_family_assignment(enmap_builder):
    """Verify VNIR/SWIR assignment matches detector boundary."""
    vendable = enmap_builder.vend_dataset()
    metadata = enmap_builder.file_helper.file_metadata
    vnir_channels = set(metadata.detector_boundary.vnir_expected_channels)

    for i, (family, band_char) in enumerate(
        zip(vendable.spectral_family_order, metadata.band_characterisation)
    ):
        if band_char.band_number in vnir_channels:
            assert family == SpectralFamily.VNIR, (
                f"Band {band_char.band_number} should be VNIR"
            )
        else:
            assert family == SpectralFamily.SWIR, (
                f"Band {band_char.band_number} should be SWIR"
            )
