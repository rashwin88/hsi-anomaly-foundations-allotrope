"""
Test out building vendable thermal datasets from landsat images
"""

import pytest

from pystac import Item

from app.utils.dataset_builder.landsat_dataset_builder import LandsatDataBuilder
from app.utils.files.tif_helper import TIFHelper
from app.models.images.cube_representation import CubeRepresentation


@pytest.mark.large_files
def test_landsat_data_builder(live_source_data):
    """
    Full test of landsat data building
    """

    builder = LandsatDataBuilder(
        file_source_configuration=live_source_data.get("phase_2_thermal_1")
    )

    # Test all initializations
    assert isinstance(builder.stac_item, Item)
    assert isinstance(builder.file_helper, TIFHelper)
    assert builder.band_information is None
    assert CubeRepresentation.BSQ == builder.default_cube_representation
    assert builder.extract_band_information() is None

    vendable = builder.vend_dataset()

    assert vendable.normalized_thermal_cube.shape[0] == 1
    assert vendable.validity_cube.min() == 0
    assert int(vendable.validity_cube.sum()) > 0


@pytest.mark.large_files
def test_landsat_with_qa_data(live_source_data):
    """
    Full test of landsat data building
    """

    builder = LandsatDataBuilder(
        file_source_configuration=live_source_data.get("thermal_usgs")
    )

    # Test all initializations
    assert isinstance(builder.stac_item, Item)
    assert isinstance(builder.file_helper, TIFHelper)
    assert builder.band_information is None
    assert CubeRepresentation.BSQ == builder.default_cube_representation
    assert builder.extract_band_information() is None

    vendable = builder.vend_dataset(
        provider_qa_pixel_source=live_source_data.get("thermal_usgs_qa")
    )

    assert vendable.normalized_thermal_cube.shape[0] == 1
    assert vendable.validity_cube.min() == 0
    assert int(vendable.validity_cube.sum()) > 0
    assert vendable.provider_cloud_presence.min() == 0
    assert vendable.provider_cloud_presence.max() == 1
    assert vendable.provider_cloud_presence.sum() > 0
    assert vendable.provider_water_presence.min() == 0
    assert vendable.provider_water_presence.max() == 1
    assert vendable.provider_water_presence.sum() > 0
    assert vendable.provider_snow_presence.min() == 0
    assert vendable.provider_snow_presence.max() == 0
    assert vendable.provider_snow_presence.sum() == 0
