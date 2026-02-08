"""
Test for stac item and asset creation
"""

import pytest
from pystac import MediaType
from pystac import Item
from app.utils.stac.stac_utils.stac_items import StacCreator
from app.models.file_processing.sources import FileSourceConfig


def test_class_initialization(live_source_data):
    """
    Tests whether class is initialized properly
    """

    source_config: FileSourceConfig = live_source_data.get("hyperspectral_1")

    # initialize the stac creation
    stac_creator: StacCreator = StacCreator(file_path=source_config.source_path)

    assert stac_creator.metadata.get("platform") == "Prisma"
    assert len(stac_creator.bounding_box) == 4
    assert stac_creator.media_type == MediaType.HDF5

    ## doing the same for landsat
    source_config: FileSourceConfig = live_source_data.get("thermal_1")

    # initialize the stac creation
    stac_creator: StacCreator = StacCreator(file_path=source_config.source_path)

    assert stac_creator.metadata.get("platform") == "landsat-9"
    assert len(stac_creator.bounding_box) == 4
    assert stac_creator.media_type == MediaType.COG

    # Test some error on init
    with pytest.raises(TypeError):
        stac_creator: StacCreator = StacCreator(
            file_path=source_config.source_path + ".html"
        )


def test_stac_item_creation(live_source_data):
    """
    Tests the actual item creation
    """
    source_config: FileSourceConfig = live_source_data.get("hyperspectral_1")

    # initialize the stac creation
    stac_creator: StacCreator = StacCreator(file_path=source_config.source_path)
    stac_item: Item = stac_creator.build_stack()

    # Now we can verify things
    assert isinstance(stac_item, Item)
    assert len(stac_item.assets) > 0
    assert stac_item.geometry.get("type") == "Polygon"
    assert stac_item.assets.get("primary_input_datacube").media_type == MediaType.HDF5
