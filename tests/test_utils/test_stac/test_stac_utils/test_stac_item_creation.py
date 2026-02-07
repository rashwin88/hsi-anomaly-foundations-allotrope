"""
Test for stac item and asset creation
"""

from _pytest._code import source
from h5py import File
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

    ## doing the same for landsat
    source_config: FileSourceConfig = live_source_data.get("thermal_1")

    # initialize the stac creation
    stac_creator: StacCreator = StacCreator(file_path=source_config.source_path)

    assert stac_creator.metadata.get("platform") == "landsat-9"
    assert len(stac_creator.bounding_box) == 4
