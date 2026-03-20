"""
Tests for EnMAP STAC integration: filename parsing, bounding box, and STAC item creation.
"""

import datetime
import pytest

from app.utils.stac.stac_utils.file_name_parsers import FileNameParser
from app.utils.stac.stac_utils.get_enmap_bounding_box import get_enmap_bounding_box
from app.utils.stac.stac_utils.stac_items import StacCreator


SCENE_FOLDER = "tests/test_payloads/ENMAP01-____L2A-DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z"
SCENE_NAME = "ENMAP01-____L2A-DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z"


def test_filename_parser_enmap():
    """Verify parsed platform, datetime, processing level from EnMAP folder name."""
    parser = FileNameParser()
    metadata = parser.parse(SCENE_NAME)

    assert metadata["platform"] == "EnMAP"
    assert metadata["processing:level"] == "L2A"
    assert metadata["product_type"] == "STANDARD_ALL"
    assert isinstance(metadata["datetime"], datetime.datetime)
    assert metadata["datetime"].year == 2024
    assert metadata["datetime"].month == 1
    assert metadata["datetime"].day == 28


def test_bounding_box():
    """Verify bounding box values from XML are reasonable."""
    bbox = get_enmap_bounding_box(SCENE_FOLDER)
    assert len(bbox) == 4
    min_lon, min_lat, max_lon, max_lat = bbox
    assert min_lon < max_lon
    assert min_lat < max_lat
    # Should be in Indian Ocean / Gujarat region based on test data
    assert 69 < min_lon < 71
    assert 22 < min_lat < 24


def test_stac_item_creation():
    """Verify full STAC item builds correctly from EnMAP folder."""
    creator = StacCreator(file_path=SCENE_FOLDER)
    item = creator.build_stack()

    assert item.id == SCENE_NAME
    assert item.properties["platform"] == "EnMAP"
    assert item.properties["processing:level"] == "L2A"
    assert item.bbox is not None
    assert len(item.bbox) == 4
    assert item.geometry is not None
    assert "primary_input_datacube" in item.assets
