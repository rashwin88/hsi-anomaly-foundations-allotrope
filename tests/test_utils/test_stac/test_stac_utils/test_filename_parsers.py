"""
Tests for proper filename parsing
"""

from datetime import datetime

import pytest
from app.utils.stac.stac_utils.file_name_parsers import FileNameParser


@pytest.fixture
def helper() -> FileNameParser:
    """
    Vends a filename parser
    """
    return FileNameParser()


def test_prisma_parsing(helper):
    """
    Tests whether prisma filenames are parsed correctly
    """
    file_name = "PRS_L2D_STD_20210516050459_20210516050503_0001.he5"
    parsed_data = helper.parse(file_name)

    assert parsed_data.get("platform") == "Prisma"
    assert parsed_data.get("processing_level") == "L2D"
    assert parsed_data.get("product_type") == "STD"
    assert parsed_data.get("datetime") == datetime.strptime(
        "20210516050459", "%Y%m%d%H%M%S"
    )


def test_lc09_parsing(helper):
    """
    Tests whether prisma filenames are parsed correctly
    """
    file_name = "LC09_L2SP_141045_20250604_20250605_02_T1_ST_B10.TIF"
    parsed_data = helper.parse(file_name)

    assert parsed_data.get("platform") == "landsat-9"
    assert parsed_data.get("processing_level") == "L2SP"
    assert parsed_data.get("product_type") == "ST"
    assert parsed_data.get("datetime") == datetime.strptime("20250604", "%Y%m%d")
