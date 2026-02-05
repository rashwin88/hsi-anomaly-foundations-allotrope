"""
Tests all file source configs
"""

import pytest
from app.models.file_processing.file_categories import FileCategory
from app.models.file_processing.sources import FileSourceConfig
from pydantic import ValidationError


def test_file_category_properly_recognized(mock_tif_source, mock_source):
    """
    Makes sure that the file cateory is properly validated
    """

    # he5 source
    file_source_config = mock_source
    assert file_source_config.file_category == FileCategory.HDFS

    # tif source
    file_source_config = mock_tif_source
    assert file_source_config.file_category == FileCategory.TIF

    # assert that validation error correctly raised
    with pytest.raises(ValidationError):
        file_source_config = FileSourceConfig(source_path="mock.pdf")
