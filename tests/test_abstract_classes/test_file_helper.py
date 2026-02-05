"""
Tests the abstractness of the file_helper
class and whether it can be instantiated on its own
"""

import pytest
from app.abstract_classes.file_helper import FileHelper


def test_class_cannot_be_instantiated(mock_source):
    """
    Tests that the base class cannot be instantiated.
    """
    with pytest.raises(TypeError):
        FileHelper(mock_source)  # pylint: disable=abstract-class-instantiated
