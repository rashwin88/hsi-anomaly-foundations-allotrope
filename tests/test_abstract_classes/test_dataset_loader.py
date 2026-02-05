"""
Makes sure that the DatasetLoader cannot be instantiated on its own
"""

import pytest
from app.abstract_classes.dataset_builder import DatasetBuilder


def test_class_cannot_be_instantiated(mock_source):
    """
    Tests that the base class cannot be instantiated.
    """
    with pytest.raises(TypeError):
        DatasetBuilder(mock_source)  # pylint: disable=abstract-class-instantiated
