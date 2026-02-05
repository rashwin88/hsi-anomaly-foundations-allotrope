"""
Tests for ensuring that reference definitions are properly defined
"""

import pytest
from app.models.hyperspectral_concepts.references import (
    ReferenceType,
    ReferenceDefinition,
)


def test_correct_errors_on_initiation():
    """
    Tests whether the correct errors are raised on initiation.
    """

    with pytest.raises(ValueError):
        ReferenceDefinition(
            description="Test Reference", reference_type=ReferenceType.FILE_REFERENCE
        )

    with pytest.raises(ValueError):
        ReferenceDefinition(
            description="Test Reference",
            reference_type=ReferenceType.ROOT_METADATA_FIELD,
        )

    with pytest.raises(ValueError):
        ReferenceDefinition(
            description="Test Reference",
            reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        )
