"""
Ensures that the bounding box creation for PRISMA is working
properly
"""

import pytest
from app.utils.stac.stac_utils.get_prisma_bounding_box import get_prisma_bounding_box


HYPERSPECTRAL = "hyperspectral_1"


def test_prisma_bounding_box_creation(live_source_data):
    """
    Tests to ensure that the Prisma bounding boxes are created properly
    """
    source = live_source_data.get(HYPERSPECTRAL)
    path = source.source_path

    # Now run the actual test
    bounds = get_prisma_bounding_box(path)

    assert len(bounds) == 4
    assert bounds[1] < bounds[3]
    assert bounds[0] < bounds[2]

    # Change the source and ensure there is an error
    path = "invalid"
    with pytest.raises(Exception):
        bounds = get_prisma_bounding_box(path)
