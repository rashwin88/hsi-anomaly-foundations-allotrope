"""
Test bounding box creation for landsat
"""

import pytest
from app.utils.stac.stac_utils.get_landsat_bounding_box import get_landsat_bounding_box


THERMAL = "thermal_2"
INVALID = "hyperspectral_1"


@pytest.mark.large_files
def test_bounding_box_creation(live_source_data):
    """
    Tests whether bounding boxes are properly created in the case of TIF files
    """

    source = live_source_data.get(THERMAL)
    path = source.source_path

    # Now run the valid test
    bounds = get_landsat_bounding_box(path)

    assert len(bounds) == 4
    assert bounds[1] < bounds[3]
    assert bounds[0] < bounds[2]

    # Change the source and ensure there is an error
    path = "invalid"
    with pytest.raises(Exception):
        bounds = get_landsat_bounding_box(path)
