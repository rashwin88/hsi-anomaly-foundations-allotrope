"""
Tests paginated s3 object listing
"""

import pytest

from app.utils.general_utils.paginated_s3_listing import get_all_objects_paginated


@pytest.mark.network_access
def test_paginated_listing_of_objects():
    """
    Tests whether objects can be listed in a paginated manner
    """

    bucket_name = "allotrope-raw-data-india"
    prefix = "patches/final/landsat/128-pixel-training/"

    # Now create the list
    object_list = get_all_objects_paginated(
        bucket_name=bucket_name, prefix_key=prefix, page_size=50
    )

    # Assert that the right number of objects are found
    assert len(object_list) == 339

    # Assert that the objects are of the right pattern
    assert "final_shard" in object_list[0]
