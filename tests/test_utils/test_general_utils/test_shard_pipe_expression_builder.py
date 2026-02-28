"""
Tests the shard pipe expression builder.
"""

import pytest

from app.utils.general_utils.shard_pipe_expression_builder import (
    shard_pipe_expression_builder,
)


@pytest.mark.network_access
def test_shard_pipe_expression_building():
    """
    Tests to see if the shard pipe expression is built out correctly
    """
    bucket_name = "allotrope-raw-data-india"
    data_key = "patches/final/landsat/128-pixel-training/"
    # Set an expectation for the pipe expression
    expected_pipe_expression = f"pipe: aws s3 cp s3://{bucket_name}/{data_key}final_shard_{{00000..00338}}.tar -"

    # Construct the actual shard pipe expression
    actual_expression = shard_pipe_expression_builder(
        data_key=data_key, bucket_name=bucket_name, region_name="ap-south-1"
    )

    assert expected_pipe_expression == actual_expression
