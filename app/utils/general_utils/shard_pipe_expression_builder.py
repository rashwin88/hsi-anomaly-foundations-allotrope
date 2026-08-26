"""
Builds a shard pipe expression function for web datasets
"""

import logging

from app.utils.general_utils import s3_config
from app.utils.general_utils.paginated_s3_listing import get_all_objects_paginated

# Kept as a name because callers import it; the value now comes from one place.
DEFAULT_BUCKET_NAME: str = s3_config.BUCKET
logger = logging.getLogger("ShardPipeExpressions")
logger.setLevel(logging.DEBUG)


def shard_pipe_expression_builder(
    data_key: str,
    bucket_name: str = DEFAULT_BUCKET_NAME,
    region_name: str = s3_config.REGION,
) -> str:
    """
    Given an S3 location constructs a shard pipe expression for the same
    """

    # Pull out all the objects from the bucket and datakey
    object_list = get_all_objects_paginated(
        bucket_name=bucket_name,
        prefix_key=data_key,
        region_name=region_name,
        page_size=200,
    )

    # Now that the final objects are present, we can proceed.
    # The general pattern is <shard prefix>_<shard number>

    shard_numbers = [
        key.split("/")[-1].split(".")[0].split("_")[-1] for key in object_list
    ]

    # Identify the key for the intermediate shards
    sample_key = object_list[0]
    shard_elements = sample_key.split("/")[-1].split(".")[0].split("_")
    shard_identifier = "_".join(shard_elements[: len(shard_elements) - 1])

    # Put everything together
    final_range = (
        f"{shard_identifier}_{{{min(shard_numbers)}..{max(shard_numbers)}}}.tar"
    )

    # Construct the pipe expression
    pipe_expression = f"pipe: aws s3 cp s3://{bucket_name}/{data_key}{final_range} -"

    return pipe_expression
