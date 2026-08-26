"""
Creates paginated forms of S3 Objects
"""

from typing import List, Optional

from app.utils.general_utils import s3_config


def get_all_objects_paginated(
    bucket_name: str,
    prefix_key: str,
    region_name: Optional[str] = None,
    page_size: int = 100,
) -> List[str]:
    """
    Gets a list of all objects under a given S3 prefix

    `region_name=None` uses the configured region (`s3_config.REGION`).
    """

    # create an S3 client
    if region_name is None:
        client = s3_config.client()
    else:
        import boto3

        client = boto3.client("s3", region_name=region_name)
    # Get the paginator
    paginator = client.get_paginator("list_objects_v2")

    # Build the page iterator
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix_key,
        PaginationConfig={"PageSize": page_size},
    )

    # Create an empty output object
    output = []

    for page in page_iterator:
        for key in page.get("Contents"):
            output.append(key.get("Key"))

    # Return the output
    return output
