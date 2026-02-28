"""
Creates paginated forms of S3 Objects
"""

from typing import List
import boto3


def get_all_objects_paginated(
    bucket_name: str,
    prefix_key: str,
    region_name: str = "ap-south-1",
    page_size: int = 100,
) -> List[str]:
    """
    Gets a list of all objects under a given S3 prefix
    """

    # create an S3 client
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
