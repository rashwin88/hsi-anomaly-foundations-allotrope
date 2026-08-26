"""Single definition of where Allotrope's S3 data lives.

Before this, the bucket name was a literal in 14 places and the region a bare
string at 6 of 8 client-construction sites, spread over 10 files - with no way
to know whether you had found them all. Consolidated 2026-08-26.

Overridable without touching code:

    ALLOTROPE_S3_BUCKET
    ALLOTROPE_S3_REGION

`app/` must stay free of the database and FastAPI; reading environment
variables is fine here.
"""

import os

BUCKET: str = os.environ.get("ALLOTROPE_S3_BUCKET", "allotrope-raw-data-india")
REGION: str = os.environ.get("ALLOTROPE_S3_REGION", "ap-south-1")


def client():
    """An S3 client for the configured region.

    boto3 is imported here rather than at module scope so that callers who
    only need BUCKET or REGION - the shard-path builders, for instance - do
    not drag the AWS SDK into their import graph.
    """
    import boto3

    return boto3.client("s3", region_name=REGION)
