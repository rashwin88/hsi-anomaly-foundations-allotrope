"""
Uploads a file from the local to an S3 location and
then deletes the file from local
"""

from typing import Any
import os

from app.utils.general_utils.s3_upload_progress_callback import TqdmProgressCallback


def s3_upload_and_cleanup(
    file_name: str, s3_prefix: str, bucket_name: str, client: Any
):
    """
    The post hook triggered by ShardWriter the exact moment a shard reaches a specific size and closes.
    """
    # Extract just the file names
    base_name = os.path.basename(file_name)
    s3_key = os.path.join(s3_prefix, base_name)

    progress_callback = TqdmProgressCallback(filepath=file_name, filename=base_name)

    print(f"Uploading {file_name} to s3://{bucket_name}/{s3_key}...")

    try:
        # upload_file automatically uses multipart uploads for large files
        client.upload_file(file_name, bucket_name, s3_key, Callback=progress_callback)

        # Once the upload succeeds without an exception, free up the EBS buffer
        os.remove(file_name)
        print(f"Success. Deleted local buffer: {file_name}\n")

    except Exception as e:
        # If the upload drops, the file remains on your EBS volume so you don't lose the patch data
        print(f"Upload failed for {file_name}: {e}")
        raise e
