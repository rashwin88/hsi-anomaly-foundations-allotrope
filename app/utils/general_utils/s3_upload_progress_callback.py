"""
Call back showing S3 upload progress
"""

from tqdm import tqdm
import os


class TqdmProgressCallback:
    """
    A helper class to bridge boto3's callback mechanism with tqdm.
    """

    def __init__(self, filepath, filename):
        # Get the exact file size in bytes to set the 100% target for tqdm
        self._size = os.path.getsize(filepath)
        # Initialize the progress bar with byte formatting
        self._pbar = tqdm(
            total=self._size, unit="B", unit_scale=True, desc=f"Uploading {filename}"
        )
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        # boto3 passes the number of bytes uploaded in the current chunk
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)

        # Cleanly close the progress bar when the upload finishes
        if self._seen_so_far >= self._size:
            self._pbar.close()
