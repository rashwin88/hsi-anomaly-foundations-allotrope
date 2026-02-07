"""
Define all applicable fields for a dataset that we want to capture
"""

from enum import Enum


class ApplicableFields(str, Enum):
    """
    All applicable fields for a dataset
    """

    PLATFORM = "platform"
    PATH_ROW = "part_row"
    ACQUISITION_DATE_STR = "acq_date_str"
    PROCESSING_LEVEL = "processing_level"
    BAND = "band"
