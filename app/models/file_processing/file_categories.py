from enum import Enum


class FileCategory(Enum):
    """
    Defines the category of the file.
    We start with 2 categories he5 (hyperspectral) and tif (raster).
    """

    HDFS = "he5"
    TIF = "tif"
