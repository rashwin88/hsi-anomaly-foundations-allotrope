from enum import Enum


class FileCategory(Enum):
    """
    Defines the category of the file.
    We start with 2 categories he5 (hyperspectral) and tif (raster).
    """

    HDFS = "he5"
    TIF = "tif"
    ENMAP_FOLDER = "enmap_folder"
    ENVI_FOLDER = "envi_folder"      # a scene folder holding ENVI .hdr / .bin pairs (AVIRIS-NG)
    ENVI_HEADER = "envi_header"      # one ENVI .hdr whose data file sits beside it (Tanager ENVI)
    HOTSAT_FOLDER = "hotsat_folder"  # a HotSat-1 bundle: metadata.json + *_ortho.tiff / *_stacked.tiff
