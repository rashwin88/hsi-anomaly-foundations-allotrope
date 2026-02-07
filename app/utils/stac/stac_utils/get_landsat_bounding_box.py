"""
Gets the bounding box from a landsat file
"""

from typing import List
import logging

import rasterio
from rasterio.warp import transform_bounds


logger = logging.getLogger("LandSatBoundingBoxIdentification")
logger.setLevel(logging.INFO)


def get_landsat_bounding_box(path: str) -> List[float]:
    """
    Opens up a land sat tif file and extracts the bounds from it in the WGS84 form
    Note that STAC needs the bounds of the Item to be specified as coordinates on the 3D WGS84 ellipsoid.
    This function performs that conversion for landsat files specified as TIF.
    Applicable all Landsat and similar providers.
    """
    try:
        # Open up the file
        with rasterio.open(path) as src:
            # Transform the bounds from the File CRS to the Lat/Lon in WGS84
            # This is a simple operation in the
            min_lon, min_lat, max_lon, max_lat = transform_bounds(
                src.crs, "EPSG:4326", *src.bounds
            )
            return [min_lon, min_lat, max_lon, max_lat]
    except Exception as err:
        logging.error(
            "Extraction of bounding boxes from the provided path: %s errored out", path
        )
        raise err
