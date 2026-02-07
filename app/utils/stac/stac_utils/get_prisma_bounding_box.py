"""
Bounding box for Prisma Files
"""

from typing import List
import logging

import numpy as np
import h5py

logger = logging.getLogger("PrismaBoundingBoxIdentification")
logger.setLevel(logging.INFO)

# Setting up a simple Prisma identifier
# Being very explicit that this is for prisma alone
# Some abstraction possible here, but thats what we have
PRISMA_IDENTIFIER = "PRS_L2D_HCO"


def get_prisma_bounding_box(
    path: str, provider_identification_string: str = PRISMA_IDENTIFIER
) -> List[float]:
    """
    Opens up a Prisma HDFS and extracts the bounds from it.

    STAC needs the bounds of the item to be clearly specified as
    coordinates in the 3D WGS84 standard.

    Uses the internal geolocation arrays for the same.

    We use the provider identification string in case there are some other
    providers we need to consider later on.
    """
    try:
        with h5py.File(path, "r") as file:
            if provider_identification_string not in file["HDFEOS"]["SWATHS"]:
                raise KeyError(f"Cannot find {provider_identification_string} in file")
            # get the geo group
            geo_group = file["HDFEOS"]["SWATHS"][provider_identification_string][
                "Geolocation Fields"
            ]

            # Read in the lat and long arrays
            lats = geo_group["Latitude"][:]
            lons = geo_group["Longitude"][:]

            # We dont need to filter out any masks in the case of L2D data
            # TODO: How to handle this in the case of other data providers

            min_lat, max_lat = np.min(lats), np.max(lats)
            min_lon, max_lon = np.min(lons), np.max(lons)

            return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]

    except Exception as err:
        logger.error("Bounds identification for PRISMA failed with error: %s", str(err))
        raise err
