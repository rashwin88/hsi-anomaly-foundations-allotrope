"""
Builds a STAC Item for a scene - its identity, footprint and date.

STAC (SpatioTemporal Asset Catalog) is the standard vocabulary for describing
geospatial data. Using it means scene metadata is expressed the same way whether
it came from PRISMA, EnMAP or Landsat, and can be consumed by any STAC-aware
tool without bespoke glue.

Combines two sources: FileNameParser for what the filename encodes (platform,
date, processing level) and a per-sensor bounding-box function for the actual
geographic footprint. The bounding box requires opening the file, so this is the
first point in ingestion that reads real data.

Each sensor needs its own bounding-box routine because the georeferencing lives
somewhere different in each: rasterio bounds for GeoTIFF, per-pixel lat/lon
arrays for PRISMA swath data, the XML sidecar for EnMAP.

The item produced here is stored on the vendable and travels with the scene,
which is what later lets an export state where on Earth an anomaly was found.
"""

from typing import Dict, List, Any, Tuple
import logging
import os

from pystac import Item, Asset, MediaType

from app.utils.stac.stac_utils.file_name_parsers import FileNameParser
from app.utils.stac.stac_utils.get_landsat_bounding_box import get_landsat_bounding_box
from app.utils.stac.stac_utils.get_prisma_bounding_box import get_prisma_bounding_box
from app.utils.stac.stac_utils.get_enmap_bounding_box import get_enmap_bounding_box
from app.utils.stac.stac_configurations.asset_roles import AssetRole

logger = logging.getLogger("STACItemCreator")
logger.setLevel(logging.INFO)


class StacCreator:
    """
    Creates STAC items and assets for the provided files
    """

    def __init__(self, file_path: str):
        """
        Class constructor
        """
        self.file_path: str = file_path
        # First parse and create the filename
        # basename, not split("/"): on Windows an EnMAP *folder* path is
        # backslash-separated, so splitting on "/" returned the whole path and
        # the EnMAP-folder check below rejected it. Identical on Linux.
        self.file_name: str = os.path.basename(file_path.rstrip("/\\"))
        logger.info("File Name: %s", self.file_name)
        self.file_id = self.file_name.split(".")[0]

        # Determine the media type of the file also
        if self.file_name.endswith(("TIF", "TIFF")):
            self.media_type = MediaType.COG
        elif self.file_name.endswith(("he5")):
            self.media_type = MediaType.HDF5
        elif not self.file_name.startswith("ENMAP"):
            raise TypeError(
                f"Unsupported file type: {self.file_name!r}. "
                "Expected .TIF, .TIFF, .he5, or an EnMAP folder."
            )
        else:
            # For folder-based sources (e.g., EnMAP), defer media type until after parsing
            self.media_type = None

        # Create the metadata helper
        self.helper: FileNameParser = FileNameParser()
        self.metadata: Dict[str, str] = self.helper.parse(file_name=self.file_name)

        # Collect bounding boxes
        # And also set the asset roles
        if self.metadata.get("platform") == "Prisma":
            self.bounding_box: List[float] = get_prisma_bounding_box(self.file_path)
            self.asset_role = AssetRole.HYPERSPECTRAL.value
        elif self.metadata.get("platform") in ("landsat-8", "landsat-9"):
            self.bounding_box: List[float] = get_landsat_bounding_box(self.file_path)
            self.asset_role = AssetRole.THERMAL.value
        elif self.metadata.get("platform") == "EnMAP":
            self.bounding_box: List[float] = get_enmap_bounding_box(self.file_path)
            self.asset_role = AssetRole.HYPERSPECTRAL.value
            if self.media_type is None:
                self.media_type = MediaType.COG
        self.geom = self._build_geojson_geometry()

    def _build_geojson_geometry(self) -> Dict[str, Any]:
        """
        Builds the GeoJSON geometry given a bounding box
        """
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [self.bounding_box[0], self.bounding_box[1]],
                    [self.bounding_box[2], self.bounding_box[1]],
                    [self.bounding_box[2], self.bounding_box[3]],
                    [self.bounding_box[0], self.bounding_box[3]],
                    [self.bounding_box[0], self.bounding_box[1]],
                ]
            ],
        }

    def build_stack(self) -> Item:
        """
        Builds the actual pystac item
        """
        # We create two basic dicts - item properties which are properties to be stored
        # At the item level and extra fields which are to be stored at the asset level
        # We then loop through the metadata and construct both of these
        # Making a choice as it is a lot simpler to do this in a control flow rather than doing something more
        # Sophisticated at the moment

        item_props = {
            key: value
            for key, value in self.metadata.items()
            if key in ["platform", "processing:level", "product_type"]
        }

        # Create the item
        item = Item(
            id=self.file_id,
            geometry=self.geom,
            bbox=self.bounding_box,
            datetime=self.metadata.get("datetime"),
            properties=item_props,
        )

        # Add the asset to the item
        asset = Asset(
            href=self.file_path,
            media_type=self.media_type,
            roles=[self.asset_role + "_input_data"],
        )
        # We keep the asset ket as "primary_datacube" we can add more to this later on
        item.add_asset(key="primary_input_datacube", asset=asset)
        return item
