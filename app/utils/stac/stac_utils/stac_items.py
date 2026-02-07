"""
Class that unifies all the parsers and creates a
proper stac item and corresponding assets
"""

from typing import Dict, List
import logging

import pystac

from app.utils.stac.stac_utils.file_name_parsers import FileNameParser
from app.utils.stac.stac_utils.get_landsat_bounding_box import get_landsat_bounding_box
from app.utils.stac.stac_utils.get_prisma_bounding_box import get_prisma_bounding_box

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
        self.file_name: str = file_path.split("/")[-1]
        logger.info("File Name: %s", self.file_name)

        # Create the metadata helper
        self.helper: FileNameParser = FileNameParser()
        self.metadata: Dict[str, str] = self.helper.parse(file_name=self.file_name)

        # Collect bounding boxes
        if self.metadata.get("platform") == "Prisma":
            self.bounding_box: List[float] = get_prisma_bounding_box(self.file_path)
        elif self.metadata.get("platform") == "landsat-9":
            self.bounding_box: List[float] = get_landsat_bounding_box(self.file_path)
