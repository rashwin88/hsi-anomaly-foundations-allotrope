"""
File name parsers for geo-spatial files
"""

from typing import Dict, Callable
import datetime

from app.models.dataset.applicable_metadata import ApplicableFields
from app.utils.stac.stac_configurations.platform_mappings import PLATFORM_MAPPINGS
from app.utils.stac.stac_configurations.processing_levels import ProcessingLevels


class FileNameParser:
    """
    Extracts different metadata from the file name
    """

    def __init__(self):
        """
        Class constructor
        """
        pass

    def parse(self, file_name: str) -> Dict[str, str]:
        """
        Parses to get information from a file name
        """
        method = self._router(file_name)
        return method(file_name)

    def _router(self, file_name: str) -> Callable:
        """
        Routes and returns a callable that will do the actual parsing
        """
        if file_name.startswith("PRS"):
            return FileNameParser.prisma
        elif file_name.startswith("LC09"):
            return FileNameParser.landsat_09
        elif file_name.startswith("ENMAP"):
            return FileNameParser.enmap

    @staticmethod
    def prisma(file_name: str) -> Dict[str, str]:
        """
        Parses the prisma file name to yield usable information
        """
        # Remove everything after the .
        file_id = file_name.split(".")[0]
        parts = file_id.split("_")
        # Prisma is of the form PRS_<PROCESSING_LEVEL>_<PRODUCT_TYPE>_<ACQUISITION_START>_<ACQUISITION_END>_<SEQUENCE_NUMBER>
        return {
            ApplicableFields.PLATFORM.value: PLATFORM_MAPPINGS.get(parts[0]),
            ApplicableFields.PROCESSING_LEVEL.value: ProcessingLevels(parts[1]).value,
            ApplicableFields.PRODUCT_TYPE.value: parts[2],
            ApplicableFields.DATETIME.value: datetime.datetime.strptime(
                parts[3], "%Y%m%d%H%M%S"
            ),
        }

    @staticmethod
    def landsat_09(file_name: str) -> Dict[str, str]:
        """
        Parses LC09 file names
        """
        file_id = file_name.split(".")[0]
        parts = file_id.split("_")

        # LC09_<PROCESSING_LEVEL>_<Path and Row>_<Acquitisiton Date>_<Processing Date>_<Collection number>_<Collection Category>_<Product_Type>_<Sensor Band>

        return {
            ApplicableFields.PLATFORM.value: PLATFORM_MAPPINGS.get(parts[0]),
            ApplicableFields.PROCESSING_LEVEL.value: ProcessingLevels(parts[1]).value,
            ApplicableFields.DATETIME.value: datetime.datetime.strptime(
                parts[3], "%Y%m%d"
            ),
            ApplicableFields.PRODUCT_TYPE.value: parts[7],
            ApplicableFields.BAND.value: parts[8],
        }

    @staticmethod
    def enmap(file_name: str) -> Dict[str, str]:
        """
        Parses EnMAP folder/file names.
        Pattern: ENMAP01-____L2A-DT{datatake_id}_{datetime}Z_{tile_id}_V{version}_{processing_datetime}Z
        """
        # Split on hyphens to get: ['ENMAP01', '____L2A', 'DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z']
        parts = file_name.split("-")
        platform_code = parts[0]  # ENMAP01

        # The DT segment contains the core metadata separated by underscores
        dt_segment = parts[2]
        dt_parts = dt_segment.split("_")
        # dt_parts: ['DT0000059367', '20240128T063655Z', '018', 'V010506', '20260305T173243Z']

        datetime_str = dt_parts[1].rstrip("Z")  # '20240128T063655'

        return {
            ApplicableFields.PLATFORM.value: PLATFORM_MAPPINGS.get(platform_code),
            ApplicableFields.PROCESSING_LEVEL.value: ProcessingLevels.L2A.value,
            ApplicableFields.DATETIME.value: datetime.datetime.strptime(
                datetime_str, "%Y%m%dT%H%M%S"
            ),
            ApplicableFields.PRODUCT_TYPE.value: "STANDARD_ALL",
        }
