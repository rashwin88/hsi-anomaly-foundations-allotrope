"""
Extracts scene metadata from filenames, before anything opens the file.

Satellite products encode platform, acquisition date, processing level and
scene identity in the filename itself:

    PRS_L2D_STD_20231229050902_..._0001.he5      PRISMA
    LC09_L2SP_144052_20240101_..._T1_ST_B10.TIF  Landsat 9
    ENMAP01-____L2A-DT000..._20240128T063655Z    EnMAP

Used by StacCreator to populate a STAC item, and by scene onboarding to
duplicate-check a new upload before committing to the expensive work of vending
it. Parsing a string is cheap; opening a 2 GB HDF5 file to learn its date is not.

Dispatch is by filename PREFIX (`PRS`, `LC09`, `ENMAP01`), so a renamed file
parses as the wrong sensor or not at all. That is deliberate - the naming
conventions are the providers', not ours - but it does mean the pipeline expects
files to arrive with their original names.
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
        elif file_name.startswith(("LC08", "LC09")):
            return FileNameParser.landsat
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
    def landsat(file_name: str) -> Dict[str, str]:
        """
        Parses Landsat file names (LC08, LC09).
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
