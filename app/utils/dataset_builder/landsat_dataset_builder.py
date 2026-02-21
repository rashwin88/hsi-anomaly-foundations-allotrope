"""
Dataset builder for landsat thermal datasets
"""

import logging
from typing import Union

from pystac import Item
import numpy as np

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.abstract_classes.file_helper import FileHelper
from app.models.dataset.vendables import VendableThermalDataset
from app.utils.stac.stac_utils.stac_items import StacCreator
from app.models.file_processing.sources import FileSourceConfig
from app.utils.image_transformation.image_cube_operations import (
    ImageCubeOperations,
    CubeRepresentation,
)
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier
from app.utils.data_transformations.l2sp_dn_to_temperature_transformer import (
    Lc09L2spStTransformer,
)

from app.statistical_models.b10_adaptive_cloud_masker import B10AdaptiveCloudMasker
from app.models.units.surface_temperature import Temperature
from app.utils.files.tif_helper import TIFHelper

logger = logging.getLogger("LandsatDataBuilder")
logger.setLevel(logging.INFO)


class LandsatDataBuilder(DatasetBuilder):
    """
    Data builder for landsat files.

    Responsibilities:
    - Read Landsat metadata.
    - Convert DN values to ST using the configured transformer.
    - Assemble normalized hyperspectral cube and validity masks for downstream use.
    - Produce a vendable dataset in a canonical BSQ representation.
    """

    def __init__(self, file_source_configuration: FileSourceConfig):
        """
        Initializes the builder and prepares metadata and helpers.
        """
        super().__init__(file_source_configuration=file_source_configuration)
        # Create the STAC item as early as possible for metadata access.
        logger.info("Creating STAC item for PRISMA dataset.")
        self._stac_item = StacCreator(
            file_path=self.file_source_config.source_path
        ).build_stack()
        # Load the helper for TIF access and parse band metadata.
        logger.info("Initializing HE5 helper and extracting band metadata.")
        self._file_helper: TIFHelper = self.initialize_helper()
        logger.info("Band metadata loaded for SWIR/VNIR.")
        self.dn_to_surface_temperature_transformer = Lc09L2spStTransformer()
        self.b10_cloud_masker = B10AdaptiveCloudMasker()
        logger.info("Loaded all transformations")
        self.cube_reshaper = ImageCubeOperations()

    @property
    def stac_item(self) -> Item:
        return self._stac_item

    @property
    def file_helper(self) -> FileHelper:
        return self._file_helper

    @property
    def band_information(
        self,
    ) -> None:
        return None

    @property
    def default_cube_representation(self) -> CubeRepresentation:
        return CubeRepresentation.BSQ

    def initialize_helper(self) -> TIFHelper:
        return TIFHelper(
            file_source_config=self.file_source_config,
            template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.PRISMA_HYPERSPECTRAL),
        )

    def _transformation_pipeline(
        self,
        input_data: Union[np.ndarray, np.ma.MaskedArray],
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        A transformation pipeline for the dataset.
        """
        # Convert to reflectance values
        dn_to_st_transformed = self.dn_to_surface_temperature_transformer.transform(
            input_data=input_data, unit=Temperature.CELSIUS
        )
        return dn_to_st_transformed

    def extract_band_information(self) -> None:
        return None

    def vend_dataset(self) -> VendableThermalDataset:
        """
        Returns a vendable thermal dataset
        """

        # First collect the thermal image in its native format with masking
        logger.info("Collecting raw image")
        raw_masked_image = self.file_helper.extract_specific_bands(
            bands=[], masking_needed=True, mode="all"
        )
        raw_image = raw_masked_image.data
        # Flips the bits in the raw data mask 1 is valid and 0 is invalid
        validity_mask = (~raw_masked_image.mask).astype(np.int8)
        logger.info("Valid pixels %s", validity_mask.sum())
        logger.info("Validity mask shape %s", validity_mask.shape)

        # Transform the raw image into ST
        st_image = self._transformation_pipeline(raw_image)
        print(f"Max Temp = {st_image.max()}")
        print(f"Min Temp = {st_image.min()}")

        # get the cloud masks
        # Train the masker
        self.b10_cloud_masker.configure()
        # Always use the original mask and not the flipped mask
        self.b10_cloud_masker.train(
            input_cube=np.ma.MaskedArray(data=st_image, mask=raw_masked_image.mask)
        )
        cloud_detection = self.b10_cloud_masker.predict(
            np.ma.MaskedArray(data=st_image, mask=raw_masked_image.mask)
        )
        # Flip the cloud mask 0 = cloud, 1 = no cloud
        cloud_mask = (~cloud_detection.cloud_mask).astype(np.int8)
        logger.info("Unclouded Pixels %s", cloud_mask.sum())
        logger.info("Cloud Mask Shape %s", cloud_mask.shape)
        # Get the overall mask
        overall_mask = cloud_mask * validity_mask

        return VendableThermalDataset(
            normalized_thermal_cube=st_image,
            validity_cube=overall_mask,
            pure_validity_mask=validity_mask,
            cloud_mask=cloud_mask,
        )
