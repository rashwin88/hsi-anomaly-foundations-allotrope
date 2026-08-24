"""
FileHelper for EnMAP L2A scenes, which are FOLDERS rather than single files.

Sits beneath EnmapDatasetBuilder. Unlike PRISMA (one .he5) or Landsat (one
.tif), an EnMAP scene is a directory holding the cube, several quality-mask
rasters and an XML sidecar. So FileSourceConfig.source_path points at the
directory, and this class locates members by filename suffix.

    *-SPECTRAL_IMAGE.TIF     the cube, BSQ (C, H, W), 224 bands
    *-QL_QUALITY_CLOUD.TIF   cloud, cirrus, haze, cloud shadow, snow
    *-METADATA.XML           wavelengths, FWHM, gains, detector boundary

Wavelengths and FWHM come from the XML via EnmapXmlParser, not from the raster.

Two consequences worth knowing. EnMAP is the only sensor supplying ready-made
quality masks, which is why BandFilterConfig.quality_masks_to_apply exists and
applies to EnMAP alone. And the template here overloads
DIRECT_PROPERTY_DEFINITION to carry filename suffixes rather than dataset
attributes - semantically closer to a file reference, but the validator would
reject file_name on that reference type.

This one DOES parameterise the generic (FileHelper[EnmapMetadata]), so
file_metadata is properly typed - unlike the older HE5Helper and TIFHelper.
"""

import os
import logging
from typing import Dict, List, Optional, Literal, Union

import numpy as np
import rasterio

from app.abstract_classes.file_helper import FileHelper
from app.models.file_processing.sources import FileSourceConfig
from app.models.file_processing.enmap_metadata import EnmapMetadata
from app.models.hyperspectral_concepts.file_components import EnmapFileComponents
from app.models.hyperspectral_concepts.references import ReferenceDefinition
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.utils.files.enmap_xml_parser import EnmapXmlParser


logger = logging.getLogger("EnmapHelper")
logger.setLevel(logging.INFO)


class EnmapHelper(FileHelper[EnmapMetadata]):
    """
    File helper for EnMAP L2A folder-based scenes.
    source_path on FileSourceConfig is the folder path.
    """

    def __init__(
        self,
        file_source_config: FileSourceConfig,
        template: Dict[EnmapFileComponents, ReferenceDefinition],
    ):
        super().__init__(file_source_config=file_source_config, template=template)
        self.scene_folder: str = file_source_config.source_path
        self.scene_name: str = os.path.basename(self.scene_folder)
        self._xml_parser = EnmapXmlParser(
            self._resolve_path(EnmapFileComponents.METADATA_XML)
        )
        self._file_metadata: EnmapMetadata = self._construct_metadata_structure()
        self.masked_pixel_value: int = (
            self._file_metadata.spatial_info.background_value
        )

    def _resolve_path(self, component: EnmapFileComponents) -> str:
        """
        Resolves absolute path to a component file using the template suffix.
        Handles both standard (.TIF) and COG (.tiff) naming conventions:
            Standard: SCENE-SPECTRAL_IMAGE.TIF
            COG:      SCENE-SPECTRAL_IMAGE_COG.tiff
        """
        suffix = self._template[component].property_name
        standard_path = os.path.join(self.scene_folder, self.scene_name + suffix)
        if os.path.exists(standard_path):
            return standard_path
        # Try COG variant: -SPECTRAL_IMAGE.TIF → -SPECTRAL_IMAGE_COG.tiff
        cog_suffix = suffix.replace(".TIF", "_COG.tiff").replace(".XML", ".XML")
        cog_path = os.path.join(self.scene_folder, self.scene_name + cog_suffix)
        if os.path.exists(cog_path):
            return cog_path
        # Fall back to standard path (will error downstream with a clear message)
        return standard_path

    @property
    def file_metadata(self) -> EnmapMetadata:
        return self._file_metadata

    @property
    def template(self) -> Dict[EnmapFileComponents, ReferenceDefinition]:
        return self._template

    def _construct_metadata_structure(self) -> EnmapMetadata:
        """Delegates to EnmapXmlParser to parse the XML sidecar file."""
        return self._xml_parser.parse()

    def access_dataset(self, path: str) -> np.ndarray:
        """Opens a TIF at the given path and reads all bands."""
        with rasterio.open(path) as src:
            return src.read()

    def extract_specific_bands(
        self,
        bands: Union[List[int], int] = 1,
        masking_needed: Optional[bool] = False,
        spectral_family: Optional[SpectralFamily] = None,
        mode: Literal["all", "specific"] = "specific",
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        Reads bands from the SPECTRAL_IMAGE TIF.
        Bands are 1-based (rasterio convention).
        """
        spectral_path = self._resolve_path(EnmapFileComponents.SPECTRAL_IMAGE)
        with rasterio.open(spectral_path) as src:
            if mode == "all":
                data = src.read(masked=masking_needed)
            else:
                if isinstance(bands, int):
                    bands = [bands]
                data = src.read(bands, masked=masking_needed)
        return data

    def extract_pixel_mask(self) -> np.ndarray:
        """
        Reads the 224-band pixel validity mask TIF.
        Returns uint8 array of shape (C, H, W) where 0=valid, 1=flagged.
        Note: callers must invert before use as a validity mask.
        """
        mask_path = self._resolve_path(EnmapFileComponents.PIXEL_MASK)
        with rasterio.open(mask_path) as src:
            return src.read()

    def extract_quality_mask(self, component: EnmapFileComponents) -> np.ndarray:
        """
        Reads a single-band quality mask TIF (cloud, cirrus, haze, etc.).
        Returns uint8 array of shape (H, W).
        """
        mask_path = self._resolve_path(component)
        with rasterio.open(mask_path) as src:
            return src.read(1)
