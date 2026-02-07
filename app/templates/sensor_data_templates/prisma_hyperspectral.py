"""
Defining a template for Prisma Data
"""

from typing import Dict

from app.models.hyperspectral_concepts.references import (
    ReferenceDefinition,
    ReferenceType,
)
from app.models.hyperspectral_concepts.file_components import (
    HyperspectralFileComponents,
)

PRISMA_HYPERSPECTRAL_TEMPLATE: Dict[
    HyperspectralFileComponents, ReferenceDefinition
] = {
    HyperspectralFileComponents.SWIR_CUBE_DATA: ReferenceDefinition(
        description="The file name corresponding to the SWIR cube data",
        reference_type=ReferenceType.FILE_REFERENCE,
        file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube",
    ),
    HyperspectralFileComponents.VNIR_CUBE_DATA: ReferenceDefinition(
        description="The file name corresponding to the VNIR cube data",
        reference_type=ReferenceType.FILE_REFERENCE,
        file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube",
    ),
    HyperspectralFileComponents.VNIR_PIXEL_ERR_MATRIX: ReferenceDefinition(
        description="The error matrix corresponding to the VNIR",
        reference_type=ReferenceType.FILE_REFERENCE,
        file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX",
    ),
    HyperspectralFileComponents.SWIR_PIXEL_ERR_MATRIX: ReferenceDefinition(
        description="The error matrix corresponding to the SWIR",
        reference_type=ReferenceType.FILE_REFERENCE,
        file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_PIXEL_L2_ERR_MATRIX",
    ),
    # SWIR Wavelengths
    HyperspectralFileComponents.SWIR_CENTRAL_WAVELENGTH_LIST: ReferenceDefinition(
        description="Metadata reference for the list of central wavelengths of SWIR",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="List_Cw_Swir",
    ),
    HyperspectralFileComponents.SWIR_CENTRAL_WAVELENGTH_FLAGS: ReferenceDefinition(
        description="Metadata reference for the flags for central wavelengths of SWIR",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="List_Cw_Swir_Flags",
    ),
    HyperspectralFileComponents.SWIR_FWHM_LIST: ReferenceDefinition(
        description="The list of FWHMs of the SWIR bands",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="List_Fwhm_Swir",
    ),
    # VNIR Wavelengths
    HyperspectralFileComponents.VNIR_CENTRAL_WAVELENGTH_LIST: ReferenceDefinition(
        description="Metadata reference for the list of central wavelengths of VNIT",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="List_Cw_Vnir",
    ),
    HyperspectralFileComponents.VNIR_CENTRAL_WAVELENGTH_FLAGS: ReferenceDefinition(
        description="Metadata reference for the flags for central wavelengths of VNIR",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="List_Cw_Vnir_Flags",
    ),
    HyperspectralFileComponents.VNIR_FWHM_LIST: ReferenceDefinition(
        description="The list of FWHMs of the VNIR bands",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="List_Fwhm_Vnir",
    ),
}
