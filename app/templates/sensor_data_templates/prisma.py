from app.models.hyperspectral_concepts.references import (
    ReferenceDefinition,
    ReferenceType,
)

SWIR_CUBE_DATA = ReferenceDefinition(
    description="The file name corresponding to the SWIR cube data",
    reference_type=ReferenceType.FILE_REFERENCE,
    file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube",
)

VNIR_CUBE_DATA = ReferenceDefinition(
    description="The file name corresponding to the VNIR cube data",
    reference_type=ReferenceType.FILE_REFERENCE,
    file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube",
)

VNIR_PIXEL_ERR_MATRIX = ReferenceDefinition(
    description="The error matrix corresponding to the VNIR",
    reference_type=ReferenceType.FILE_REFERENCE,
    file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX",
)

SWIR_PIXEL_ERR_MATRIX = ReferenceDefinition(
    description="The error matrix corresponding to the SWIR",
    reference_type=ReferenceType.FILE_REFERENCE,
    file_name="HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_PIXEL_L2_ERR_MATRIX",
)

# SWIR Wavelengths
SWIR_CENTRAL_WAVELENGTH_LIST = ReferenceDefinition(
    description="Metadata reference for the list of central wavelengths of SWIR",
    reference_type=ReferenceType.ROOT_METADATA_FIELD,
    root_metadata_field_name="List_Cw_Swir",
)

SWIR_CENTRAL_WAVELENGTH_FLAGS = ReferenceDefinition(
    description="Metadata reference for the flags for central wavelengths of SWIR",
    reference_type=ReferenceType.ROOT_METADATA_FIELD,
    root_metadata_field_name="List_Cw_Swir_Flags",
)

SWIR_FWHM_LIST = ReferenceDefinition(
    description="The list of FWHMs of the SWIR bands",
    reference_type=ReferenceType.ROOT_METADATA_FIELD,
    root_metadata_field_name="List_Fwhm_Swir",
)

# VNIR Wavelengths
VNIR_CENTRAL_WAVELENGTH_LIST = ReferenceDefinition(
    description="Metadata reference for the list of central wavelengths of VNIT",
    reference_type=ReferenceType.ROOT_METADATA_FIELD,
    root_metadata_field_name="List_Cw_Vnir",
)

VNIR_CENTRAL_WAVELENGTH_FLAGS = ReferenceDefinition(
    description="Metadata reference for the flags for central wavelengths of VNIR",
    reference_type=ReferenceType.ROOT_METADATA_FIELD,
    root_metadata_field_name="List_Cw_Vnir_Flags",
)

VNIR_FWHM_LIST = ReferenceDefinition(
    description="The list of FWHMs of the VNIR bands",
    reference_type=ReferenceType.ROOT_METADATA_FIELD,
    root_metadata_field_name="List_Fwhm_Vnir",
)
