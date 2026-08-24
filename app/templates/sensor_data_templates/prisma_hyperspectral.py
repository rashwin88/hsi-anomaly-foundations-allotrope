"""
The map from logical PRISMA components to their physical location in the .he5.

Consumed by HE5Helper via TEMPLATE_MAPPINGS. This is the file that knows the
cube lives at "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube" - so the helper
does not have to, and one FileHelper contract can serve HDF-EOS, GeoTIFF and
ENVI alike.

PRISMA needs two of the three reference types, which is what makes it the
clearest example of the mechanism:

    FILE_REFERENCE        cube data and error matrices, at internal HDF-EOS
                          dataset paths
    ROOT_METADATA_FIELD   wavelengths (List_Cw_*), bad-band flags, FWHM and the
                          L2Scale{Min,Max} factors - these are root ATTRIBUTES,
                          not datasets, and need a different h5py call

Note the `PRS_L2D_HCO` path segment: HCO is the co-registered product. A
different PRISMA processing level uses a different segment, which is why the
path is data here rather than a constant in the helper.

To support a new sensor with a similar container, write a template like this and
register it in template_mappings.py - no helper changes required.
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
    # Scale Factors
    HyperspectralFileComponents.L2_SCALE_MIN_SWIR: ReferenceDefinition(
        description="Minimum scaling factor of the SWIR bands",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="L2ScaleSwirMin",
    ),
    HyperspectralFileComponents.L2_SCALE_MAX_SWIR: ReferenceDefinition(
        description="Maximum scaling factor of the SWIR bands",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="L2ScaleSwirMax",
    ),
    HyperspectralFileComponents.L2_SCALE_MIN_VNIR: ReferenceDefinition(
        description="Minimum scaling factor of the VNIR bands",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="L2ScaleVnirMin",
    ),
    HyperspectralFileComponents.L2_SCALE_MAX_VNIR: ReferenceDefinition(
        description="Maximum scaling factor of the VNIR bands",
        reference_type=ReferenceType.ROOT_METADATA_FIELD,
        root_metadata_field_name="L2ScaleVnirMax",
    ),
}
