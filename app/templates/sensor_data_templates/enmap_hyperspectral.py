"""
Defining a template for EnMAP L2A Hyperspectral Data.
Each component maps to a file suffix within the scene folder.
"""

from typing import Dict

from app.models.hyperspectral_concepts.references import (
    ReferenceDefinition,
    ReferenceType,
)
from app.models.hyperspectral_concepts.file_components import EnmapFileComponents


ENMAP_HYPERSPECTRAL_TEMPLATE: Dict[EnmapFileComponents, ReferenceDefinition] = {
    EnmapFileComponents.SPECTRAL_IMAGE: ReferenceDefinition(
        description="The 224-band spectral image GeoTIFF",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-SPECTRAL_IMAGE.TIF",
    ),
    EnmapFileComponents.PIXEL_MASK: ReferenceDefinition(
        description="Per-band pixel validity mask (224 bands, uint8, 0=invalid 1=valid)",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-QL_PIXELMASK.TIF",
    ),
    EnmapFileComponents.QUALITY_CLOUD: ReferenceDefinition(
        description="Cloud quality mask (single band, uint8)",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-QL_QUALITY_CLOUD.TIF",
    ),
    EnmapFileComponents.QUALITY_CIRRUS: ReferenceDefinition(
        description="Cirrus quality mask (single band, uint8)",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-QL_QUALITY_CIRRUS.TIF",
    ),
    EnmapFileComponents.QUALITY_CLOUDSHADOW: ReferenceDefinition(
        description="Cloud shadow quality mask (single band, uint8)",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-QL_QUALITY_CLOUDSHADOW.TIF",
    ),
    EnmapFileComponents.QUALITY_HAZE: ReferenceDefinition(
        description="Haze quality mask (single band, uint8)",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-QL_QUALITY_HAZE.TIF",
    ),
    EnmapFileComponents.QUALITY_SNOW: ReferenceDefinition(
        description="Snow quality mask (single band, uint8)",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-QL_QUALITY_SNOW.TIF",
    ),
    EnmapFileComponents.QUALITY_CLASSES: ReferenceDefinition(
        description="Quality classes mask (single band, uint8)",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-QL_QUALITY_CLASSES.TIF",
    ),
    EnmapFileComponents.METADATA_XML: ReferenceDefinition(
        description="XML metadata sidecar file",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="-METADATA.XML",
    ),
}
