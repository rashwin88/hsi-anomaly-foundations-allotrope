"""
The map from logical EnMAP components to filenames within the scene folder.

Consumed by EnmapHelper via TEMPLATE_MAPPINGS. The EnMAP counterpart to
prisma_hyperspectral.py, and structurally the odd one out: an EnMAP scene is a
DIRECTORY, so every component resolves to a file suffix rather than a path
inside a single container.

    -SPECTRAL_IMAGE.TIF    the 224-band cube
    -QL_PIXELMASK.TIF      per-band validity
    -QL_QUALITY_*.TIF      cloud, cirrus, cloud shadow, haze, snow
    -METADATA.XML          wavelengths, FWHM, gains, detector boundary

EnMAP is the only sensor shipping ready-made quality masks, which is why
BandFilterConfig.quality_masks_to_apply exists and applies to EnMAP alone.

The wart, repeated from references.py because this is where you meet it: every
entry uses DIRECT_PROPERTY_DEFINITION with `property_name` holding a filename
SUFFIX, not an object attribute. Semantically these are file references; the
type is reused because ReferenceDefinition's validator would reject `file_name`
on this reference type. Don't read `property_name` here as "attribute on the
opened dataset".
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
