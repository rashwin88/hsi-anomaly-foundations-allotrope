"""
Defining a template for landsat thermal
"""

from typing import Dict
from weakref import ref

from app.models.hyperspectral_concepts.references import (
    ReferenceDefinition,
    ReferenceType,
)

from app.models.hyperspectral_concepts.file_components import (
    ThermalComponents,
)

LANDSAT_THERMAL_TEMPLATE: Dict[ThermalComponents, ReferenceDefinition] = {
    ThermalComponents.COORDINATE_REFERENCE_SYSTEM: ReferenceDefinition(
        description="The coordinate reference system of the thermal band",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="crs",
    ),
    ThermalComponents.BOUNDS: ReferenceDefinition(
        description="The bounds of the spatial data in UTM coords",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="bounds",
    ),
    ThermalComponents.WIDTH: ReferenceDefinition(
        description="Width of the dataset",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="width",
    ),
    ThermalComponents.HEIGHT: ReferenceDefinition(
        description="Width of the dataset",
        reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
        property_name="height",
    ),
}
