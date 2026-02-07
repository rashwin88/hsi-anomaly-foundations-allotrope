"""
Defines different template mappings
"""

from enum import Enum
from app.templates.sensor_data_templates.prisma_hyperspectral import (
    PRISMA_HYPERSPECTRAL_TEMPLATE,
)
from app.templates.sensor_data_templates.landsat_thermal import LANDSAT_THERMAL_TEMPLATE


class TemplateIdentifier(str, Enum):
    """
    Identifiers for different templates
    """

    PRISMA_HYPERSPECTRAL = "PRISMA_HYPERSPECTRAL"
    LANDSAT_THERMAL = "LANDSAT_THERMAL"


TEMPLATE_MAPPINGS = {
    TemplateIdentifier.PRISMA_HYPERSPECTRAL: PRISMA_HYPERSPECTRAL_TEMPLATE,
    TemplateIdentifier.LANDSAT_THERMAL: LANDSAT_THERMAL_TEMPLATE,
}
