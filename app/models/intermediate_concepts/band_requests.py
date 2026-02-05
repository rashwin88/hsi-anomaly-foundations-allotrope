"""
Implements the models for requesting bands from a dataset
"""

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.images.cube_representation import CubeRepresentation


class BandRequestOptions(BaseModel):
    """
    Options to be specified when requesting for a cube of bands from a dataset
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_representation: CubeRepresentation = Field(
        ..., description="The output cubes desired representational format"
    )

    masking_required: Optional[bool] = Field(
        default=False,
        description="Whether any invalid or masked pixels need to be included in the response",
    )

    spectral_family: Optional[SpectralFamily] = Field(
        default=None,
        description="THe spectral family, VNIR, SWIR .etc from which the bands must be pulled",
    )

    normalization_needed: Optional[bool] = Field(
        ...,
        description="Whether some form of normalization is needed on the output cube.",
    )
