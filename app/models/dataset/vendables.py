"""
Defines vendable datasets for each dataset builder
"""

from typing import List
from pydantic import BaseModel, Field, ConfigDict, SkipValidation
import numpy as np

from app.models.hyperspectral_concepts.spectral_family import SpectralFamily


class VendableHyperspectralDataset(BaseModel):
    """
    Models a vendable Hyperspectral Dataset that can be used by downstream applications
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    normalized_hyperspectral_cube: SkipValidation[np.ndarray] = Field(
        ..., description="A fully normalized hyperspectral cube"
    )

    validity_cube: SkipValidation[np.ndarray] = Field(
        ...,
        description="The full validity cube. If a band is is in valid then every pixel in that band must be invalid.",
    )

    spectral_family_order: List[SpectralFamily] = Field(
        ...,
        description="An ordered list of spectral families to which each band belongs",
    )

    band_cw_order: List[float] = Field(
        ...,
        description="A list which has the CW of each band in order of occurence in the cube.",
    )
