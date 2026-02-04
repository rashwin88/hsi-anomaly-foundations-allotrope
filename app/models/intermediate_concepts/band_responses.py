"""
Defines the response when a dataset is requested for specific bands by indices or
wavelengths.
"""

from typing import Dict, Union
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, SkipValidation


class BandByIndexResponse(BaseModel):
    """
    Defines the response when a dataset is asked for a set of bands by their indices
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: SkipValidation[Union[np.ndarray, np.ma.MaskedArray]] = Field(
        ..., description="The output of the band extraction"
    )
    band_lookup: Dict[int, int] = Field(
        ...,
        description="A lookup dictionary that tells us which band is in which remapped index",
    )


class BandByWavelengthResponse(BaseModel):
    """
    Response model when a dataset is requested for bands by wavelengths
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: SkipValidation[Union[np.array, np.ma.MaskedArray]] = Field(
        ..., description="The output of the band extraction"
    )

    band_lookup: Dict[float, int] = Field(
        ...,
        description="The lookup dict between the specified wavelengths and the index in the output cube",
    )
