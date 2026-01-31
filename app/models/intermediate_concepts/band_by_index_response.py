"""
Defines the response when a dataset is requested for specific bands by indices
"""

from typing import Dict, Union
import numpy as np
from pydantic import BaseModel, Field


class BandByIndexResponse(BaseModel):
    """
    Defines the response when a dataset is asked for a set of bands by their indices
    """

    output: Union[np.ndarray, np.ma.MaskedArray] = Field(
        ..., description="The output of the band extraction"
    )
    band_lookup: Dict[int, int] = Field(
        ...,
        description="A lookup dictionary that tells us which band is in which remapped index",
    )
