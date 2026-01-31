"""
Defines a response model for requesting band by wavelength
"""

from typing import Dict
import numpy as np
from pydantic import BaseModel, Field


class BandByWavelengthResponse(BaseModel):
    """
    Response model when a dataset is requested for bands by wavelengths
    """

    output: Union[np.array, np.ma.MaskedArray]
