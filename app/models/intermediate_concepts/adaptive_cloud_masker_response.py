"""
Defines a model for the adaptive cloud masker response
"""

from typing import Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from sklearn.mixture import GaussianMixture


class AdaptiveCloudMaskerResponse(BaseModel):
    """
    Defines the response of the Adaptive cloud masker
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cloud_mask: np.ndarray = Field(
        ..., description="The final cloud mask as a boolean array"
    )
    n_comp: int = Field(
        ..., description="The number of components used in the Gaussian Mixture Model"
    )
    model: GaussianMixture = Field(..., Field="The gaussian mixture model")
    anchors: Any = Field(..., description="The percentile probes used in the model")
    pixels_masked: int = Field(
        ..., description="The total number of pixels masked by the model"
    )
