"""
Defines what a patching request to an image must contain
"""

import numpy as np

from pydantic import BaseModel, Field, ConfigDict, SkipValidation


class PatchRequest(BaseModel):
    """
    Defines an intent to break up an image into patches.

    A Patching request is simple. We have a width, a height and a stride.
    We make a critical assumption, the patch cannot be larger in height or in width than the actual image.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_cube: SkipValidation[np.ndarray] = Field(
        ...,
        description="The pixel cube which is to be patched",
    )

    width: int = Field(..., description="The width of the patch in pixels")
    height: int = Field(..., description="the height of the patch in pixels")
    stride: int = Field(
        ...,
        description="The stride length taken as the patch moves horizontally and vertically on the image.",
    )
