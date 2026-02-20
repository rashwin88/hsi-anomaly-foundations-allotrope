"""
Defines a patching plan, the concept of a patch
"""

from typing import List, Tuple

from pydantic import BaseModel, Field, SkipValidation, ConfigDict

from app.models.patches.patching_request import PatchRequest


class PatchingPlan(BaseModel):
    """
    Defines a patching plan
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    originating_request: PatchRequest = Field(
        ..., description="The request through which the patch plan was generated"
    )

    patch_coordinates: List[Tuple[int, int]] = Field(
        ..., description="The top left corner coordinates of each patch"
    )
