"""
Output model for anomaly detection inference.
"""

from typing import Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SkipValidation


class AnomalyDetectionResult(BaseModel):
    """Immutable result produced by the inference harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score_map: SkipValidation[np.ndarray] = Field(
        ..., description="(H, W) per-pixel anomaly scores."
    )

    source_cube_shape: Tuple[int, ...] = Field(
        ..., description="Shape of the input cube that produced these scores."
    )

    patches_processed: int = Field(
        ..., description="Total patches scored. 1 when running full-scene."
    )
