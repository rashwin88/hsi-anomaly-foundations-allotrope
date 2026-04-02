"""
Configuration model for foundation model inference.

An InferenceConfig specifies which model to load, which checkpoint
to restore, and the patch size to infer on.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.models.training.training_config import (
    FoundationModelName,
    ModelSpecificConfig,
)


class InferenceConfig(BaseModel):
    """
    Complete specification for foundation model inference.

    Example JSON:
    {
        "foundation_model_name": "spatial_autoencoder",
        "model_config": {
            "model_type": "spatial_autoencoder",
            "in_channels": 1,
            "base_channels": 32,
            "num_stages": 3
        },
        "checkpoint_path": "checkpoints/spatial_ae/spatial_autoencoder_v0.2.0_epoch50.pt",
        "patch_size": 64
    }
    """

    foundation_model_name: FoundationModelName
    model_config_: ModelSpecificConfig = Field(..., alias="model_config")
    checkpoint_path: str = Field(
        ..., description="Path to a .pt checkpoint to load weights from."
    )
    patch_size: int = Field(..., gt=0, description="Patch size to infer on.")
    stride: int | None = Field(
        default=None,
        description="Sliding window stride for full-scene inference. "
        "Defaults to patch_size // 2 if None.",
    )
    checkerboard_cell_size: int = Field(
        default=1,
        gt=0,
        description="Size of each checkerboard cell in pixels. "
        "1 = single-pixel checkerboard, 2 = 2x2 blocks, etc.",
    )
    device: str | None = Field(
        default=None, description="None = auto-detect via get_device()"
    )
    pixel_stats_path: str | None = Field(
        default=None,
        description="Path to pixel normalization stats JSON (mean/std). "
        "Required for normalized training.",
    )
    masking_strategy: Literal["checkerboard", "random"] = Field(
        default="checkerboard",
        description="Inference masking strategy. "
        "'checkerboard' uses a deterministic token-level checkerboard pattern. "
        "'random' uses a random 50% mask with its complement for the two passes. "
        "Random masking avoids systematic grid artifacts in the residual map.",
    )
