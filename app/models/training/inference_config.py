"""
Configuration model for foundation model inference.

An InferenceConfig specifies which model to load, which checkpoint
to restore, and the patch size to infer on.
"""

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
