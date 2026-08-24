"""
Configuration model for foundation model inference.

An InferenceConfig specifies which model to load, which checkpoint
to restore, and the patch size to infer on.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.models.training.training_config import (
    FoundationModelName,
    ModelSpecificConfig,
)


class PixelStatsOverride(BaseModel):
    """
    Per-scene normalisation stats that replace a checkpoint's baked-in values.

    Needed for uncalibrated sensors. HotSat-1 ships raw DN (~5000 +/- 400); a
    model whose PixelNormalize buffers were fitted on Celsius (~290 +/- 10)
    would see wildly out-of-distribution input and reconstruct noise, drowning
    any real anomaly in the residual.

    Trade-off: scores become scene-relative - comparable within one scene, but
    not across scenes. The anomaly_scoring action records ``normalization_mode``
    so the UI can say so.
    """

    mean: list[float] = Field(..., description="Per-band mean, one per input channel.")
    std: list[float] = Field(..., description="Per-band std. Callers must guard zeros.")
    source: str = Field(..., description="Provenance tag, e.g. 'per_scene_dn_zscore'.")

    @model_validator(mode="after")
    def _lengths_match(self) -> "PixelStatsOverride":
        # A mismatch would otherwise surface as an opaque shape error deep
        # inside PixelNormalize, long after the real mistake.
        if len(self.mean) != len(self.std):
            raise ValueError(
                f"mean has {len(self.mean)} entries but std has {len(self.std)}; "
                "both must be one per input channel."
            )
        return self


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
    pixel_stats_override: PixelStatsOverride | None = Field(
        default=None,
        description="In-memory per-scene stats. When set these win over "
        "pixel_stats_path, which callers should then pass as None.",
    )
    masking_strategy: Literal["checkerboard", "random"] = Field(
        default="checkerboard",
        description="Inference masking strategy. "
        "'checkerboard' uses a deterministic token-level checkerboard pattern. "
        "'random' uses a random 50% mask with its complement for the two passes. "
        "Random masking avoids systematic grid artifacts in the residual map.",
    )
    inference_batch_size: int = Field(
        default=16,
        gt=0,
        description="Number of patches to process in parallel during full-scene inference. "
        "Higher values use more GPU memory but are significantly faster.",
    )
    erosion_kernel_size: int = Field(
        default=15,
        gt=0,
        description="Kernel size for eroding the validity mask at scene boundaries. "
        "Pixels within kernel_size//2 of any invalid pixel are excluded from "
        "reconstruction accumulation and residual computation. "
        "Should be >= OPE kernel size (7) to cover the full receptive field overlap.",
    )
