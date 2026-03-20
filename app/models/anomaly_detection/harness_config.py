"""
Configuration models for the anomaly detection inference harness.
"""

from pydantic import BaseModel, Field, model_validator


class PatchConfig(BaseModel):
    """Spatial patching parameters for patch-level inference."""

    patch_height: int = Field(..., gt=0)
    patch_width: int = Field(..., gt=0)
    stride: int = Field(..., gt=0)


class InferenceHarnessConfig(BaseModel):
    """
    Controls how the inference harness feeds data to an AnomalyDetector.

    When `patch_config` is None the detector receives the full scene.
    When set, the harness tiles the scene into patches, runs detection on
    each batch, and reassembles scores via overlap-averaging.
    """

    patch_config: PatchConfig | None = Field(
        default=None,
        description="None = full-scene inference. Set to enable patch-level inference.",
    )

    batch_size: int = Field(
        default=1,
        gt=0,
        description="Number of patches per call to detect_batch.",
    )

    fit_on_full_scene: bool = Field(
        default=True,
        description="If True the harness calls detector.fit on the full scene before patch-level detection.",
    )

    @model_validator(mode="after")
    def _batch_size_requires_patching(self) -> "InferenceHarnessConfig":
        if self.patch_config is None and self.batch_size != 1:
            raise ValueError(
                "batch_size > 1 is only meaningful with patch_config set."
            )
        return self
