"""
Configuration models for foundation model training.

A single TrainingConfig fully describes a training run:
model architecture, data loading, epoch definition, LR schedule,
and checkpointing.
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Foundation model registry
# ---------------------------------------------------------------------------


class FoundationModelName(str, Enum):
    SPATIAL_AUTOENCODER = "spatial_autoencoder"
    SPECTRAL_COMPRESSOR = "spectral_compressor"


# ---------------------------------------------------------------------------
# Model-specific configs (discriminated union members)
# ---------------------------------------------------------------------------


class SpatialAutoencoderConfig(BaseModel):
    """Architecture config for SpatialAutoencoder."""

    model_type: Literal["spatial_autoencoder"] = "spatial_autoencoder"
    in_channels: int = 1
    base_channels: int = 32
    num_stages: int = 3


class SpectralCompressorConfig(BaseModel):
    """Architecture config for SpectralCompressor (linear autoencoder)."""

    model_type: Literal["spectral_compressor"] = "spectral_compressor"
    in_channels: int = 224
    compressed_channels: int = 30


# Pydantic picks the right config based on the model_type literal field.
# Adding a new model = add a config class above and extend this union.
ModelSpecificConfig = Annotated[
    Union[SpatialAutoencoderConfig, SpectralCompressorConfig],
    Field(discriminator="model_type"),
]


# ---------------------------------------------------------------------------
# Data config
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """
    Controls data loading and epoch definition.

    One epoch = a fixed mix of all patch sizes. For each size, the trainer
    consumes train_samples_per_epoch[size] training samples and
    test_samples_per_epoch[size] validation samples. This keeps a single
    epoch definition across all sizes so a common LR schedule can be applied.

    The shard_key_template is formatted at runtime with:
      - split: "train" or "test"
      - stage: "intermediate" or "final"
      - size: patch size (e.g. 128)
      - stride: size // 2 (always half the patch size)
    """

    train_samples_per_epoch: dict[int, int] = Field(
        ...,
        description="Samples per epoch per patch size for training. "
        "Keys are patch sizes, values are sample counts. "
        "e.g. {64: 2000, 128: 1000, 256: 500, 512: 100}",
    )
    test_samples_per_epoch: dict[int, int] = Field(
        ...,
        description="Samples per epoch per patch size for validation. "
        "Validation runs on ALL sizes every epoch. "
        "e.g. {64: 100, 128: 100, 256: 100, 512: 100}",
    )
    num_epochs: int = Field(..., gt=0, description="Total training epochs")
    shard_key_template: str = Field(
        default="patches/landsat/{split}/{stage}/w{size}_h{size}_s{stride}/",
        description="S3 key template. Formatted with split, stage, size, stride.",
    )
    stage: str = Field(
        default="final",
        description="Shard stage: 'intermediate' or 'final'",
    )
    bucket_name: str = "allotrope-raw-data-india"
    region_name: str = "ap-south-1"
    batch_size: int = 32
    num_workers: int = 2
    shardshuffle: int = 10

    @property
    def patch_sizes(self) -> list[int]:
        """Derive patch sizes from the train sample keys."""
        return sorted(self.train_samples_per_epoch.keys())

    def resolve_shard_key(self, split: str, size: int) -> str:
        """Build the S3 shard key for a given split and patch size."""
        return self.shard_key_template.format(
            split=split,
            stage=self.stage,
            size=size,
            stride=size // 2,
        )

    @model_validator(mode="after")
    def _test_keys_cover_train_keys(self) -> "DataConfig":
        """Every patch size in training must have a test sample count."""
        for size in self.train_samples_per_epoch:
            if size not in self.test_samples_per_epoch:
                raise ValueError(
                    f"patch_size {size} is in train_samples_per_epoch "
                    f"but missing from test_samples_per_epoch"
                )
        return self


# ---------------------------------------------------------------------------
# LR schedule config
# ---------------------------------------------------------------------------


class LRScheduleConfig(BaseModel):
    """
    Learning rate schedule applied once per epoch.

    Supported scheduler types:
      - "cosine": CosineAnnealingLR, decays to min_lr over num_epochs
      - "step": StepLR, decays by step_gamma every step_size epochs
      - "plateau": ReduceLROnPlateau, decays when val loss stalls
    """

    scheduler_type: str = Field(
        default="cosine",
        description="One of: cosine, step, plateau",
    )
    warmup_epochs: int = Field(
        default=5,
        ge=0,
        description="Linear warmup from min_lr to learning_rate",
    )
    min_lr: float = Field(
        default=1e-6,
        description="Minimum learning rate (floor for cosine, factor target for plateau)",
    )
    step_size: int = Field(
        default=10,
        description="Epoch interval for StepLR (only used when scheduler_type='step')",
    )
    step_gamma: float = Field(
        default=0.5,
        description="Multiplicative decay factor for StepLR",
    )


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


class CheckpointConfig(BaseModel):
    """
    Checkpointing strategy.

    Each checkpoint contains model_state_dict, optimizer_state_dict,
    epoch, train_loss, val_losses (per patch size), and the full
    training config serialized as JSON for reproducibility.

    File naming: {model_name}_v{version}_epoch{epoch}.pt
    After each save, only the top keep_top_k checkpoints by val loss are kept.
    """

    checkpoint_dir: str = "checkpoints/"
    save_every_n_epochs: int = 5
    keep_top_k: int = 3
    save_to_s3: bool = False
    s3_checkpoint_key: str | None = None


# ---------------------------------------------------------------------------
# Top-level training config
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """
    Complete specification for a foundation model training run.

    Example JSON:
    {
        "foundation_model_name": "spatial_autoencoder",
        "version": "0.1.0",
        "model_config": {
            "model_type": "spatial_autoencoder",
            "in_channels": 1,
            "base_channels": 32,
            "num_stages": 3
        },
        "data": {
            "num_epochs": 100,
            "train_samples_per_epoch": {"64": 2000, "128": 1000, "256": 500, "512": 100},
            "test_samples_per_epoch": {"64": 100, "128": 100, "256": 100, "512": 100},
            "stage": "final"
        },
        "learning_rate": 1e-3,
        "lr_schedule": {
            "scheduler_type": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-6
        }
    }
    """

    foundation_model_name: FoundationModelName
    version: str = Field(..., description="Semantic version, e.g. '0.1.0'")
    # Named model_config_ with alias because Pydantic v2 reserves model_config
    model_config_: ModelSpecificConfig = Field(..., alias="model_config")
    data: DataConfig
    checkpoint: CheckpointConfig = CheckpointConfig()
    lr_schedule: LRScheduleConfig = LRScheduleConfig()
    learning_rate: float = 1e-3
    device: str | None = Field(
        default=None, description="None = auto-detect via get_device()"
    )

    @model_validator(mode="after")
    def _patch_sizes_compatible_with_model(self) -> "TrainingConfig":
        """
        For spatial models, every patch size must be divisible by
        2^num_stages to ensure clean downsampling and upsampling.
        """
        if isinstance(self.model_config_, SpatialAutoencoderConfig):
            divisor = 2 ** self.model_config_.num_stages
            for size in self.data.patch_sizes:
                if size % divisor != 0:
                    raise ValueError(
                        f"patch_size {size} must be divisible by "
                        f"2^num_stages = {divisor}"
                    )
        return self
