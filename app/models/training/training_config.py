"""
Configuration models for foundation model training.

A single TrainingConfig fully describes a training run:
model architecture, data loading, epoch definition, LR schedule,
and checkpointing.
"""

from enum import Enum
from typing import Annotated, Literal, Union, Tuple

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Foundation model registry
# ---------------------------------------------------------------------------


class FoundationModelName(str, Enum):
    SPATIAL_AUTOENCODER = "spatial_autoencoder"
    SPECTRAL_COMPRESSOR = "spectral_compressor"
    SPATIAL_MASKED_AUTOENCODER = "spatial_masked_autoencoder"
    SPATIAL_MASKED_AUTOENCODER_L1 = "spatial_masked_autoencoder_l1"
    SPATIAL_MASKED_AUTOENCODER_L1_UNNORMALIZED = "spatial_masked_autoencoder_l1_unnormalized"
    NORMALIZED_MASKED_AUTOENCODER = "normalized_masked_autoencoder"
    SEGFORMER_MAE = "segformer_mae"
    HYPERSPECTRAL_SEGFORMER_MAE = "hyperspectral_segformer_mae"



# ---------------------------------------------------------------------------
# Model-specific configs (discriminated union members)
# ---------------------------------------------------------------------------


class SpatialAutoencoderConfig(BaseModel):
    """Architecture config for SpatialAutoencoder."""

    model_type: Literal["spatial_autoencoder"] = "spatial_autoencoder"
    in_channels: int = 1
    base_channels: int = 32
    num_stages: int = 3
    kernel_size: int = 4

class SpatialMaskedAutoEncoderConfig(BaseModel):
    """Architecture config for SpatialMaskedAutoencoder."""

    model_type: Literal["spatial_masked_autoencoder"] = "spatial_masked_autoencoder"
    in_channels: int = 1
    base_channels: int = 32
    num_stages: int = 3
    kernel_size: int = 4
    masking_range: Tuple[float, float] = (0.35, 0.55)

class NormalizedMaskedAutoEncoderConfig(BaseModel):
    """Architecture config for NormalizedMaskedSpatialAutoencoder."""

    model_type: Literal["normalized_masked_autoencoder"] = "normalized_masked_autoencoder"
    in_channels: int = 1
    base_channels: int = 32
    num_stages: int = 3
    kernel_size: int = 4
    masking_range: Tuple[float, float] = (0.35, 0.55)

class SegFormerMAEConfig(BaseModel):
    """Architecture config for SegFormer MAE reconstruction model."""

    model_type: Literal["segformer_mae"] = "segformer_mae"
    in_channels: int = 1
    embed_dims: list[int] = [32, 64, 160, 256]
    num_heads: list[int] = [1, 2, 5, 8]
    reduction_ratios: list[int] = [8, 4, 2, 1]
    num_blocks: list[int] = [2, 2, 2, 2]
    decoder_dim: int = 256
    expansion_ratio: int = 4
    drop_rate: float = 0.3
    mask_ratio: float = 0.5
    trim_fraction: float = 0.0

class HyperspectralSegFormerMAEConfig(BaseModel):
    """Architecture config for Hyperspectral SegFormer MAE with spectral compression."""

    model_type: Literal["hyperspectral_segformer_mae"] = "hyperspectral_segformer_mae"
    in_channels: int = 165
    compressed_channels: int = 24
    embed_dims: list[int] = [32, 64, 160, 256]
    num_heads: list[int] = [1, 2, 5, 8]
    reduction_ratios: list[int] = [8, 4, 2, 1]
    num_blocks: list[int] = [2, 2, 2, 2]
    decoder_dim: int = 256
    expansion_ratio: int = 4
    drop_rate: float = 0.3
    mask_ratio: float = 0.5
    trim_fraction: float = 0.0
    sam_weight: float = 1.0
    sam_ramp_epochs: int = 20
    erosion_kernel_size: int = Field(
        default=1,
        ge=1,
        description="Kernel size for eroding validity mask at patch boundaries during training. "
        "Excludes border pixels whose OPE receptive fields overlap with invalid regions. "
        "1 = minimal erosion (default for training). Inference uses a larger value (e.g. 15).",
    )


class SpectralCompressorConfig(BaseModel):
    """Architecture config for SpectralCompressor (linear autoencoder)."""

    model_type: Literal["spectral_compressor"] = "spectral_compressor"
    in_channels: int = 224
    compressed_channels: int = 30


# Pydantic picks the right config based on the model_type literal field.
# Adding a new model = add a config class above and extend this union.
ModelSpecificConfig = Annotated[
    Union[SpatialAutoencoderConfig, SpectralCompressorConfig, SpatialMaskedAutoEncoderConfig, NormalizedMaskedAutoEncoderConfig, SegFormerMAEConfig, HyperspectralSegFormerMAEConfig],
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
    data_provider: str = Field(
        ..., description="Data provider - prsima, landsat .etc")
    shard_key_template: str = Field(
        # Important to set this properly The default is landsat which is dangerous.
        default="patches/{provider}/{split}/{stage}/w{size}_h{size}_s{stride}/",
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
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Accumulate gradients over N mini-batches before updating. "
        "Effective batch size = batch_size × gradient_accumulation_steps.",
    )

    pixel_stats_path: str | None = Field(
        default=None,
        description="Path to pixel normalization stats JSON (mean/std). "
        "Required for normalized training.",
    )

    @property
    def patch_sizes(self) -> list[int]:
        """Derive patch sizes from the train sample keys."""
        return sorted(self.train_samples_per_epoch.keys())

    def resolve_shard_key(self, split: str, size: int) -> str:
        """Build the S3 shard key for a given split and patch size."""
        return self.shard_key_template.format(
            provider=self.data_provider,
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
# Hot storage config
# ---------------------------------------------------------------------------


class HotStorageConfig(BaseModel):
    """
    Controls local shard caching for fast training.

    Sync-once strategy: shards are downloaded from S3 once and kept on
    local disk for the entire training run (and future runs if
    skip_sync_if_exists=True). No rotation — shardshuffle provides
    different ordering each epoch for data diversity.

    To get fresh data, either set skip_sync_if_exists=False or delete
    the local_cache_dir manually before running.

    Disk usage = (train_shards_per_size + test_shards_per_size) × num_sizes × ~1GB.
    """

    enabled: bool = Field(
        default=False,
        description="Enable hot storage. When False, streams directly from S3.",
    )
    local_cache_dir: str = Field(
        default="/tmp/hot_shards/",
        description="Local directory for cached shards",
    )
    train_shards_per_size: int = Field(
        default=200,
        gt=0,
        description="Number of train shards to sync per patch size (one-time)",
    )
    test_shards_per_size: int = Field(
        default=10,
        gt=0,
        description="Number of test shards to sync per patch size (one-time)",
    )
    skip_sync_if_exists: bool = Field(
        default=True,
        description="Skip S3 download if local shards already exist. "
        "Avoids repeated egress costs across training runs.",
    )
    max_parallel_downloads: int = Field(
        default=30,
        gt=0,
        description="Max concurrent aws s3 cp downloads during sync",
    )


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
# Wandb config
# ---------------------------------------------------------------------------


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "allotrope"
    run_name: str | None = Field(
        default=None, description="Auto-generated from model name + version if None"
    )
    tags: list[str] = []


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
    hot_storage: HotStorageConfig = HotStorageConfig()
    wandb: WandbConfig = WandbConfig()
    learning_rate: float = 1e-3
    device: str | None = Field(
        default=None, description="None = auto-detect via get_device()"
    )
    resume_from: str | None = Field(
        default=None,
        description="Path to a .pt checkpoint to load from.",
    )
    resume_mode: Literal["resume", "finetune"] = Field(
        default="resume",
        description="'resume' restores weights, optimizer, epoch, and scheduler. "
        "'finetune' loads weights only — fresh optimizer, scheduler, and epoch.",
    )

    @model_validator(mode="after")
    def _patch_sizes_compatible_with_model(self) -> "TrainingConfig":
        """
        For spatial models, every patch size must be divisible by
        2^num_stages to ensure clean downsampling and upsampling.
        """
        if isinstance(self.model_config_, (SegFormerMAEConfig, HyperspectralSegFormerMAEConfig)):
            # SegFormer total stride = 4 * 2 * 2 * 2 = 32
            divisor = 32
            for size in self.data.patch_sizes:
                if size % divisor != 0:
                    raise ValueError(
                        f"patch_size {size} must be divisible by "
                        f"SegFormer total stride = {divisor}"
                    )
            return self
        if isinstance(self.model_config_, (SpatialAutoencoderConfig, NormalizedMaskedAutoEncoderConfig)):
            divisor = 2 ** self.model_config_.num_stages
            for size in self.data.patch_sizes:
                if size % divisor != 0:
                    raise ValueError(
                        f"patch_size {size} must be divisible by "
                        f"2^num_stages = {divisor}"
                    )
        return self
