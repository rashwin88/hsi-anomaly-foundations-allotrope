"""
Configuration models for foundation model training.

A single TrainingConfig fully describes a training run:
model architecture, data loading, epoch definition, LR schedule,
and checkpointing.
"""

from enum import Enum
from typing import Annotated, Literal, Union, Tuple

from app.utils.general_utils import s3_config
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
    SPECTRAL_STEM_MAE = "spectral_stem_mae"
    SPECTRAL_STEM_FINETUNE = "spectral_stem_finetune"



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


class SpectralStemMAEConfig(HyperspectralSegFormerMAEConfig):
    """The SegFormer MAE with a swappable spectral stem - the three variants of the
    SpectralEarth pretraining experiment (docs/proposals/spectral-attention-bottleneck.md).
    Inherits every SegFormer, masking and loss field; `in_channels` is unused (the band
    count comes from `bands` or, for cross_attention, from each batch's wavelengths)."""

    decoder_type: Literal["segformer", "progressive"] = Field(default="segformer", description="Reconstruction "
        "decoder: SegFormer's (one-shot PixelShuffle at 1/4 resolution) or progressive (x2 stages with a masked "
        "full-resolution skip from the stem; FT09, research/experiments/decoder-capacity/).")
    decoder_stage_dims: list[int] = Field(default_factory=lambda: [192, 96], min_length=2, max_length=2,
        description="progressive decoder: channels of its 1/2- and full-resolution stages.")
    model_type: Literal["spectral_stem_mae"] = "spectral_stem_mae"
    stem: Literal["linear", "gate", "cross_attention", "reduced_attention"] = "cross_attention"
    bands: int = Field(default=202, description="Fixed input band count for the linear and gate stems.")
    compressed_channels: int = 32
    token_dim: int = 32
    attention_heads: int = 4
    gate_hidden: int = 32
    bin_nm: float = Field(default=40.0, gt=0, description="reduced_attention: wavelength bin width for the attention tokens.")
    band_weighting: Literal["count", "width"] = Field(
        default="count", description="reduced_attention: how bands are averaged in bins and the skip path. "
        "'width' weights each band by the spectrum it covers, so a sensor's band density does not change "
        "the result (docs/lld/spectral-stem.md); 'count' (one vote per band) only for E001-E004 checkpoints.")
    band_augment_prob: float = Field(
        default=0.0, ge=0, le=1, description="Wavelength stems: chance a training batch's input is a random "
        "subset of its bands (thinning, gaps or an even spacing); the loss still covers every band.")
    band_jitter_nm: float = Field(
        default=0.0, ge=0, description="On augmented batches, shift each input band's stated wavelength by "
        "up to +/- this much (values unchanged), so wavelength networks stay smooth between trained bands.")
    band_list_path: str | None = Field(
        default=None, description="JSON with a wavelengths_nm list (e.g. a shard set's stats.json): every "
        "sample is aligned onto it by wavelength in the loader workers; bands it lacks become nodata. "
        "Needed when samples carry different band lists (EnMAP native, PRISMA, mixed sets).")
    metrics_csv: str | None = Field(
        default=None, description="Append one row per epoch here - normally the run folder under "
        "research/experiments/indradhanu-se-pretrain/<id>/metrics.csv.")
    amp_bf16: bool = Field(default=False, description="Run the model under bf16 autocast (CUDA); "
                           "the loss stays fp32. Roughly halves C's activation memory.")
    early_stop_patience: int | None = Field(
        default=None, ge=1, description="Stop when avg val loss has not improved on its best by "
        "early_stop_min_delta (relative) for this many epochs. None = run all num_epochs.")
    early_stop_min_delta: float = Field(default=0.01, ge=0, le=0.5, description="Relative improvement that counts, e.g. 0.01 = 1%.")
    early_stop_min_epochs: int = Field(default=20, ge=0, description="Never stop before this epoch (lets the SAM ramp finish).")
    checkpoint_s3_uri: str | None = Field(
        default=None, description="s3://bucket/prefix/ - last.pt is uploaded every epoch and best.pt on "
        "each improvement, so a terminated instance loses nothing.")
    auto_resume: bool = Field(
        default=False, description="On start, resume from checkpoint_dir/last.pt, or else from "
        "checkpoint_s3_uri/last.pt, if either exists; otherwise start fresh.")


class FinetuneSource(BaseModel):
    """One sensor's shards for fine-tuning: local shard folders (staged before the run), its share of
    training batches, an optional band list to align its samples onto, and its label format."""
    name: str
    train_dir: str
    test_dir: str
    share: float = Field(gt=0)
    band_list_path: str | None = None
    labels: Literal["enmap", "pseudo"] | None = Field(default=None, description="'enmap': label_* layers -> segmentation "
                                                    "targets; 'pseudo': teacher probabilities from pseudo_dir (FT04).")
    pseudo_dir: str | None = Field(default=None, description="pseudo: folder of <tile key>.npy (5, H, W) uint8 teacher "
                                   "probabilities (scripts/pseudo_label.py); a tile without one is unlabelled.")
    skip_scenes_path: str | None = Field(default=None, description="JSON list of meta.json scene_id values to leave out "
                                         "(e.g. the older copy of a scene processed twice).")


class SpectralStemFinetuneConfig(SpectralStemMAEConfig):
    """Fine-tuning a pretrained spectral-stem model on several sensors, with a segmentation head
    (docs/proposals/fine-tuning-and-segmentation.md). Inherits every model and run-control field; the
    normalisation curve comes from init_from's checkpoint, never from data (frozen at pretraining)."""

    model_type: Literal["spectral_stem_finetune"] = "spectral_stem_finetune"
    sources: list[FinetuneSource] = Field(min_length=1)
    init_from: str | None = Field(default=None, description="Pretrained checkpoint (local path or s3://) to start from.")
    mode: Literal["probe", "adapt", "joint", "scale", "decoder"] = Field(
        default="joint", description="probe: train the segmentation head only, backbone frozen. adapt: reconstruction "
        "only. joint: reconstruction + segmentation loss on labelled batches. scale: train the uncertainty head only "
        "(Laplace NLL of the expected error on hidden pixels), backbone frozen (docs/proposals/uncertainty-head.md). "
        "decoder: train the reconstruction decoder only (reconstruction loss), everything else frozen - the warm-up "
        "for a new decoder (FT09a).")
    seg_classes: int = Field(default=5, ge=0)
    seg_decoder_dim: int = 128
    scale_decoder_dim: int = Field(default=0, ge=0, description="Uncertainty head width; 0 = no head. Mode 'scale' needs it.")
    gate_pct: float = Field(default=0.0, ge=0, lt=100, description="mixed batches: per source group, the hidden pixels "
                            "with the top gate_pct % reconstruction error (dilated by gate_halo px) are left out of the "
                            "reconstruction loss - the model never practises rebuilding rare odd pixels (FT07, GT-HAD "
                            "style gating; research/experiments/indradhanu-finetune/FT07-gated/). 0 = off.")
    gate_halo: int = Field(default=2, ge=0, description="Gate dilation, px.")
    mask_block: int = Field(default=1, ge=1, description="MAE masking unit in stage-1 tokens: 1 = single 4 px tokens, "
                            "2 = 8 px blocks, so a target and its halo are hidden together (FT08). Inference uses the "
                            "same block as its checkerboard cell.")
    seg_weight: float = Field(default=1.0, ge=0, description="joint: weight of the segmentation loss.")
    plateau_elevation_m: float | None = Field(default=3500.0, description="Label policy: cloud/thin-cloud unknown at or above "
                                              "this mean ground elevation (a guard; no native-v1 scene reaches it).")
    low_water_vapour: float | None = Field(default=750.0, description="Label policy: cloud/thin-cloud unknown below this "
                                           "scene_wv (g/cm2 x 1000; 750 = 0.75 g/cm2, from the 2026-10-16 label QA).")
    cirrus_min_level: int = Field(default=2, ge=1, le=3)
    finetune_metrics_csv: str | None = Field(default=None, description="Per-epoch per-sensor validation and per-class IoU.")
    mixed_batches: bool = Field(default=False, description="adapt / joint: every training batch holds every source, "
                                "share x batch_size samples each (at least 1), instead of one source per batch.")

    seg_head_from: str | None = Field(default=None, description="Checkpoint (local or s3://) whose seg_head.* weights "
                                      "replace the fresh head after init_from (e.g. FT01's probe).")
    seg_class_weights: list[float] | None = Field(default=None, description="Training segmentation loss weights per "
                                                  "class (water, cloud, thin_cloud, shadow, snow); None = equal.")
    implant_prob: float = Field(default=0.0, ge=0, le=1, description="mixed batches: share of training tiles given implanted objects, trained against the clean tile (docs/proposals/detection-supervised-finetuning.md).")
    implant_max_objects: int = Field(default=4, ge=1)
    implant_near_weight: float = Field(default=5.0, ge=1, description="loss weight on hidden pixels within 3 px of an implant.")
    implant_real_path: str | None = Field(default=None, description="real.json from build_implant_bench.py real.")
    implant_psf_sigma: float = 1.4
    implant_sources: list[Literal["signature", "material"]] = Field(default_factory=lambda: ["signature", "material"])
    implant_seed: int = 1
    loader_workers_per_source: int | None = Field(default=None, ge=0, description="DataLoader workers per source "
                                                  "(default data.num_workers); the mixed loader runs one per source.")
    grad_loss_weight: float = Field(default=0.0, ge=0, description="mixed batches: weight of the edge term - L1 between "
                                    "the spatial gradients of reconstruction and input on hidden pixels (0 = off).")

    @model_validator(mode="after")
    def _mixed_batches_not_in_probe(self):
        if self.mixed_batches and self.mode == "probe":
            raise ValueError("mixed_batches is implemented for modes 'adapt' and 'joint'")
        if self.implant_prob and not (self.mixed_batches and self.implant_real_path):
            raise ValueError("implant_prob needs mixed_batches and implant_real_path")
        if self.gate_pct and not self.mixed_batches:
            raise ValueError("gate_pct is implemented for mixed_batches")
        if self.mode == "scale" and not self.scale_decoder_dim:
            raise ValueError("mode 'scale' needs scale_decoder_dim > 0")
        if self.grad_loss_weight and not self.mixed_batches:
            raise ValueError("grad_loss_weight is implemented for mixed_batches only")
        return self


class SpectralCompressorConfig(BaseModel):
    """Architecture config for SpectralCompressor (linear autoencoder)."""

    model_type: Literal["spectral_compressor"] = "spectral_compressor"
    in_channels: int = 224
    compressed_channels: int = 30


# Pydantic picks the right config based on the model_type literal field.
# Adding a new model = add a config class above and extend this union.
ModelSpecificConfig = Annotated[
    Union[SpatialAutoencoderConfig, SpectralCompressorConfig, SpatialMaskedAutoEncoderConfig, NormalizedMaskedAutoEncoderConfig, SegFormerMAEConfig, HyperspectralSegFormerMAEConfig, SpectralStemMAEConfig, SpectralStemFinetuneConfig],
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
    bucket_name: str = s3_config.BUCKET
    region_name: str = s3_config.REGION
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
    def _early_stop_needs_epoch_checkpoints(self) -> "TrainingConfig":
        """Early stopping and auto-resume read and write their state at checkpoint time."""
        m = self.model_config_
        if isinstance(m, SpectralStemMAEConfig) and (m.early_stop_patience or m.auto_resume or m.checkpoint_s3_uri):
            if self.checkpoint.save_every_n_epochs != 1:
                raise ValueError("early_stop_patience / auto_resume / checkpoint_s3_uri need "
                                 "checkpoint.save_every_n_epochs = 1")
        return self

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
