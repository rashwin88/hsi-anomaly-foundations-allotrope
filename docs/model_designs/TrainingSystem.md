### Foundation Model Training System

#### Motivation

We have multiple foundation models (SpatialAutoencoder, SpectralCompressor, and future architectures like SegFormer) that each require training with different configurations. Training patches exist at multiple sizes (64, 128, 256, 512) in S3 as webdataset shards (~1GB each, ~2000 shards per size). We need a system that is:

- **Config-driven** — one Pydantic config + one JSON file fully describes a training run
- **Model-agnostic** — adding a new model means adding one config class and one trainer class
- **Multi-size** — one epoch trains and validates on all patch sizes
- **Sample-capped** — each "epoch" consumes a fixed number of samples per size (not a full pass over the data)
- **Storage-efficient** — hot storage rotates shards locally to avoid repeated S3 streaming

#### Training Configuration

The top-level `TrainingConfig` is a Pydantic model:

```
TrainingConfig
├── foundation_model_name: FoundationModelName (enum)
├── version: str (semantic version, e.g. "0.1.0")
├── model_config: ModelSpecificConfig (discriminated union)
├── data: DataConfig
├── checkpoint: CheckpointConfig
├── lr_schedule: LRScheduleConfig
├── hot_storage: HotStorageConfig
├── learning_rate: float
└── device: str | None (None = auto-detect)
```

##### Model-Specific Configs via Discriminated Union

Each model type has its own flat config class with a `model_type` literal field. Pydantic's discriminated union selects the right one based on this field:

$$
\text{model\_type} = \begin{cases}
\text{"spatial\_autoencoder"} \rightarrow \text{SpatialAutoencoderConfig}(in\_channels, base\_channels, num\_stages) \\
\text{"spectral\_compressor"} \rightarrow \text{SpectralCompressorConfig}(in\_channels, compressed\_channels)
\end{cases}
$$

Adding a new model: write one new config class with a unique `model_type` literal, add it to the union. No inheritance needed.

##### DataConfig — Multi-Size Epochs with Sample Caps

```
DataConfig
├── train_samples_per_epoch: dict[int,int]    e.g. {64: 2000, 128: 1000, 256: 500, 512: 100}
├── test_samples_per_epoch: dict[int,int]     e.g. {64: 100, 128: 100, 256: 100, 512: 100}
├── num_epochs: int                           total training epochs
├── shard_key_template: str                   S3 key pattern with {split}, {stage}, {size}, {stride}
├── stage: str                                "intermediate" or "final"
├── bucket_name: str
├── region_name: str
├── batch_size: int
├── num_workers: int
└── shardshuffle: int
```

One epoch = a fixed mix of all patch sizes. For each size, the trainer consumes `train_samples_per_epoch[size]` training samples and `test_samples_per_epoch[size]` validation samples. This keeps a single epoch definition across all sizes so a common LR schedule can be applied.

Patch sizes are derived from the keys of `train_samples_per_epoch` — no separate list needed.

The shard key template resolves at runtime via `resolve_shard_key(split, size)`:

```
template: "patches/landsat/{split}/{stage}/w{size}_h{size}_s{stride}/"
resolve_shard_key("train", 128) → "patches/landsat/train/final/w128_h128_s64/"
```

Stride is always `size // 2`.

##### Patch-Level Filtering

Patches with less than 40% valid pixels (combined `pure_validity_mask * custom_quality_mask`) are discarded before training. They don't contribute gradients and don't count toward the epoch sample budget. This is handled in the concrete trainer's `compute_loss()`, not in the base class.

Both `compute_loss()` and `validation_step()` return `(loss, num_valid_samples)` so the training loop can track how many samples actually contributed.

##### Validation Strategy

Validation runs on ALL patch sizes every epoch, not just the current training size. This lets you detect if training on one size degrades performance on others. Test samples use the same filtering (40% valid pixel threshold).

$$
\text{Total train samples per epoch} = \sum_{s \in \text{sizes}} \text{train\_samples\_per\_epoch}[s]
$$

##### LR Schedule

```
LRScheduleConfig
├── scheduler_type: str       "cosine", "step", or "plateau"
├── warmup_epochs: int        linear warmup period
├── min_lr: float             floor learning rate
├── step_size: int            epoch interval for StepLR
└── step_gamma: float         multiplicative decay factor
```

The scheduler steps once per epoch after validation. For `plateau`, it uses the average validation loss across all sizes.

##### CheckpointConfig

```
CheckpointConfig
├── checkpoint_dir: str
├── save_every_n_epochs: int
├── keep_top_k: int          keep best K by average validation loss
├── save_to_s3: bool
└── s3_checkpoint_key: str | None
```

Each checkpoint contains:
- `epoch`, `model_state_dict`, `optimizer_state_dict`, `train_loss`, `val_losses` (per size), `avg_val_loss`
- `config.model_dump()` — the full Pydantic config serialized as JSON for reproducibility

File naming: `{model_name}_v{version}_epoch{epoch}.pt`

After each save, existing checkpoints are sorted by average validation loss and only the top K are kept.

#### Data Loading: Two Modes

##### Mode 1: S3 Streaming (default)

Dataloaders are built once upfront using `shard_pipe_expression_builder`, which constructs `pipe: aws s3 cp ...` expressions. WebDataset streams shards directly from S3.

- `shardshuffle` randomizes shard order each iteration — different data every epoch
- Broken pipe warnings from `aws s3 cp` are expected (stream closes early when sample cap is hit)
- Suitable for quick tests or when local storage is limited

##### Mode 2: Hot Storage (recommended for real training)

```
HotStorageConfig
├── enabled: bool
├── local_cache_dir: str              e.g. "/tmp/hot_shards/"
├── train_shards_per_size: int        e.g. 20
├── test_shards_per_size: int         e.g. 5
└── epochs_per_rotation: int          e.g. 5
```

Instead of streaming every batch from S3, a rotating window of shards is synced to local disk:

1. **Test shards synced once at startup** — fixed for the entire run to ensure consistent validation
2. **Train shards rotated** every `epochs_per_rotation` epochs:
   a. Randomly select `train_shards_per_size` shards per size from S3
   b. Download to local disk via `aws s3 cp` (with tqdm progress)
   c. Build local dataloaders pointing at `.tar` files on disk
   d. Train for `epochs_per_rotation` epochs at local disk speed
   e. Delete local train shards, sync fresh ones

```
Startup:
  sync 5 test shards per size → /tmp/hot_shards/test/   ← stays forever

Rotation 1:
  sync 20 random train shards per size → /tmp/hot_shards/train/
  epoch 1-5: train + validate from local disk (near-instant)
  delete /tmp/hot_shards/train/

Rotation 2:
  sync 20 NEW random train shards → /tmp/hot_shards/train/
  epoch 6-10: same pattern, same test shards
  ...

End:
  delete both train/ and test/
```

Disk usage per rotation = `train_shards_per_size × num_sizes × ~1GB per shard`. With 20 shards × 4 sizes = ~80GB, well within a 256GB RAM server.

Each rotation sees different random shards, so the model trains on diverse data over the course of training. To cover all 2000 shards per size with 20 per rotation, you need 100 rotations × 5 epochs = 500 total epochs.

#### Training Harness Architecture

An abstract base class `FoundationTrainer` provides the training loop, dataloader construction, LR scheduling, checkpointing, device management, and hot storage rotation. Concrete trainers implement three methods:

| Method | Returns | Responsibility |
|--------|---------|---------------|
| `build_model()` | `nn.Module` | Instantiate the model from config |
| `compute_loss(batch, model)` | `(loss, num_kept)` | Training loss + count of valid samples |
| `validation_step(batch, model)` | `(loss_value, num_kept)` | Validation loss + count of valid samples |

A factory function `get_trainer(config)` maps `FoundationModelName` → concrete trainer class.

#### Masked Loss

Invalid pixels are zeroed before the forward pass (to avoid skewing BatchNorm statistics) and excluded from the loss via the combined validity mask:

$$
M = \text{pure\_validity\_mask} \times \text{custom\_quality\_mask}
$$

$$
\mathcal{L} = \frac{\sum_{b,c,i,j} (\hat{x} - x)^2 \cdot M}{\sum M}
$$

This is handled inside `compute_loss()` in each concrete trainer.

#### Patch Size Validation

A Pydantic `model_validator` on `TrainingConfig` ensures that every patch size is divisible by the model's spatial reduction factor:

$$
\forall s \in \text{patch\_sizes}: s \mod 2^{\text{num\_stages}} = 0
$$

A separate validator ensures every patch size in `train_samples_per_epoch` has a corresponding entry in `test_samples_per_epoch`.

#### File Organization

```
app/models/training/training_config.py           — all configs (Pydantic)
app/abstract_classes/foundation_trainer.py        — ABC with training loop + hot storage
app/foundation_models/trainers/
    spatial_autoencoder_trainer.py                 — first concrete trainer
    trainer_factory.py                            — registry + factory
configs/spatial_ae_v0.1.json                      — example config
scripts/train_foundation_model.py                 — CLI entry point
```

#### Running Training

```bash
PYTHONPATH=. python scripts/train_foundation_model.py configs/spatial_ae_v0.1.json
```

#### Performance Notes

- **batch_size** should be ≤ smallest sample cap to avoid over-streaming from S3
- **num_workers** controls parallel data loading — set to 8-16 on multi-core servers
- **shardshuffle** randomizes shard order per iteration — set to 10+ for good mixing
- Hot storage eliminates S3 streaming overhead during training — only rotation syncs hit S3
- On a 48-core / 256GB server, `num_workers: 16` and `train_shards_per_size: 20` works well
