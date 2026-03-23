### Foundation Model Training System

#### Motivation

We have multiple foundation models (SpatialAutoencoder, SpectralCompressor, and future architectures like SegFormer) that each require training with different configurations. Training patches exist at multiple sizes (64, 128, 256, 512) in S3 as webdataset shards. We need a system that is:

- **Config-driven** — one Pydantic config fully describes a training run
- **Model-agnostic** — adding a new model means adding one config class and one trainer class
- **Curriculum-capable** — trains on multiple patch sizes in sequence within a single run
- **Sample-capped** — each "epoch" is a fixed number of samples (not a full pass over the data)

#### Training Configuration

The top-level `TrainingConfig` is a Pydantic model:

```
TrainingConfig
├── foundation_model_name: FoundationModelName (enum)
├── version: str (semantic version, e.g. "0.1.0")
├── model_config: ModelSpecificConfig (discriminated union)
├── data: DataConfig
├── checkpoint: CheckpointConfig
├── learning_rate: float
├── max_epochs: int
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

##### DataConfig — Curriculum Training with Sample Caps

```
DataConfig
├── patch_sizes: list[int]                    e.g. [64, 128, 256, 512]
├── epochs_per_size: int                      e.g. 25
├── train_samples_per_epoch: dict[int,int]    e.g. {64: 10000, 128: 5000, 256: 2000, 512: 500}
├── test_samples_per_epoch: dict[int,int]     e.g. {64: 2000, 128: 1000, 256: 400, 512: 100}
├── bucket_name: str
├── region_name: str
├── batch_size: int
├── num_workers: int
└── shardshuffle: int
```

The training loop iterates patch sizes in order. For each size, it trains for `epochs_per_size` epochs, where each training epoch consumes exactly `train_samples_per_epoch[patch_size]` samples. After each training epoch, a validation pass consumes `test_samples_per_epoch[patch_size]` samples from the test shards. Model weights carry over between sizes — this is curriculum training.

$$
\text{Total train samples} = \sum_{s \in \text{patch\_sizes}} \text{epochs\_per\_size} \times \text{train\_samples\_per\_epoch}[s]
$$

$$
\text{Total test samples} = \sum_{s \in \text{patch\_sizes}} \text{epochs\_per\_size} \times \text{test\_samples\_per\_epoch}[s]
$$

Smaller patches have more samples available and are cheaper to process, so they get higher sample counts. Larger patches are more memory-intensive, so they get fewer.

Test sample counts should be large enough to produce a stable validation loss but small enough to not slow down training. A typical ratio is ~20% of the training count.

Since webdataset with `shardshuffle` randomizes shard order, and patches within shards are pre-shuffled from the data pipeline, each epoch sees a different random subset naturally.

##### CheckpointConfig

```
CheckpointConfig
├── checkpoint_dir: str
├── save_every_n_epochs: int
├── keep_top_k: int          keep best K by validation loss
├── save_to_s3: bool
└── s3_checkpoint_key: str | None
```

Each checkpoint contains:
- `epoch`, `patch_size` (current curriculum stage), `model_state_dict`, `optimizer_state_dict`, `train_loss`, `val_loss`
- `config.model_dump()` — the full Pydantic config serialized as JSON for reproducibility

File naming: `{model_name}_v{version}_size{patch_size}_epoch{epoch}.pt`

After each save, existing checkpoints are sorted by validation loss and only the top K are kept.

#### Training Harness Architecture

An abstract base class `FoundationTrainer` provides the training loop, dataloader construction, checkpointing, and device management. Concrete trainers implement three methods:

| Method | Responsibility |
|--------|---------------|
| `build_model()` | Instantiate the `nn.Module` from config |
| `compute_loss(batch, model)` | Define the loss for one batch |
| `validation_step(batch, model)` | Evaluation logic (returns val loss) |

The base class `train()` method:

```
for patch_size in config.data.patch_sizes:
    train_loader = build_dataloader(patch_size, split="train")
    test_loader = build_dataloader(patch_size, split="test")
    train_cap = config.data.train_samples_per_epoch[patch_size]
    test_cap = config.data.test_samples_per_epoch[patch_size]

    for epoch in range(config.data.epochs_per_size):
        # --- Training ---
        samples_seen = 0
        for batch in train_loader:
            if samples_seen >= train_cap:
                break
            loss = compute_loss(batch, model)
            loss.backward()
            optimizer.step()
            samples_seen += batch_size

        # --- Validation ---
        samples_seen = 0
        val_losses = []
        for batch in test_loader:
            if samples_seen >= test_cap:
                break
            val_result = validation_step(batch, model)
            val_losses.append(val_result)
            samples_seen += batch_size

        if epoch % save_every_n_epochs == 0:
            save_checkpoint()
```

A factory function `get_trainer(config)` maps `FoundationModelName` → concrete trainer class, following the same pattern as the existing `detector_factory.py`.

#### Patch Size Validation

A Pydantic `model_validator` on `TrainingConfig` ensures that every patch size is divisible by the model's spatial reduction factor. For `SpatialAutoencoder` with `num_stages=3`, the reduction is $2^3 = 8$:

$$
\forall s \in \text{patch\_sizes}: s \mod 2^{\text{num\_stages}} = 0
$$

It also validates that every entry in `patch_sizes` has a corresponding key in both `train_samples_per_epoch` and `test_samples_per_epoch`.

#### File Organization

```
app/models/training/training_config.py           — all configs (Pydantic)
app/abstract_classes/foundation_trainer.py        — ABC with training loop
app/foundation_models/trainers/
    spatial_autoencoder_trainer.py                 — first concrete trainer
    trainer_factory.py                            — registry + factory
scripts/train_foundation_model.py                 — CLI entry point
```

#### Masked Loss

Invalid pixels are zeroed before the forward pass and excluded from the loss via the validity mask:

$$
\mathcal{L} = \frac{\sum_{b,c,i,j} (\hat{x} - x)^2 \cdot M}{\sum M}
$$

This is handled inside `compute_loss()` in each concrete trainer, since different models may access the mask differently from the batch dict.
