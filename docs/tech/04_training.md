# Chapter 4: Training

This chapter walks through how Allotrope trains its seven foundation models. It begins with the shared training abstraction — the `FoundationTrainer` base class and the `trainer_factory` registry — and then visits each concrete trainer in turn. Each visit has three parts: an implementation walkthrough with line-anchored links, a plain-language explanation of the objective being optimized, and small numerical examples so the math is concrete rather than incantatory.

The chapter is split into per-section files so you can jump straight to the trainer you need.

| Section | File | Tagline |
|---|---|---|
| 4.1 | [01_training_abstraction.md](04_training/01_training_abstraction.md) | The `FoundationTrainer` base class, `trainer_factory`, the loop, and checkpointing. |
| 4.2 | [02_spatial_autoencoder_trainer.md](04_training/02_spatial_autoencoder_trainer.md) | Plain autoencoder baseline, masked $L_2$. |
| 4.3 | [03_spatial_masked_autoencoder_trainer.md](04_training/03_spatial_masked_autoencoder_trainer.md) | Random pixel masking with $L_2$, mild mask range $[0.13, 0.25]$. |
| 4.4 | [04_spatial_masked_autoencoder_trainer_l1.md](04_training/04_spatial_masked_autoencoder_trainer_l1.md) | The $L_2 \to L_1$ pivot; heavy masking $[0.50, 0.75]$. |
| 4.5 | [05_unnormalized_l1_trainer.md](04_training/05_unnormalized_l1_trainer.md) | $L_1$ in raw Kelvin space; the only truly unnormalized architecture. |
| 4.6 | [06_normalized_masked_autoencoder_trainer.md](04_training/06_normalized_masked_autoencoder_trainer.md) | $L_1$ with normalization baked back in; explicit mask channels. |
| 4.7 | [07_segformer_mae_trainer.md](04_training/07_segformer_mae_trainer.md) | True MAE-style token removal; trimmed $L_1$; two-pass complementary validation. |
| 4.8 | [08_hyperspectral_segformer_mae_trainer.md](04_training/08_hyperspectral_segformer_mae_trainer.md) | 165-band hyperspectral; $L_1$ + ramped SAM; gradient accumulation. |
| 4.9 | [09_training_config_notes.md](04_training/09_training_config_notes.md) | Config surface: `TrainingConfig` knobs and `InferenceConfig` hooks that bleed into training. |
| 4.10 | [10_summary_table.md](04_training/10_summary_table.md) | Side-by-side table and decision tree for picking a trainer. |

## Themes

1. **Masked reconstruction** as a self-supervised pretext task — the model only sees pixels and validity masks, never labels.
2. **$L_2$ vs $L_1$** — $L_1$ is robust to outliers, which is exactly the right inductive bias when anomalies are the things we want to detect at inference.
3. **Where normalization lives** — most models bake input normalization into `forward()`; one variant deliberately does not.
4. **Filtering and masking arithmetic** — patches below 40% validity are dropped; invalid pixels are zeroed; loss is averaged only over the cells we want the model to predict.
