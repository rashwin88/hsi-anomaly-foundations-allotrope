# Spatial Masked Autoencoder — Experiment Log

Masked autoencoder variant: randomly masks 13–25% of valid pixels per sample and reconstructs only those held-out pixels. Validation uses full reconstruction (no masking) for stable comparison.

## Experiments

|Type| Version | Base Channels | Stages | Bottleneck Shape (128 patch) | LR | Scheduler | Warmup | Epochs | Batch Size | Mask Ratio | Comments |
|--------|---------|--------------|--------|-----------------------------|----|-----------|--------|--------|------------|------------|----------|
|spatial masked autoencoder| 0.1.0 | 64 | 2 | (128, 32, 32) | 1e-3 | cosine | 5 | 300 | 64 | 13–25% | Baseline — less compression to avoid uniform predictions.  Not great, clusters predictions around cloud masks. Edge issues crop up often max recall of 54%||
|spatial masked auto encoder| 0.2.0 | 32 | 3 | (128, 16, 16) | 1e-3 | cosine | 5 | 300 | 64 | 13–25% | Very noisy at 512 px validation. Poor convergence||
|spatial masked auto encoder (L1)| 0.1.0 | 64 | 2 | (128, 32, 32) | 1e-3 | cosine | 5 | 300 | 64 | 50-75% | ||

