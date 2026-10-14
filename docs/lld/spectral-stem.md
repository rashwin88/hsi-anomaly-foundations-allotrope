# Spectral stem (SpectralEarth pretraining variants)

**Purpose** - one SegFormer MAE whose spectral stem can be swapped between today's 1x1
compressor (A), a per-pixel gate (B) and cross-attention over wavelength-tagged band tokens
(C), so the SpectralEarth pretraining experiment changes the stem and nothing else. Design and
decision rule: `docs/proposals/spectral-attention-bottleneck.md`; runs:
`research/experiments/indradhanu-se-pretrain/`; taught in `indradhanu workings/31`-`39`.

## Interfaces

```python
# app/foundation_models/components/
wavelength_encoding(wavelengths_nm, dim=32) -> (..., dim)              # spectral_tokens.py
WavelengthStats(wl, mean, std).at(wl) / .normalize(x, wl) / .denormalize(x, wl)   # pixel_normalization.py
SpectralCrossAttentionCompressor(D=32, d=32, heads=4)(x, wl, band_valid) -> (B, D, H, W)   # spectral_attention.py
SpectralCrossAttentionDecoder(D=32, d=32, heads=4)(z, wl) -> (B, N, H, W)
SpectralGate(bands, hidden=32)(x) -> x * sigmoid(MLP(x))               # spectral_gate.py
SpectralStemSegFormerMAE(stem, stats_wavelengths_nm, stats_mean, stats_std, bands=202, ...)
    (x, wavelengths_nm, band_valid=None, keep_mask=None) -> (B, N, H, W) reflectance

# app/foundation_models/trainers/spectral_stem_mae_trainer.py
SpectralStemMAETrainer                    # registered for foundation_model_name "spectral_stem_mae"
# app/foundation_models/pixel_stats.py
resolve_wavelength_stats(path) -> (wavelengths_nm, mean, std)
```

Config: `model_config.model_type = "spectral_stem_mae"` (`SpectralStemMAEConfig`, a subclass
of `HyperspectralSegFormerMAEConfig`) adds `stem`, `bands`, `token_dim`, `attention_heads`,
`gate_hidden`, `metrics_csv`, and the run controls below (`amp_bf16`, `early_stop_patience`,
`early_stop_min_delta`, `early_stop_min_epochs`, `checkpoint_s3_uri`, `auto_resume`). `data.pixel_stats_path` must be a wavelength stats file (the
shards' `stats.json`). Example configs: `research/experiments/indradhanu-se-pretrain/S01-laptop-smoke/`.

## Data flow

```
shard sample  pixels.npy int16 (202,128,128), wavelengths.npy (202,)
  -> _decode_batch    reflectance = raw x 0.0001; band_valid = raw != -32768;
                      pixel mask = any band valid; identical wavelength rows collapse to (N,)
  -> model            WavelengthStats.normalize -> dead voxels := 0 -> stem in -> BatchNorm2d(32)
                      -> SegFormer encoder (token removal) / decoder -> stem out -> denormalize
  -> loss             L1 + lambda(t) * SAM on hidden x eroded pixels x valid bands
  -> metrics.csv      one row per epoch (train/val L1 and SAM deg, lr, patches/s, peak GPU GB)
```

## Run control (for long cloud runs)

- **bf16** (`amp_bf16`): the model runs under autocast on CUDA; the loss is computed in fp32.
- **Early stopping**: an epoch counts as an improvement only if avg val loss beats the best by
  `early_stop_min_delta` (relative, default 1%); smaller gains do not reset the count, so a slow
  creep still stops. Stops after `early_stop_patience` flat epochs, never before
  `early_stop_min_epochs` (default 20, past the SAM ramp). Agreed rule for E001-E003: 1% / 10
  epochs / minimum 20 / ceiling 200.
- **Checkpoints**: besides the base trainer's top-K `..._epochN.pt`, every epoch writes
  `last.pt` (write-then-rename) and, on a new lowest val loss, `best.pt`; both carry
  `early_stop` state and are uploaded to `checkpoint_s3_uri` when set. Top-K cleanup never
  touches them.
- **Resume**: `auto_resume` loads `checkpoint_dir/last.pt`, or downloads `last.pt` from
  `checkpoint_s3_uri` if the local disk is fresh; early-stop state is restored with weights,
  optimizer and scheduler. `metrics.csv` is appended to, so it continues across a resume.
- All three need `checkpoint.save_every_n_epochs = 1` (validated), because the early-stop
  decision and the uploads happen at checkpoint time.

Verified on the laptop GPU (`S01-laptop-smoke/resume_test*.py`): stops at the right epoch;
a run killed after epoch 2 and resumed from S3 reproduces the uninterrupted run's lr
sequence exactly.

## Invariants

1. **Only the stem differs between variants.** Normalisation, the BatchNorm, encoder, decoder,
   masking and loss are shared code. Pinned by `tests/test_foundation_models/
   test_spectral_stem_model.py` (shared parameters identical across stems).
2. **A dead voxel's value reaches no stem.** Fixed-band stems get 0 after normalisation; the
   cross-attention compressor masks it with `-inf`. A pixel with no valid band outputs exactly 0.
3. **A dead band never enters the loss** - zeroed in prediction and target before SAM, and
   excluded from the per-pixel L1 mean.
4. **Everything spectral is per pixel.** Nothing moves information between pixels except the
   unchanged SegFormer, so a hidden pixel cannot reconstruct itself (course part 38, trap 1).
5. **Parameter counts match the course**: compressor 5,345, decoder 6,401, gate 13,162 (202
   bands) / 10,757 (165). Tests pin them; a change must update the course.

## Failure modes

| What | Seen as |
|---|---|
| linear or gate stem given another band count | `ValueError` naming both counts |
| band-position stats file (app/constants/*.json) as `pixel_stats_path` | `ValueError: no wavelengths_nm` |
| more than 65,535 pixel rows in one fused-attention call on CUDA | `invalid configuration argument` - prevented by chunking at `SDPA_MAX_ROWS = 32,768` |
| C at too large a batch | CUDA OOM; on Windows, silent spill to shared memory and a ~10x slowdown first |
| early stop / auto-resume / S3 upload with `save_every_n_epochs` != 1 | config `ValueError` at load |
| no AWS credentials on the training host with `checkpoint_s3_uri` set | `NoCredentialsError` at the first checkpoint - the run dies at epoch 1, not hours in |

## Decisions

- **New classes, not a modified Indradhanu.** `HyperspectralSegFormerMAE` and its trainer are
  untouched, so shipped checkpoints keep loading.
- **No BatchNorm inside the cross-attention compressor**; the model adds one BatchNorm2d for
  every stem (design decision 8).
- **One `WavelengthStats` module** holding the curve once, with normalize and denormalize.
- **Pixel-row chunking for attention** - exact, because pixels are independent.
- **Cost, measured** (S01, RTX 4000 Ada 20 GB, real encoder size): C ~3.7 GiB per 128 x 128
  patch in bf16 and ~5 patches/s, against ~391 (A) and ~282 (B). Accepted on 2026-10-14 in
  favour of training on g6e (L40S 48 GB). Measured and set aside: 40 nm wavelength bins for the
  compressor (~2x; the per-band decoder is the other half of the cost - S01 section 3). Not
  tried: `d = 16`, recomputing the stem in the backward pass. These are the levers if cost
  becomes the limit.
- **Rejected:** band masking for pretraining (spectral self-completion leak).
