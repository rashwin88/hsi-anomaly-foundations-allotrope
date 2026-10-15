# Spectral stem (SpectralEarth pretraining variants)

**Purpose** - one SegFormer MAE whose spectral stem can be swapped between today's 1x1
compressor (A), a per-pixel gate (B), cross-attention over wavelength-tagged band tokens (C) and
reduced attention over wavelength bins (C', the variant now carried forward), so the SpectralEarth
pretraining experiment changes the stem and nothing else. Design and decision rule:
`docs/proposals/spectral-attention-bottleneck.md`; runs: `research/experiments/indradhanu-se-pretrain/`;
taught in `indradhanu workings/31`-`42`.

## Interfaces

```python
# app/foundation_models/components/
wavelength_encoding(wavelengths_nm, dim=32) -> (..., dim)              # spectral_tokens.py
WavelengthStats(wl, mean, std).at(wl) / .normalize(x, wl) / .denormalize(x, wl)   # pixel_normalization.py
SpectralCrossAttentionCompressor(D=32, d=32, heads=4)(x, wl, band_valid) -> (B, D, H, W)   # spectral_attention.py
SpectralCrossAttentionDecoder(D=32, d=32, heads=4)(z, wl) -> (B, N, H, W)
SpectralGate(bands, hidden=32)(x) -> x * sigmoid(MLP(x))               # spectral_gate.py
SpectralReducedCompressor(D=32, d=32, heads=4, bin_nm=40, band_weighting="count")(x, wl, band_valid) -> (B, D, H, W)  # spectral_reduced.py
band_widths(wl, max_gap_nm=20) -> (N,) nm each band stands for                                 # spectral_reduced.py
WavelengthLinearDecoder(D=32)(z, wl) -> (B, N, H, W)
SpectralStemSegFormerMAE(stem, stats_wavelengths_nm, stats_mean, stats_std, bands=202, ...)
    (x, wavelengths_nm, band_valid=None, keep_mask=None, out_wavelengths_nm=None) -> (B, M, H, W) reflectance
    .trained_band_mask(wavelengths_nm, tol_nm=15) -> (N,) bool    # near a pretraining wavelength

# app/foundation_models/trainers/spectral_stem_mae_trainer.py
SpectralStemMAETrainer                    # registered for foundation_model_name "spectral_stem_mae"
decode_i16dz_member(key, data)            # webdataset decoder for compressed pixels (app/utils/shard_codec.py)
align_sample(sample, target_nm)           # loader-worker transform onto a band list (app/utils/band_alignment.py)
sample_band_subset(wl, min_bands=40)      # band-list augmentation: indices of a random subset
# app/abstract_classes/foundation_trainer.py - two hooks, identity/empty by default
FoundationTrainer._sample_decoders() -> list      # extra webdataset decoders, run in the workers
FoundationTrainer._sample_transform() -> callable # per-sample transform after decoding, in the workers
# app/foundation_models/pixel_stats.py
resolve_wavelength_stats(path) -> (wavelengths_nm, mean, std)
```

Config: `model_config.model_type = "spectral_stem_mae"` (`SpectralStemMAEConfig`, a subclass
of `HyperspectralSegFormerMAEConfig`) adds `stem`, `bands`, `token_dim`, `attention_heads`,
`gate_hidden`, `bin_nm` (reduced_attention), `band_weighting`, `band_augment_prob` and
`band_jitter_nm` (band-list robustness, below), `band_list_path` (align every sample onto the
wavelengths in that JSON, e.g. a shard set's stats.json), `metrics_csv`, and the run controls below (`amp_bf16`, `early_stop_patience`,
`early_stop_min_delta`, `early_stop_min_epochs`, `checkpoint_s3_uri`, `auto_resume`). `data.pixel_stats_path` must be a wavelength stats file (the
shards' `stats.json`). Example configs: `research/experiments/indradhanu-se-pretrain/S01-laptop-smoke/`.

## Data flow

```
shard sample  pixels.npy or pixels.i16dz (compressed, decoded in the workers) int16 (N,128,128),
              wavelengths.npy (N,); with band_list_path, aligned onto that list in the workers
  -> _decode_batch    reflectance = raw x 0.0001; band_valid = raw != -32768;
                      pixel mask = any band valid; identical wavelength rows collapse to (N,)
  -> model            WavelengthStats.normalize -> dead voxels := 0 -> stem in -> BatchNorm2d(32)
                      -> SegFormer encoder (token removal) / decoder -> stem out -> denormalize
  -> loss             L1 + lambda(t) * SAM on hidden x eroded pixels x valid bands
  -> validation score L1 + sam_weight * SAM (full weight, not the ramp) - ranks epochs for
                      early stop and best.pt; tagged VAL_SCORE = "l1+full_sam" in checkpoints
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

## The reduced stem (C')

Attention over fixed 40 nm wavelength bins from 400 nm (53 bins; 42 occupied by SpectralEarth's
202 bands) instead of per band, with three parts that keep band-level detail:
1. bin token = mean over the bin's valid bands of `v_b * k(lambda_b)` - a learned weight vector per
   band, so ~5 bands map into 32 numbers one-to-one (a shared weight would keep only the bin sum);
2. a per-band skip path `mean_b(v_b * W(lambda_b))` straight to the 32 channels;
3. a per-band decoder `U(lambda_n) . z + c(lambda_n)`, no attention.
`k`, `W`, `U` are small networks (Linear 32->64, GELU, Linear) of the wavelength encoding.
Everything is matrix products over the band axis; nothing of shape (pixels x bands x 32) is kept.
Takes **one band list per batch** (`(N,)` wavelengths) and refuses per-sample lists - use
`band_list_path` when samples differ (EnMAP's 5 differently-numbered scenes, mixed sensors).

Measured 2026-10-14 on the L40S (one training step, bf16, real encoder, patches/s and GiB peak):

| stem | b4 | b8 | b16 | b32 |
|---|---|---|---|---|
| linear | 140.8 / 0.4 | 270.6 / 0.8 | 506.7 / 1.4 | 780.5 / 2.8 |
| gate | 138.1 / 0.5 | 260.2 / 1.0 | 449.3 / 1.9 | 612.2 / 3.7 |
| cross_attention | 13.5 / 11.3 | 12.2 / 18.3 | 10.2 / 32.4 | OOM |
| reduced_attention | 70.6 / 3.8 | **80.0 / 5.4** | 69.9 / 8.5 | 48.6 / 14.6 |

~2.6x fewer multiply-adds than per-band attention, ~6x faster: the gain is mostly memory traffic
(not storing and re-reading the per-band activations). In a full epoch (E004, batch 8, 6 workers,
two-pass validation) it runs at ~55 patches/s, 9.2 min per 30,000 patches.

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
6. **The normalisation curve is fixed at pretraining and never recomputed** (decided 2026-10-14).
   It is measured once from the pretraining shards' `stats.json` and then lives only in the
   checkpoint's `stats.*` buffers. Fine-tuning (EnMAP, PRISMA) and inference keep that curve; the
   EnMAP shards' own `stats.json` is a record of the data, not an input. Changing the curve would
   shift every input the pretrained weights learned to read. No per-sensor or per-band stats
   files for these models.

## Band-list robustness (adopted 2026-10-14, verified by E005)

E004 read SpectralEarth's 202 bands well and PRISMA's native 185 badly (set1 ROC-AUC 0.205 native,
0.725 resampled): its wavelength networks `k`, `W`, `U` had only ever been evaluated at one band list,
and the skip path's plain `1/N` mean gave every band one vote, so a sensor's band density changed the
mix (thinning SpectralEarth's own bands to PRISMA spacing doubled the water error). Measurements in
`research/experiments/indradhanu-se-pretrain/E004-reduced-attention/README.md`; taught in course part 42.
Three fixes, each off by default so E001-E004 checkpoints behave exactly as trained:

1. **`band_weighting: "width"`** - every band counts by the nm it covers (`band_widths`: half the gap to
   each neighbour, gaps capped at 20 nm so a band beside a water-vapour gap does not claim it), in the
   in-bin mean and the skip path. Averages become integrals over wavelength; on an evenly spaced list
   the result equals `"count"` exactly. No parameters. The inferencer builds the model with the
   checkpoint's own setting.
2. **`band_augment_prob`** (trainer) - that share of training batches reads a random subset of its bands
   (`sample_band_subset`: thinning to 50-95%, 1-3 gaps of 20-150 nm, or the bands nearest an even 6-15 nm
   grid; never below 40 bands) and reconstructs **every** band via `out_wavelengths_nm`; the loss covers
   all of them. Validation always reads the full list, so it stays comparable across runs.
3. **`band_jitter_nm`** - on augmented batches each input band's *stated* wavelength moves by up to
   +/- this much (values unchanged), so `k`, `W` are smooth between trained wavelengths - what a denser
   sensor (AVIRIS-NG, ~5 nm) needs, since thinning can only make lists sparser than SpectralEarth.

Plus a rule at load time: **`trained_band_mask`** keeps bands within 15 nm of a pretraining wavelength
(the checkpoint's normalisation curve); `scripts/evaluate_prisma_bench.py` drops the rest for spectral
stems (set1 native: 185 -> 165, the 20 inside SpectralEarth's water-vapour gaps and past 2445 nm).

**Invariant - augmentation never trains self-completion.** The loss is taken only on pixels hidden by
the token mask (`loss_mask` comes from the prediction mask), and a hidden pixel's own bands are never
an input; it is rebuilt from its neighbours' band subsets. Scoring never augments. This is what keeps
design decision 7 ("hide pixels, never bands while scoring") intact. If the loss is ever extended to
visible pixels, augmentation must be switched off for it.

Gotchas: one band list per batch (augmentation draws one subset per batch); augmentation requires a
wavelength stem (`ValueError` at `build_model` otherwise); per-sample wavelength lists `(B, N)` are
never augmented.

## Failure modes

| What | Seen as |
|---|---|
| linear or gate stem given another band count | `ValueError` naming both counts |
| band-position stats file (app/constants/*.json) as `pixel_stats_path` | `ValueError: no wavelengths_nm` |
| more than 65,535 pixel rows in one fused-attention call on CUDA | `invalid configuration argument` - prevented by chunking at `SDPA_MAX_ROWS = 32,768` |
| C at too large a batch | CUDA OOM; on Windows, silent spill to shared memory and a ~10x slowdown first |
| early stop / auto-resume / S3 upload with `save_every_n_epochs` != 1 | config `ValueError` at load |
| reduced stem given per-sample wavelengths `(B, N)` | `ValueError` - align with `band_list_path` |
| `band_list_path` target band missing from a sample | that band becomes nodata (masked), not zero |
| resume from a checkpoint scored the old way (ramped SAM) | early-stop state ignored, logged; weights/optimizer/scheduler still restored |
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
- **Per-band cross-attention (C) superseded by reduced attention (C') on 2026-10-14**: E003 ran 3
  epochs at 13.5 patches/s (41 min/epoch); C' runs ~6x faster per step at batch 8. Accepted that
  attention now looks at bins - detail is kept by fixes 1-3, not by attention resolution.
- **Validation at full SAM weight.** The ramped weight (0 -> 0.5 over 10 epochs) made epoch 1
  unbeatable, so early stop would have fired at the first allowed epoch whatever the model did
  (found live in E003; the same flaw cost the October H200 trial its final checkpoint).
- **Alignment by wavelength in the workers, not resampling.** A band a sample lacks becomes nodata,
  which every wavelength stem already masks exactly. Pre-training batches are single-sensor
  (SpectralEarth); mixing is for fine-tuning only, and the label-key policy for mixed batches is open.
- **One inference method: native bands** (decided 2026-10-15). A wavelength-stem model reads each
  sensor's own band list, minus bands `trained_band_mask` rejects - never a resampled grid.
  `scripts/evaluate_prisma_bench.py` defaults to it; resampling to SpectralEarth's 202 (`se202`) stays
  only for the fixed-band stems, which cannot read anything else, and for diagnosis.
- **Rejected:** band masking for pretraining (spectral self-completion leak) - training a pixel to
  fill its *own* missing bands, which lets an anomaly rebuild itself. Band-list augmentation (below)
  is not that: see its invariant.
