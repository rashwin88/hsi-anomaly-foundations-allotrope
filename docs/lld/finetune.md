# Fine-tuning trainer and segmentation head

**Purpose** - continue training a pretrained spectral-stem model (F001) on EnMAP, PRISMA and AVIRIS-NG at their
own bands, and train a cloud / water / shadow / snow segmentation head on its encoder, in the stages of
`docs/proposals/fine-tuning-and-segmentation.md`. Runs: `research/experiments/indradhanu-finetune/`.

## Interfaces

```python
# app/foundation_models/components/spectral_stem_seg_former_mae.py
SpectralStemSegFormerMAE(..., seg_classes=0, seg_decoder_dim=128)
  .encode(x, wl, band_valid=None, keep_mask=None) -> [F1..F4]     # shared by both outputs
  .segment(x, wl, band_valid=None) -> (B, seg_classes, H, W)      # unmasked encoder + seg_head
  .extend_trained_wavelengths(wl, tol_nm=0.5) -> int              # grows the trained_wl buffer
  .trained_band_mask(wl, tol_nm=15) -> (N,) bool                  # reads trained_wl
  .forward_mixed([(x, wl, band_valid, keep_mask, out_wl), ...])     # one (B_g, M_g, H, W) per group
      -> [x_hat_g, ...]                                             # wavelength stems only
  .segment_mixed([(x, wl, band_valid), ...]) -> [logits_g, ...]     # joint + mixed batches (FT03 on)
# app/utils/segmentation/targets.py
CLASSES = ("water", "cloud", "thin_cloud", "shadow", "snow")
LabelPolicy(plateau_elevation_m=3500, low_water_vapour=750, cirrus_min_level=2)
enmap_targets(labels, meta, policy) -> (targets (5,H,W) float32, known (5,H,W) bool)
PSEUDO_CONF = 0.9
pseudo_targets(probs (5,H,W) uint8|float, refl, wl, valid, conf=PSEUDO_CONF) -> (targets, known)   # FT04
masked_bce_dice(logits, targets, known, dice_weight=1.0, class_weights=None) -> loss;  IoUTally().add(...).iou()
cloud_plausible(refl, wl) -> (H, W) bool          # label-QA cloud test; no caller since FT04 dropped the cloud guard
# app/foundation_models/trainers/finetune_trainer.py   (registered as spectral_stem_finetune)
FinetuneTrainer;  prepare_sample(sample, target_nm, labels, policy, implant, pseudo_dir);  keep_sample(sample, skip)
pseudo_sample_targets(sample, pseudo_dir) -> (targets, known)   # <pseudo_dir>/<key>.npy -> pseudo_targets; all unknown
                                                               # if the tile has no teacher file (test split)
implant_sample(sample, cfg)                    # FT05: implants into the pixels, clean.npy + implant.npy (3-px near mask)
single_thread_worker(worker_id)                # DataLoader worker_init_fn: one BLAS / torch thread per worker
gradient_l1(x_hat, x, band_valid, loss_mask) -> scalar   # edge term: |d x_hat - d x|, neighbour pairs both in mask
  ._mixed_loss(groups, model) -> (loss, kept)    # compute_loss routes a {"_groups": ...} batch here
_MixedLoader(loaders, shares, seed)   # one source per batch, drawn by share
_AllSourcesLoader(loaders)            # every source per batch: {"_groups": [(name, batch), ...]}
_SequentialLoader(loaders)            # validation: every test source in turn
# app/models/training/training_config.py
SpectralStemFinetuneConfig(sources=[FinetuneSource(name, train_dir, test_dir, share, band_list_path,
                           labels: "enmap" | "pseudo" | None, pseudo_dir, skip_scenes_path)],
                           init_from, mode, seg_*, label policy fields, seg_head_from,
                           finetune_metrics_csv, mixed_batches=False,   # mixed_batches: adapt or joint
                           grad_loss_weight=0.0,                        # edge term: mixed_batches only
                           seg_class_weights=None,                      # FT04: [1, 1, 1, 3, 3]
                           implant_prob=0.0, implant_max_objects=4, implant_near_weight=5.0,
                           implant_real_path, implant_psf_sigma=1.4, implant_sources, implant_seed=1,   # FT05
                           loader_workers_per_source=None)              # FT05 from epoch 6, FT04: 2
```

```bash
python scripts/pseudo_label.py <checkpoint> <shard dir> <out dir> [--band-list <json>] [--batch 32]
    # teacher pass: <out>/<tile key>.npy, (5, 128, 128) uint8 = round(255 x sigmoid); resumes (skips written tiles)
```

## Data flow

```
per source: local shards -> decode (.i16dz) -> keep_sample (skip scenes) -> prepare_sample
            (align onto the source's band list; EnMAP label_* + meta -> seg_targets, seg_known, or, for a
             "pseudo" source, the teacher's <key>.npy -> pseudo_targets -> seg_targets, seg_known; FT05:
             implant_sample; drop meta.json and label_*) -> DataLoader (worker_init_fn single_thread_worker,
             loader_workers_per_source or data.num_workers workers)
train, default: _MixedLoader draws one source per batch by share (endless; restarts an exhausted source)
train, mixed_batches (adapt / joint): _AllSourcesLoader - one DataLoader per source with batch size
       max(1, round(share x batch_size)); each step takes one sub-batch from every source
       (batch_size 8, shares 0.40 / 0.35 / 0.25 -> 3 / 3 / 2)
  -> _mixed_loss: per source, drop thin samples, own MAE mask, own band-augmentation draw
     -> forward_mixed: stem pass per source -> concatenate (8, D, H, W) -> BatchNorm -> encoder
        -> decoder -> split by source -> spectral_out at that source's output wavelengths
     -> L1 + ramped SAM (+ grad_loss_weight x gradient_l1, FT02c) per source, weighted by its kept samples;
        the edge term is logged as train_grad_l1 in finetune_metrics.csv, kept out of train L1
  probe   seg loss only, on labelled batches; backbone frozen, eval() so BatchNorm stats stay fixed
  adapt   SpectralStemMAETrainer.compute_loss (MAE + band augmentation)
  joint   MAE loss + seg_weight x seg loss on labelled batches (second, unmasked encoder pass; with
          mixed_batches, segment_mixed over every source that carries targets, masked_bce_dice per source with
          seg_class_weights, averaged over sources)
  FT05    implanted input, MAE loss against clean.npy; hidden pixels within 3 px of an implant weighted
          implant_near_weight in the loss mask
  scale   (FT06) uncertainty head only (scale_decoder_dim > 0), backbone frozen and eval(): Laplace NLL
          e/b + log b on hidden, valid, eroded pixels, e = mean |normalised residual| over valid bands (detached);
          no band augmentation; per-sensor val NLL and Spearman(b, e) -> finetune_metrics.csv. Failed detection
          (2026-10-20): b rises on targets themselves - research/experiments/indradhanu-finetune/FT06-uncertainty/
validation: every test source in turn; per-sensor L1/SAM and per-class IoU -> finetune_metrics.csv
            probe ranks epochs by seg loss, scale by its NLL, adapt/joint by the full-SAM reconstruction score
```

## Invariants

1. **The normalisation curve comes from `init_from`'s checkpoint**, never from data (frozen at pretraining).
2. **One band list per stem pass** (was "one band list per batch" until mixed batches, 2026-10-16). A source's
   samples are aligned onto its list. Without `mixed_batches` a batch is one source; with it, a batch holds every
   source but each source's sub-batch goes through the stem alone, and only the stem's D-channel outputs are
   concatenated. `forward_mixed` refuses the fixed-band stems (linear, gate).
3. **Unknown labels never enter the loss** - per class, over known pixels only; a class with no known pixel in a
   batch is left out of the average.
4. **Old checkpoints keep working:** no `trained_wl` -> record = the curve's wavelengths; no `seg_head` keys ->
   loaded with the head absent (pretrained) or fresh (fine-tuning).
5. **The record only grows**, and wavelengths within 0.5 nm of one already recorded (or of each other) count once.
6. **Pseudo-labels teach only what the teacher is sure of.** `pseudo_targets` sets known where p > 0.9 or p < 0.1
   (stored bytes >= 230 or <= 25) on pixels with >= 90% of bands finite, for water and cloud only (thin cloud,
   shadow, snow always unknown); a confident water with mean 800-900 nm reflectance > 0.15 becomes a known
   not-water (target 0, still known). Thresholds live in the loader, not the teacher's files, so they can change
   without re-running `pseudo_label.py`.
7. **Class weights form a weighted mean over the classes present**: `(w x per)[has].sum() / w[has].sum()`. A class
   with no known pixel in the batch drops out of numerator and denominator; on a pseudo-labelled sub-batch only
   water and cloud are present, so the rare-class weights act on EnMAP sub-batches only.

## Failure modes

| What | Seen as |
|---|---|
| `init_from` weights that do not match the model (other than the head) | `ValueError` listing missing / unexpected keys |
| a source folder with no `.tar` | `FileNotFoundError` naming the source |
| `segment` on a model without a head | `ValueError` |
| probe on a source without labels | its batches are skipped (no loss), so give probe labelled sources only |
| `mixed_batches` with mode probe | `ValueError` from the config validator (adapt and joint are allowed) |
| `implant_prob` without `mixed_batches` and `implant_real_path` | `ValueError` from the config validator |
| a "pseudo" source tile with no `<key>.npy` in `pseudo_dir` (or no `pseudo_dir`) | silent: the tile is all unknown, reconstruction only. Check the teacher pass covered the shards |
| loader workers with their own BLAS pools | not an error: on the g6e.2xlarge, 18 workers ran the implant transform at 4.7 s per tile instead of 1.8 s (FT05 2 h 41 min per epoch). `single_thread_worker` is always installed; keep `loader_workers_per_source` x sources at or under the vCPUs |
| `mixed_batches` on a source without `band_list_path` whose samples carry differing band lists | `ValueError` ("mixed batches need one band list per source") |
| `keep_mask` given for some groups of `forward_mixed` and not others | not checked: group 0 decides - if it has none, the others' masks are ignored; if it has one, `torch.cat` fails on the `None`. `_mixed_loss` always sets all |

## Findings that shaped the design

- **Per-sensor see-saw (FT02, 2026-10-16).** With one sensor per batch, per-sensor validation swung between
  epochs: epochs 11-12 and 15-17 PRISMA improved while EnMAP and AVIRIS-NG worsened (epoch 17: PRISMA SAM 5.04 deg,
  AVIRIS-NG 8.24 deg, against 5.93 / 6.07 at epoch 8), then reverted. Validation tiles are fixed and unshuffled,
  so the weights moved, not the data. (`research/experiments/indradhanu-finetune/FT02-adapt/README.md`)
- **The one BatchNorm.** The model has a single BatchNorm, `SpectralStemSegFormerMAE.norm` (`BatchNorm2d(D)`,
  D = 32, PyTorch default momentum 0.1), between the stem and the encoder. In training mode it normalises each
  micro-batch by that micro-batch's own statistics and folds them into its running statistics; validation and
  inference use the running ones. With sensor-pure micro-batches, training therefore always saw each sensor
  standardised to itself, while evaluation sees one running estimate dominated by the last ~20 micro-batches of
  the epoch (0.9^22 = 0.098) - whichever sensors those happened to be. Gradient accumulation (x4) mixes sensors
  in the gradient but not in these statistics, which are per micro-batch. Re-estimating the statistics on each
  PRISMA bench scene moved them 0.35-0.76 sd; residual Mahalanobis barely moved (0.853 -> 0.858), combined
  swung hard (set1 0.943 -> 0.755). Consistent with the see-saw; not shown to cause the gate miss.
- **Edges.** 49-68% of the top-1% residual Mahalanobis false positives sit on the scene's top-10% spatial edges,
  but so do 29-79% of ground-truth pixels; an edge-strength correction of the score map drops the bench
  0.86 -> 0.67. Edge handling, if any, belongs in training (FT02c), not in scoring.

## Decisions

- **One model, two heads** (agreed 2026-10-16): the head is a SegFormerDecoder (all-MLP fusion + pixel shuffle)
  with `out_channels = 5`, decoder width 128, ~372k parameters - reused, not new architecture.
- **Independent sigmoid classes, not softmax**: classes overlap (thin cloud over water).
- **Label policy from data** (EnMAP label QA, 2026-10-16): class 0 = cloud; dry scenes (wv < 0.75 g/cm2) drop cloud
  labels; snow wins; cirrus 1 unknown; duplicates skipped. Elevation is only a guard.
- **Mixing by share, not concatenation**, so AVIRIS-NG's ~200 tiles per flight line do not swamp the satellites.
- **Measured on the dev laptop (2026-10-16):** probe on one EnMAP shard, water IoU 0 -> 0.42 in 4 short epochs;
  joint mode runs end to end on EnMAP + SpectralEarth sources.
- **Start point F001 epoch 109** (2026-10-16): F001 was stopped at epoch 113 of 200 because the bench was flat
  (0.861 at epoch 60, 0.862 at 109); FT01 and FT02 load `F001_ep109.pt`.
- **Build on FT02 despite its gate miss** (user decision 2026-10-16): FT02 (0.855 vs 0.861) is not discarded;
  FT02b continues from its best.pt (epoch 8). `init_from` loads weights only, so FT02b's optimiser, cosine LR
  (back to 1e-4) and SAM ramp start again from epoch 0. The configs' `lr_schedule.warmup_epochs: 1` is never
  read (`_build_scheduler` builds a plain `CosineAnnealingLR`), so there is no warm-up, whatever the READMEs say.
- **Mixed-sensor batches** (user decision 2026-10-16, FT02b): fixes both the gradient balance per step and the
  BatchNorm statistics, with one code path. The alternatives FT02's README named were keeping BatchNorm in eval
  mode during adapt (statistics stay F001's, from SpectralEarth) and re-estimating it on a balanced mix before
  each validation (fixes what evaluation sees, not the per-sensor normalisation during training).
  At first adapt only; from FT03 joint mode mixes too (`segment_mixed`), with FT01's head as a warm start
  (`seg_head_from`).
- **Sub-batch sizes by rounding** (`max(1, round(share x batch_size))`): at batch 8 the effective shares are
  3/8, 3/8, 2/8 = 0.375 / 0.375 / 0.25, not 0.40 / 0.35 / 0.25 - PRISMA gains, EnMAP loses 2.5 points.
  Accepted; the sizes need not sum to batch_size for other shares or batch sizes.
- **Regularisation shelved** (user decision 2026-10-16): an EMA of the weights, SpectralEarth replay, a lower
  LR and L2-SP (a pull towards F001's weights) were all discussed as answers to the gate miss and shelved
  until mixed batches report; FT02's reconstruction improved on every sensor, so forgetting is not the
  leading explanation.
- **Teacher ("oracle") head deferred** (user decision 2026-10-16): cross-sensor distillation - an EMA teacher
  reading a sensor's full band list teaches the student reading a degraded list - discussed, deferred until
  after mixed batches.
- **Per-sensor heads and a synthetic-target detection head** (2026-10-16): discussed, not chosen. Trade-offs:
  per-sensor heads give up part of the one band-agnostic model; a synthetic-target head trains detection
  directly but on targets someone has to invent. Study notes: course part 54 (`indradhanu workings/`).
- **Score-map edge correction rejected** (2026-10-16): see Findings. A training-time edge term on hidden pixels
  (gradient matching, FT02c) is planned after FT02b.
- **Pseudo-labels for PRISMA and AVIRIS-NG (FT04, 2026-10-18)**: FT03 is the teacher (`scripts/pseudo_label.py`
  over the training shards; FT04 starts from FT03's own weights). Checked before launch on 150 EnMAP tiles against
  EnMAP's labels: guarded pseudo water IoU 0.875, the teacher's raw confident cloud 0.82. **No cloud guard:**
  `cloud_plausible` (bright and flat, from the label QA) rejected 69% of EnMAP's labelled clouds on a test shard,
  and set4's "11% cloud" it was aimed at is haze and a cloud / smoke streak, not bright desert (log, 2026-10-18).
  Result: IoU water / cloud 0.839 / 0.810 (FT03 0.841 / 0.815), SAM 3.88 / 4.11 / 5.46 deg (FT03 3.92 / 4.21 /
  5.48); the release candidate (research/experiments/indradhanu-finetune/FT04-final-joint/README.md).
- **Rare-class weights `[1, 1, 1, 3, 3]` (FT04)**: shadow IoU 0.205 -> 0.223; water / cloud held.
- **Implant-denoising (FT05, 2026-10-17/18)**: 50% of training tiles get 1-4 bench-v2 implants (`signature` and
  `material` sources; `host` held out), loss against the clean tile, x5 near implants. Did not pass: epoch 5 ties
  FT03 on the real bench, epoch 10 costs AVIRIS-NG SAM (5.67 vs 5.48). Half its implants had garbled signatures
  (descending-wavelength `np.interp` bug, fixed in b94dbfd after launch; docs/lld/implant-bench.md, "Corrections").
- **Single-thread loader workers, 2 per source (2026-10-18)**: FT05's loaders ran 18 processes each with its own
  BLAS thread pool on 8 vCPUs; one thread per worker and 2 workers per source took an epoch from 2 h 41 min to
  43 min. FT04 uses the same setting.
- **Not done:** oversampling for rare classes (weighting is done, FT04); a held-out snow set (EnMAP test has one
  snow tile); thin cloud, shadow and snow pseudo-labels (the teacher is not good enough at them to copy).
