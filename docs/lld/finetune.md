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
# app/utils/segmentation/targets.py
CLASSES = ("water", "cloud", "thin_cloud", "shadow", "snow")
LabelPolicy(plateau_elevation_m=3500, low_water_vapour=750, cirrus_min_level=2)
enmap_targets(labels, meta, policy) -> (targets (5,H,W) float32, known (5,H,W) bool)
masked_bce_dice(logits, targets, known) -> loss;  IoUTally().add(...).iou()
# app/foundation_models/trainers/finetune_trainer.py   (registered as spectral_stem_finetune)
FinetuneTrainer;  prepare_sample(sample, target_nm, labels, policy);  keep_sample(sample, skip)
# app/models/training/training_config.py
SpectralStemFinetuneConfig(sources=[FinetuneSource(name, train_dir, test_dir, share, band_list_path,
                           labels, skip_scenes_path)], init_from, mode, seg_*, label policy fields,
                           finetune_metrics_csv)
```

## Data flow

```
per source: local shards -> decode (.i16dz) -> keep_sample (skip scenes) -> prepare_sample
            (align onto the source's band list; EnMAP label_* + meta -> seg_targets, seg_known; drop
             meta.json and label_*) -> DataLoader
train: _MixedLoader draws one source per batch by share (endless; restarts an exhausted source)
  probe   seg loss only, on labelled batches; backbone frozen, eval() so BatchNorm stats stay fixed
  adapt   SpectralStemMAETrainer.compute_loss (MAE + band augmentation)
  joint   MAE loss + seg_weight x seg loss on labelled batches (second, unmasked encoder pass)
validation: every test source in turn; per-sensor L1/SAM and per-class IoU -> finetune_metrics.csv
            probe ranks epochs by seg loss, adapt/joint by the full-SAM reconstruction score
```

## Invariants

1. **The normalisation curve comes from `init_from`'s checkpoint**, never from data (frozen at pretraining).
2. **One band list per batch.** A source's samples are aligned onto its list; batches never mix sources.
3. **Unknown labels never enter the loss** - per class, over known pixels only; a class with no known pixel in a
   batch is left out of the average.
4. **Old checkpoints keep working:** no `trained_wl` -> record = the curve's wavelengths; no `seg_head` keys ->
   loaded with the head absent (pretrained) or fresh (fine-tuning).
5. **The record only grows**, and wavelengths within 0.5 nm of one already recorded (or of each other) count once.

## Failure modes

| What | Seen as |
|---|---|
| `init_from` weights that do not match the model (other than the head) | `ValueError` listing missing / unexpected keys |
| a source folder with no `.tar` | `FileNotFoundError` naming the source |
| `segment` on a model without a head | `ValueError` |
| probe on a source without labels | its batches are skipped (no loss), so give probe labelled sources only |

## Decisions

- **One model, two heads** (agreed 2026-10-16): the head is a SegFormerDecoder (all-MLP fusion + pixel shuffle)
  with `out_channels = 5`, decoder width 128, ~372k parameters - reused, not new architecture.
- **Independent sigmoid classes, not softmax**: classes overlap (thin cloud over water).
- **Label policy from data** (EnMAP label QA, 2026-10-16): class 0 = cloud; dry scenes (wv < 0.75 g/cm2) drop cloud
  labels; snow wins; cirrus 1 unknown; duplicates skipped. Elevation is only a guard.
- **Mixing by share, not concatenation**, so AVIRIS-NG's ~200 tiles per flight line do not swamp the satellites.
- **Measured on the dev laptop (2026-10-16):** probe on one EnMAP shard, water IoU 0 -> 0.42 in 4 short epochs;
  joint mode runs end to end on EnMAP + SpectralEarth sources.
- **Not done:** class weighting / oversampling for rare classes (shadow, snow) - decided after FT01's IoUs; a
  held-out snow set (EnMAP test has one snow tile).
