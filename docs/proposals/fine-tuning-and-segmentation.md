# Fine-tuning Indradhanu with a segmentation head

**Status:** agreed 2026-10-16 (classes, shared model, L40S); stage 0 in progress; label policy fixed 2026-10-16.
**Starts from:** F001, the SpectralEarth-pretrained reduced-attention model
(`research/experiments/indradhanu-se-pretrain/F001-final-pretrain/`), finishing 2026-10-16.
**Runs in:** `research/experiments/indradhanu-finetune/` (FT01, FT02, ... one folder per run).

## Why

1. **Real sensors.** F001 has only seen SpectralEarth (EnMAP L2A). The product scores PRISMA, EnMAP and
   AVIRIS-NG at their own bands and pixel sizes; fine-tuning on all three closes that gap and teaches the
   wavelengths SpectralEarth dropped (incl. the ~1730 nm hydrocarbon band).
2. **A learned scene segmentation**, replacing the spectral-index thresholds of the `scene_segmentation`
   action, so clouds, cloud edges, shadows and snow stop being scored as anomalies.
3. **Class-aware scoring.** Residual Mahalanobis (the scoring default since 2026-10-15) estimates "normal
   error" over the whole scene; estimated per surface class instead, water gets its own normal and the
   dark-water problem is solved by design.

The head is judged by detection: every stage is measured on the four labelled PRISMA scenes with the default
score, and nothing ships that lowers it.

## The model

One model, two outputs: F001's band-agnostic stem and SegFormer encoder, shared; the existing reconstruction
decoder; a new segmentation head on the encoder's four stage features (SegFormer all-MLP style, 1x1 projections
to a common width, upsample to 1/4, fuse, 3x3 refine, PixelShuffle to full size - the existing
`SegFormerDecoder` reused with `out_channels = 5` and width 128, ~372k parameters). Reconstruction runs the
encoder with MAE token removal; segmentation runs it unmasked - a second forward pass, on labelled batches only.

**Outputs - one logit per class, independent (sigmoid), because classes overlap (thin cloud over water):**

| class | from EnMAP label | known where |
|---|---|---|
| water | `label_classes == 2` (land = 1) | not under cloud or cirrus >= 2 |
| cloud | `label_cloud == 1`, **including every `label_classes == 0` pixel** (EnMAP's cloud class) | not where snow is set; not on dry scenes |
| thin cloud / haze | `label_cirrus >= 2` or `label_haze == 1` | not on cirrus-1 pixels; not on dry scenes |
| cloud shadow | `label_cloud_shadow == 1` | |
| snow | `label_snow == 1` | snow wins where cloud is also set |

`label_classes == 3` is provider no-data: unknown for every class. At class-0 (cloud) pixels only the cloud
class is known.

Land is the complement of water where labelled. Loss: per-class binary cross-entropy + soft Dice, over labelled
pixels only (a per-class ignore mask), so a tile without, say, trustworthy cloud labels still teaches water.

## Data

| sensor | shards | train / test tiles | labels | pixel size |
|---|---|---|---|---|
| EnMAP | `patches/enmap/native-v1` | 14,542 / 950 | all five classes | 30 m |
| PRISMA | `patches/prisma/native-v1` | 14,802 / 1,570 | none | 30 m |
| AVIRIS-NG | `patches/aviris_ng/native-v1` | 22,377 / 2,270 | none | 3-8 m |

- **One sensor per batch** (the stem takes one band list per batch). Each sensor's samples are aligned onto
  that sensor's own band list in the loader (`band_alignment`); PRISMA needs none, AVIRIS-NG aligns its three
  product-version variants (all within 0.42 nm), EnMAP aligns its renumbered scenes.
- **Per-epoch shares, not concatenation:** start at EnMAP 40% / PRISMA 35% / AVIRIS-NG 25% of batches, so
  AVIRIS-NG's 22k tiles (about 200 per flight line) do not swamp the satellites. Tuned only on the stage gates.
- **Label policy, applied at load time** (no reshard), fixed from a QA of all 15,492 EnMAP tiles (2026-10-16;
  tables in `research/experiments/indradhanu-finetune/README.md`):
  - **Dry scenes** - `scene_wv` below 0.75 g/cm2 (stored x1000: 750): cloud and thin-cloud labels become
    "no label". Below that, 90-100% of cirrus labels are surface-driven (dry ground through thin air); the rule
    removes 63% of suspect cloud/cirrus labels for 14% of plausible ones, touching 11 train scenes and no test
    scene. Elevation >= 3,500 m is kept as a guard only: no native-v1 scene exceeds 2,776 m.
  - **Overlaps:** snow wins over cloud (122,815 shared pixels); water is unknown under cloud or cirrus >= 2;
    cirrus level 1 is unknown for thin cloud rather than negative.
  - **Duplicates:** 4 scenes were processed twice; the older copies (576 tiles) are skipped
    (`band_lists/enmap_drop_duplicate_scenes.json`).
  - **Class balance after the policy** (train, share of labelled pixels / tiles with >= 1%): water 6.9% / 4,315;
    cloud 0.66% / 629; thin cloud 1.97% / 1,679; shadow 0.09% / 245; snow 0.09% / 81. Rare classes will need
    positive weighting or oversampling; **snow cannot be measured on EnMAP's test split (1 tile)** and needs a
    separate held-out check.
- **Storage on the L40S** (412 GB NVMe): raw shards total ~351 GB. EnMAP and PRISMA are recompressed to
  `pixels.i16dz` (exact; measured ~2.1x on these shards: EnMAP 51.4 GiB, PRISMA 53.9 GiB; `scripts/recompress_shards.py`)
  on the prod box first, bringing the set to 232.5 GiB with AVIRIS-NG's 127.2 GiB - all on local disk, no rotation needed. Done 2026-10-16.

## Curriculum

| stage | trains | data | gate to pass before the next |
|---|---|---|---|
| **0. Setup** | nothing | - | everything below exists and is tested; label thresholds fixed |
| **1. Probe** (FT01) | the segmentation head only; F001 frozen | EnMAP | per-class IoU on EnMAP's held-out datatakes recorded; tells how much F001's features already separate cloud / water |
| **2. Adapt** (FT02) | stem + encoder + reconstruction decoder, MAE loss, low LR, band augmentation on | all three, shares above | PRISMA bench (residual Mahalanobis) >= F001's (0.861 at epoch 60); EnMAP / PRISMA / AVIRIS-NG test reconstruction falls |
| **3. Joint** (FT03) | everything from the start, MAE + segmentation loss (MAE dominant); gradual top-down unfreezing is not implemented - an option if joint training destabilises reconstruction | all three; segmentation loss on EnMAP batches only | bench not below stage 2; IoU above stage 1 |
| **4. Pseudo-labels** (optional, FT04) | as stage 3 | + PRISMA / AVIRIS-NG pixels the stage-3 head labels with high confidence | IoU on EnMAP holds; visual check on PRISMA cloud scenes; bench holds |
| **5. Product** | - | - | masks replace the threshold rules in `scene_segmentation`; class-aware residual Mahalanobis offered in scoring, shipped only if it beats the default on the bench |

Pass criteria for each run are written in its README before launch, as for E005.

## Stage 0 - what has to exist first

1. **Trained-wavelength record:** a checkpoint buffer of the wavelengths the model has trained on, extended by
   each stage; `trained_band_mask` reads it (falls back to the normalisation curve for older checkpoints). The
   normalisation curve stays frozen (decided 2026-10-14).
2. **Label QA and policy:** label statistics per EnMAP scene against elevation and water vapour, thresholds
   chosen and documented; a loader-time label decoder producing (classes, H, W) targets + ignore masks.
3. **Multi-sensor loader:** one dataset per sensor, batches drawn by share, one band list per batch; reuses the
   hot-storage, decoding (`.i16dz`) and band-alignment hooks.
4. **Segmentation head + losses + metrics** (`app/foundation_models/components/`), with a config switch to freeze
   the backbone (stage 1) or train jointly (stage 3).
5. **Evaluation:** per-class IoU on EnMAP test datatakes each epoch; the PRISMA bench (default score) on the
   laptop at checkpoints; later the bench with segmentation masks and class-aware scoring.
6. **Infrastructure:** GPU role read access to the three shard sets; EnMAP and PRISMA recompressed; a staging
   script for the L40S once F001 frees the disk.
7. **Course:** new parts for the segmentation head, multi-task training and class-aware scoring.

## Open questions (decide as stages report)

- Whether class-aware Mahalanobis uses hard classes or soft (probability-weighted) statistics.
- Whether pseudo-labels (stage 4) are needed at all, or EnMAP-trained masks transfer well enough.
- Per-sensor shares after stage 2; whether AVIRIS-NG needs its own scale handling (3-8 m vs 30 m).
- Fine-tuning epochs and the LR schedule per stage (set per run, written before launch).
