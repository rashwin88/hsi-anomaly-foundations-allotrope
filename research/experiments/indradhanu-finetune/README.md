# Indradhanu fine-tuning: EnMAP + PRISMA + AVIRIS-NG, with a segmentation head

**Question:** starting from F001 (the SpectralEarth-pretrained reduced-attention model), does fine-tuning on the
three real sensors, with a cloud / water / shadow / snow segmentation head, improve anomaly detection on the
labelled PRISMA scenes, and give masks good enough to replace the threshold rules in `scene_segmentation`?

**Design, curriculum and gates:** `docs/proposals/fine-tuning-and-segmentation.md`. Pass criteria for each run
are written in its README before launch and not moved afterwards.

## Runs

| id | stage | mode | what | instance | hours | cost | gate result | verdict |
|---|---|---|---|---|---|---|---|---|
| FT01 | 1 | probe | segmentation head on frozen F001, EnMAP | g6e.2xlarge (kept) | - | - | - | not started |
| FT02 | 2 | adapt | reconstruction on all three sensors | g6e.2xlarge (kept) | - | - | - | not started |
| FT03 | 3 | joint | reconstruction + segmentation | g6e.2xlarge (kept) | - | - | - | not started |

## Stage 0 records

**Shards** (train / test tiles): EnMAP `patches/enmap/native-v1z` 14,542 / 950 (224 bands, labels); PRISMA
`patches/prisma/native-v1z` 14,802 / 1,570 (234 bands); AVIRIS-NG `patches/aviris_ng/native-v1` 22,377 / 2,270
(372 bands, 3-8 m). The `v1z` sets are the native-v1 shards re-encoded to `pixels.i16dz` (exact, ~2.1x;
`recompress_ft.sh`) so all three fit on the L40S. Band lists for alignment: `band_lists/enmap_224.json`,
`band_lists/aviris_ng_372.json`; PRISMA's 234 centres are fixed and need none.

**EnMAP label QA** (2026-10-16, all 104 shards / 15,492 tiles / 210 scenes; per-scene CSV and scripts kept on the
dev laptop at `C:/es/labelqa/`, per-tile tables at `patches/enmap/native-v1-analysis/`). No ground truth exists,
so two plausibility tests were used: *cloud plausible* = mean 450-650 nm reflectance > 0.2 and flat (min/max of
the 460-500 / 540-580 / 640-680 / 840-880 nm means >= 0.7); *cirrus surface-driven* = within a tile, the cirrus
pixels brighten 2100-2250 nm more than 450-650 nm relative to clear land (real thin cloud brightens the dark
visible more).

| scene water vapour (g/cm2) | scenes | cirrus level 1 surface-driven | cirrus >= 2 surface-driven |
|---|---|---|---|
| < 0.5 | 3 | 100% | 100% |
| 0.5-0.75 | 8 | 90% | 100% |
| 0.75-1.0 | 20 | 79% | 42% |
| 1.0-1.25 | 33 | 26% | 10% |
| 1.5-2.0 | 60 | 14% | 8% |
| 2.0-2.5 | 33 | 66% | 32% |
| >= 2.5 | 32 | 42% | 3% |

| policy (train) | scenes | tiles affected | suspect labels removed | plausible removed |
|---|---|---|---|---|
| wv < 0.50 | 3 | 1.5% | 44% | 0.2% |
| **wv < 0.75 (adopted)** | **11** | **5.6%** | **62.6%** | **13.6%** |
| wv < 1.0 | 30 | 15.2% | 66.6% | 15.7% |
| elevation > 1,500 m | 13 | 6.6% | 43.1% | 18.8% |
| elevation > 3,500 m (proposal's first draft) | 0 | 0 | 0 | 0 |

Findings that changed the draft: `label_classes == 0` is EnMAP's cloud class (79% of all cloud-labelled pixels),
not no-data; no scene is above 2,776 m, so dry air, not altitude, marks the false labels; 4 scenes appear twice
under two processor versions (`band_lists/enmap_drop_duplicate_scenes.json`); snow and cloud overlap on 122,815
pixels; 53% of shadow pixels have no cloud within 900 m in the same tile (an upper bound - the cloud may be in a
neighbouring tile). Class balance after the policy and the remaining caveats are in the proposal.
