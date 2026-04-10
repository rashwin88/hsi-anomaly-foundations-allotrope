# Hyperspectral Patch Generation Pipeline

The hyperspectral patch pipeline converts raw PRISMA and EnMAP satellite scenes into shuffled, ML-ready webdataset shards stored on S3. The pipeline produces mixed-sensor final shards where PRISMA and EnMAP patches are interleaved, enabling sensor-agnostic training.

## Pipeline Overview

```
S3: prisma/*.he5  +  enmap/*/...
    │
    ▼
Stage 1: Intermediate Sharding (per sensor)
    ├── Download scene from S3
    ├── Build vendable with BandFilterConfig:
    │     [1] Bad band removal
    │     [2] Atmospheric window exclusion
    │     [3] Detector edge trimming
    │     [4] Coverage-aware band pruning (224→188 bands)
    │     [5] Quality mask invalidation (EnMAP: cloud, shadow, haze)
    │     [6] Spatial masking (>40% invalid → full pixel invalidation)
    │     [7] Spectral interpolation (PCHIP/linear gap-fill)
    │     [8] Resampling to common 165-band grid (10nm, 460-2450nm)
    ├── Patch into 128×128 tiles with stride 64 (50% overlap)
    ├── Filter: discard patches with <50% valid pixels
    ├── Write to webdataset .tar shards (1 GB target)
    ├── Upload shard to S3, delete local copy
    └── Cleanup downloaded scene files
    │
    ▼
S3: patches/prisma/{split}/intermediate/w128_h128_s64/
    patches/enmap/{split}/intermediate/w128_h128_s64/
    │
    ▼
Stage 2: Final Shuffling (mixed sensors)
    ├── Read intermediate shards from BOTH sensors via S3 pipe
    ├── Shuffle patches across all scenes and sensors
    ├── Write mixed final shards (1 GB target)
    └── Upload to S3
    │
    ▼
S3: patches/hyperspectral/{split}/final/w128_h128_s64/
```

## S3 Storage Layout

```
s3://allotrope-raw-data-india/
├── prisma/                                                # raw .he5 scenes
├── enmap/                                                 # raw scene folders (COG .tiff + .XML)
├── patches/prisma/train/intermediate/w128_h128_s64/       # PRISMA-only intermediate
├── patches/prisma/test/intermediate/w128_h128_s64/
├── patches/enmap/train/intermediate/w128_h128_s64/        # EnMAP-only intermediate
├── patches/enmap/test/intermediate/w128_h128_s64/
├── patches/hyperspectral/train/final/w128_h128_s64/       # MIXED final (PRISMA + EnMAP)
└── patches/hyperspectral/test/final/w128_h128_s64/        # MIXED final (test split)
```

## Webdataset Sample Format

Each patch in the shard contains:

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `pixels.npy` | (165, 128, 128) | float32 | Surface reflectance on common grid |
| `validity_cube.npy` | (165, 128, 128) | int8 | Binary validity (all bands agree per pixel) |
| `wavelengths.npy` | (165,) | float64 | Center wavelengths, ascending, identical across sensors |
| `meta.json` | dict | — | scene_id, row/col coords, sensor, spectral families, band count |

**Per-patch size:** ~13 MB. **Patches per 1 GB shard:** ~80.

## Scene Selection and Splitting

1. All scenes discovered from S3 (sorted for determinism)
2. Shuffled with `random.Random(seed=42)`
3. Capped to `max_scenes` (default 100 per sensor) **before** splitting
4. Split at 80/20 boundary → 80 train scenes, 20 test scenes per sensor
5. Same seed always produces the same split — reproducible

No scene appears in both train and test. The cap and split happen at the scene level, not at the patch level.

## Patch Counts (100 scenes per sensor)

| | PRISMA | EnMAP | Combined |
|---|---|---|---|
| Train scenes | 80 | 80 | 160 |
| Test scenes | 20 | 20 | 40 |
| Intermediate train patches | ~20,700 | ~20,800 | ~41,500 |
| Intermediate test patches | ~5,300 | ~5,200 | ~10,500 |
| **Final train patches** | — | — | **43,000** |
| **Final test patches** | — | — | **10,000** |

Each scene produces ~260 valid patches at 128×128 with stride 64 (after 50% validity filtering).

## Wavelength Grid

All patches are resampled to a common 165-band grid that **excludes atmospheric absorption windows**:

| Segment | Range (nm) | Bands | Region |
|---------|-----------|-------|--------|
| 1 | 460–910 | 46 | Visible + NIR |
| 2 | 980–1130 | 16 | NIR/SWIR transition |
| 3 | 1160–1340 | 19 | SWIR-1 |
| 4 | 1460–1790 | 34 | SWIR-2 |
| 5 | 1960–2450 | 50 | SWIR-3 |

10nm spacing respects the coarsest sensor resolution (PRISMA ~12nm). No data fabricated across atmospheric gaps.

## Validity Thresholds (Three Levels)

| Level | Parameter | Default | Where applied |
|-------|-----------|---------|---------------|
| **Band-level** | `min_valid_pixel_pct` | 20% | Vendable construction: prunes bands with <20% valid pixels |
| **Pixel-level** | `max_invalid_voxel_fraction` | 0.4 | Vendable construction: invalidates pixel columns with >40% bad bands |
| **Patch-level** | `patch_validity_threshold` | 0.5 | Sharding: discards patches with <50% valid pixels |

All configurable via CLI flags.

## EnMAP COG Compatibility

EnMAP scenes may use either standard naming (`-SPECTRAL_IMAGE.TIF`) or Cloud Optimized GeoTIFF naming (`-SPECTRAL_IMAGE_COG.tiff`). The `EnmapHelper._resolve_path()` tries both conventions transparently. COG files are read identically by rasterio — same bands, dtype, and nodata.

## Files

| File | Role |
|------|------|
| `app/utils/patch_generation/hyperspectral_patcher.py` | Patch extraction with ascending wavelength fallback |
| `app/utils/patch_generation/intermediate/prisma_intermediate_patcher.py` | PRISMA: .he5 download → vendable → patch → shard |
| `app/utils/patch_generation/intermediate/enmap_intermediate_patcher.py` | EnMAP: folder download → vendable → patch → shard |
| `app/utils/patch_generation/final/hyperspectral_final_patcher.py` | Multi-sensor final shuffler |
| `scripts/generate_hyperspectral_patches.py` | CLI orchestration |

## Usage

```bash
# Full pipeline: 100 scenes per sensor, 128×128 patches
python -m scripts.generate_hyperspectral_patches --sizes 128 --parallel 1

# Only EnMAP intermediate (PRISMA already done)
python -m scripts.generate_hyperspectral_patches --sizes 128 --sensors enmap --skip-final --parallel 1

# Only final mixing (intermediates already done)
python -m scripts.generate_hyperspectral_patches --sizes 128 --skip-intermediate

# Custom thresholds
python -m scripts.generate_hyperspectral_patches --sizes 128 \
    --min-valid-pixel-pct 30 \
    --max-invalid-voxel-fraction 0.3 \
    --patch-validity-threshold 0.6

# Quick test: 2 scenes per sensor
python -m scripts.generate_hyperspectral_patches --sizes 128 --max-scenes 2
```

## Disk Usage

During processing, peak local disk usage is ~6 GB:
- 1 scene download (~2 GB PRISMA or ~500 MB EnMAP)
- 1 active shard being written (~1 GB)
- With `--parallel 1`: one scene + one shard at a time

Shards are uploaded to S3 and deleted locally as they complete. No disk accumulation.
