# Project Context: hsi-anomaly-foundations-allotrope

## Overview

Data engineering and preprocessing pipeline for **Hyperspectral Image (HSI) Anomaly Detection**, codenamed "Allotrope". The project ingests raw satellite imagery (PRISMA hyperspectral, EnMAP hyperspectral, and Landsat 9 thermal), normalizes it into standardized datasets, and generates training-ready patches for downstream ML models.

**Python 3.14** | **No pyproject.toml** — uses `requirements.txt`

## Key Dependencies

h5py, pydantic, numpy, torch, matplotlib, rasterio, pystac, scikit-learn, boto3, webdataset, numexpr, requests, python-dotenv, scipy

## Architecture

```
app/
├── abstract_classes/     # ABCs: FileHelper, DatasetBuilder, DataTransformer, IntermediateSharder, MlModel
├── models/               # Pydantic data models (domain types, enums, request/response objects)
├── templates/            # Sensor-specific file structure templates (PRISMA HE5, Landsat TIF, EnMAP)
├── utils/                # All processing logic
│   ├── files/            # File helpers: HE5Helper, TIFHelper, EnmapHelper
│   ├── stac/             # STAC item creation, bounding box extraction, filename parsing
│   ├── dataset_builder/  # PrismaDatasetBuilder, LandsatDataBuilder, EnmapDatasetBuilder
│   ├── data_transformations/  # DN-to-physical-unit converters, spectral band filter,
│   │                          # spectral interpolator, spectral resampler
│   ├── image_transformation/  # Cube format conversion (BIL/BSQ/BIP)
│   ├── patch_generation/      # Patch planning, intermediate sharding, final shuffling
│   │   ├── intermediate/      # LandsatIntermediateSharder, PrismaIntermediateSharder,
│   │   │                      # EnmapIntermediateSharder
│   │   └── final/             # FinalPatchShuffler, HyperspectralFinalShuffler
│   ├── band_operations/       # Band fusion (stub)
│   ├── visualization/         # Band-level visualization helpers
│   ├── external_apis/         # USGS M2M scene search/download client
│   └── general_utils/         # S3 upload, paginated listing, shard pipe expressions
├── statistical_models/   # B10AdaptiveCloudMasker (GMM-based cloud detection)
├── detectors/            # GRX, LRX, MNF+GRX, MNF+LRX, Statistical Ensemble
├── dataset/              # Package marker
└── errors/               # ImplementationIncompleteError
scripts/                  # Orchestration scripts for patch generation (Landsat, Hyperspectral)
tests/                    # Mirrors app/ structure, uses pytest with markers: large_files, large_benchmarks, network_access
notebooks/                # Exploration notebooks (band experiments, patching, visual confirmation, pipeline walkthrough)
benchmarking/             # Benchmark notebooks for hyperspectral and thermal anomaly detection
docs/                     # Concept docs, design decisions, architecture notes
```

## Data Pipeline Flow

### Thermal (Landsat 9)

```
Raw TIF Files (S3)
    │
    ▼
TIFHelper                                    ← Reads B10 band + QA_PIXEL
    │
    ▼
LandsatDataBuilder                           ← DN → Surface Temperature (K→C)
    │  ├─ B10AdaptiveCloudMasker (GMM)
    │  └─ QA_PIXEL bitwise extraction
    │
    ▼
VendableThermalDataset                       ← Temp cube + cloud/validity/QA masks
    │
    ▼
LandsatIntermediateSharder                   ← Download → Build → Patch → Filter (>50%) → Shard
    │
    ▼
FinalPatchShuffler                           ← Shuffle across scenes → Final shards
```

### Hyperspectral (PRISMA / EnMAP)

```
Raw Files (S3: prisma/*.he5, enmap/*/...)
    │
    ▼
FileHelper (HE5Helper / EnmapHelper)        ← Reads raw bands, extracts metadata
    │
    ▼
DatasetBuilder (Prisma / Enmap)              ← DN → Surface Reflectance
    │
    ▼
VendableDataset (raw, all bands)             ← 239 bands (PRISMA) / 224 bands (EnMAP)
    │
    ▼  ── BandFilterConfig controls all stages below ──
    │
[1] Bad band flag removal
[2] Wavelength exclusion (atmospheric windows)
[3] Detector edge trimming (3 bands/end)
[4] Coverage-aware band pruning (<20% valid → drop)
    │                                        ← ~188 bands survive
    ▼
[5] Quality mask invalidation (EnMAP only: cloud, shadow, haze)
[6] Spatial masking (>40% invalid voxels → full pixel invalidation)
[7] Spectral interpolation (PCHIP/linear gap-fill)
    │                                        ← Binary validity: fully valid OR fully invalid
    ▼
[8] Spectral resampling to common grid       ← 165 bands, 10nm spacing, 460-2450nm
    │                                        ← Identical shape across all sensors
    ▼
VendableDataset (clean, common grid)
    │
    ▼
PrismaIntermediateSharder / EnmapIntermediateSharder
    │                                        ← Download → Build → Patch → Filter → Shard
    ▼
HyperspectralFinalShuffler                   ← Mix PRISMA + EnMAP → Unified shards
    │
    ▼
patches/hyperspectral/{split}/final/         ← Training-ready mixed-sensor shards
```

## Supported Sensors

### PRISMA (Hyperspectral)
- **File format:** HE5 (HDF-5)
- **Bands:** 66 VNIR + 173 SWIR (239 total, ~234 valid)
- **Native format:** BIL (H x C x W), e.g. 1210 x 66 x 1219
- **Transformation:** DN → Surface Reflectance via per-band scaling factors from L2D metadata
- **After pipeline:** 165 bands on common 10nm grid (460–2450nm)

### EnMAP (Hyperspectral)
- **File format:** GeoTIFF folder (SPECTRAL_IMAGE + PIXELMASK + quality layers + METADATA.XML)
- **Bands:** 91 VNIR + 133 SWIR (224 total)
- **Native format:** BSQ (C x H x W)
- **Transformation:** `SR = DN * 0.0001` (uniform gain, all bands)
- **Quality layers:** Cloud, cirrus, haze, cloud shadow, snow (each H x W)
- **After pipeline:** 165 bands on common 10nm grid (460–2450nm)

### Landsat 9 (Thermal)
- **File format:** GeoTIFF (via rasterio)
- **Band:** B10 thermal (uint16 DN)
- **Native format:** BSQ
- **Transformation:** `ST(K) = 0.00341802 * DN + 149.0`, then K → C/F
- **Cloud masking:** B10AdaptiveCloudMasker (5-component GMM on temperature distribution)
- **QA Pixel:** Provider cloud/water/snow masks via bitwise extraction

## Key Abstractions

### FileHelper<T> (Generic ABC)
Wraps HE5 (h5py), TIF (rasterio), and EnMAP folder (rasterio+XML) files with consistent interface: `extract_specific_bands()`, `file_metadata`, `access_dataset()`. Templates are injected, not looked up — sensor-agnostic design.

### DatasetBuilder (ABC)
Converts raw files → vendable datasets. Owns STAC item, file helper, band information. Core method: `vend_dataset(band_filter_config=...)`.

### BandFilterConfig (Pydantic model)
Controls the full post-processing pipeline for hyperspectral vendables:
- `exclusion_ranges` — atmospheric absorption windows to exclude
- `edge_bands_to_trim` — detector edge bands to remove
- `min_valid_pixel_pct` — band-level coverage threshold
- `max_invalid_voxel_fraction` — pixel-level spatial masking threshold
- `quality_masks_to_apply` — EnMAP quality layers (cloud, shadow, haze)
- `common_wavelength_grid` — target grid for spectral resampling (default: 165 bands, 10nm, 460–2450nm excluding atmospheric windows)

### Spectral Processing Pipeline
- **SpectralBandFilter** — 4-stage band filtering (flags, wavelength, edge, coverage)
- **SpectralInterpolator** — PCHIP/linear gap-fill for partially-valid pixels
- **SpectralResampler** — Linear resampling to common wavelength grid (float32, memory-efficient)

### Templates
Dictionaries mapping logical file components to physical paths in HE5/TIF/EnMAP files via `ReferenceDefinition` objects. Three reference types: `FILE_REFERENCE`, `ROOT_METADATA_FIELD`, `DIRECT_PROPERTY_DEFINITION`.

### Cube Representations
- **BIL** (Band Interleaved by Line): H x C x W — PRISMA native
- **BSQ** (Band Sequential): C x H x W — EnMAP/Landsat native, standard ML format
- **BIP** (Band Interleaved by Pixel): H x W x C — visualization format
- `ImageCubeOperations.convert_cube()` handles all conversions via torch permutation

### Patch Generation (3-tier system)
1. **PatchPlanGenerator** — computes patch coordinates from cube dimensions + stride
2. **IntermediateSharder** (ABC) — per-sensor: download → build dataset → patch → filter → write webdataset shards → upload to S3
   - `LandsatIntermediateSharder` — thermal (B10 + QA_PIXEL)
   - `PrismaIntermediateSharder` — downloads .he5, builds vendable with BandFilterConfig
   - `EnmapIntermediateSharder` — downloads scene folder, builds vendable with BandFilterConfig
3. **Final Shuffler** — reads intermediate shards, shuffles across scenes, writes final shards
   - `FinalPatchShuffler` — single-sensor (Landsat)
   - `HyperspectralFinalShuffler` — multi-sensor, mixes PRISMA + EnMAP into unified shards

### Webdataset Shard Contents

**Hyperspectral patches:**
- `pixels.npy` — (165, H, W) float32 reflectance on common grid
- `validity_cube.npy` — (165, H, W) int8, binary validity
- `wavelengths.npy` — (165,) float64, ascending, identical across sensors
- `meta.json` — scene_id, coords, sensor, spectral families, band count

**Thermal patches:**
- `pixels.npy` — (1, H, W) float32 temperature
- `validity_cube.npy` — (1, H, W) int8
- `predicted_cloud_mask.npy`, `pure_validity_mask.npy`, provider QA masks
- `meta.json` — scene_id, coords, patch dims

### S3 Storage Layout

```
s3://allotrope-raw-data-india/
├── prisma/                                            # raw .he5 scenes
├── enmap/                                             # raw scene folders
├── landsat/                                           # raw Landsat scenes
├── patches/landsat/{split}/intermediate/w{W}_h{H}_s{S}/
├── patches/landsat/{split}/final/w{W}_h{H}_s{S}/
├── patches/prisma/{split}/intermediate/w{W}_h{H}_s{S}/
├── patches/enmap/{split}/intermediate/w{W}_h{H}_s{S}/
├── patches/hyperspectral/{split}/final/w{W}_h{H}_s{S}/   # mixed PRISMA+EnMAP
```

### B10AdaptiveCloudMasker
GMM-based cloud detection for Landsat B10 thermal band:
- Computes percentile distribution of temperature scene
- Fits 5-component GMM on sampled temperatures
- Labels clusters with mean < (scene_median - 12°C) as cloud
- Returns binary cloud mask + diagnostics

## STAC Integration
`StacCreator` builds PySTAC Items from filenames:
- Parses metadata (platform, date, processing level) via `FileNameParser`
- Computes bounding boxes (rasterio for TIF, h5py lat/lon arrays for HE5)
- Creates GeoJSON geometry + STAC Item with `primary_input_datacube` asset

## External APIs
`M2MClient` / `M2MSampler` — USGS M2M API client for scene search, filtering, and download orchestration. Supports random sampling from query results and S3 upload with progress tracking.

## Testing
- **Framework:** pytest with custom markers (`large_files`, `large_benchmarks`, `network_access`)
- **Coverage:** XML coverage report generated
- **Structure:** Mirrors `app/` directory layout under `tests/`
- **Scope:** Unit tests for models, file helpers, transformations, STAC utils, patch generation, dataset builders

## Environment
- `.env` file for credentials (USGS, AWS)
- `.venv` with Python 3.14
- `download_options.json` — USGS API response artifact
