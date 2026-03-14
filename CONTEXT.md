# Project Context: hsi-anomaly-foundations-allotrope

## Overview

Data engineering and preprocessing pipeline for **Hyperspectral Image (HSI) Anomaly Detection**, codenamed "Allotrope". The project ingests raw satellite imagery (PRISMA hyperspectral and Landsat 9 thermal), normalizes it into standardized datasets, and generates training-ready patches for downstream ML models.

**Python 3.14** | **Branch:** `ashwin/initial-exploration` | **No pyproject.toml** — uses `requirements.txt`

## Key Dependencies

h5py, pydantic, numpy, torch, matplotlib, rasterio, pystac, scikit-learn, boto3, webdataset, numexpr, requests, python-dotenv

## Architecture

```
app/
├── abstract_classes/     # ABCs: FileHelper, DatasetBuilder, DataTransformer, IntermediateSharder, MlModel
├── models/               # Pydantic data models (domain types, enums, request/response objects)
├── templates/            # Sensor-specific file structure templates (PRISMA HE5, Landsat TIF)
├── utils/                # All processing logic
│   ├── files/            # File helpers: HE5Helper, TIFHelper
│   ├── stac/             # STAC item creation, bounding box extraction, filename parsing
│   ├── dataset_builder/  # PrismaDatasetBuilder, LandsatDataBuilder
│   ├── data_transformations/  # DN-to-physical-unit converters
│   ├── image_transformation/  # Cube format conversion (BIL/BSQ/BIP)
│   ├── patch_generation/      # Patch planning, intermediate sharding, final shuffling
│   ├── band_operations/       # Band fusion (stub)
│   ├── visualization/         # Band-level visualization helpers
│   ├── external_apis/         # USGS M2M scene search/download client
│   ├── general_utils/         # S3 upload, paginated listing, shard pipe expressions
│   └── torch_helpers/         # Device selection (CUDA/MPS/CPU)
├── statistical_models/   # B10AdaptiveCloudMasker (GMM-based cloud detection)
├── dataset/              # Package marker
└── errors/               # ImplementationIncompleteError
tests/                    # Mirrors app/ structure, uses pytest with markers: large_files, large_benchmarks, network_access
notebooks/                # Exploration notebooks (band experiments, patching, visual confirmation)
docs/                     # Concept docs, design decisions, architecture notes
```

## Data Pipeline Flow

```
Raw Files (HE5/TIF on S3)
    │
    ▼
FileHelper (HE5Helper / TIFHelper)          ← Reads raw bands, extracts metadata
    │
    ▼
DatasetBuilder (Prisma / Landsat)            ← Applies transformations, builds vendable datasets
    │  ├─ DN → Surface Reflectance (PRISMA)
    │  └─ DN → Surface Temperature (Landsat)
    │
    ▼
VendableDataset                              ← Normalized cubes + validity/cloud masks
    │
    ▼
PatchPlanGenerator                           ← Generates patch coordinates (stride-based tiling)
    │
    ▼
IntermediateSharder (Landsat)                ← Downloads from S3, patches, writes webdataset shards
    │
    ▼
FinalPatchShuffler                           ← Shuffles intermediate shards into final training set
```

## Supported Sensors

### PRISMA (Hyperspectral)
- **File format:** HE5 (HDF-5)
- **Bands:** 66 VNIR + 173 SWIR (63 + 171 valid)
- **Native format:** BIL (H x C x W), e.g. 1210 x 66 x 1219
- **Transformation:** DN → Surface Reflectance via per-band scaling factors from L2D metadata
- **Output:** `VendableHyperspectralDataset` with normalized cube (BSQ), validity masks, spectral family order, wavelengths, FWHM

### Landsat 9 (Thermal)
- **File format:** GeoTIFF (via rasterio)
- **Band:** B10 thermal (uint16 DN)
- **Native format:** BSQ
- **Transformation:** `ST(K) = 0.00341802 * DN + 149.0`, then K → C/F
- **Cloud masking:** B10AdaptiveCloudMasker (5-component GMM on temperature distribution)
- **QA Pixel:** Provider cloud/water/snow masks via bitwise extraction
- **Output:** `VendableThermalDataset` with normalized thermal cube, cloud mask, validity mask, provider QA layers

## Key Abstractions

### FileHelper<T> (Generic ABC)
Wraps HE5 (h5py) and TIF (rasterio) files with consistent interface: `extract_specific_bands()`, `file_metadata`, `access_dataset()`. Templates are injected, not looked up — sensor-agnostic design.

### DatasetBuilder (ABC)
Converts raw files → vendable datasets. Owns STAC item, file helper, band information. Core method: `vend_dataset()`.

### Templates
Dictionaries mapping logical file components (e.g., `SWIR_CUBE_DATA`) to physical paths in HE5/TIF files via `ReferenceDefinition` objects. Three reference types: `FILE_REFERENCE`, `ROOT_METADATA_FIELD`, `DIRECT_PROPERTY_DEFINITION`.

### Cube Representations
- **BIL** (Band Interleaved by Line): H x C x W — PRISMA native
- **BSQ** (Band Sequential): C x H x W — Landsat native, standard ML format
- **BIP** (Band Interleaved by Pixel): H x W x C — visualization format
- `ImageCubeOperations.convert_cube()` handles all conversions via torch permutation

### Patch Generation
Three-tier system:
1. **PatchPlanGenerator** — computes patch coordinates from cube dimensions + stride
2. **LandsatIntermediateSharder** — downloads scenes from S3, builds datasets, patches, writes intermediate webdataset shards (filters patches with <50% validity)
3. **FinalPatchShuffler** — reads intermediate shards, shuffles across scenes, writes final training shards

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
