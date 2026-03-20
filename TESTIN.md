# Test Commands

## Run All Tests
```bash
python -m pytest
```

## Run All Tests (with verbose output)
```bash
pytest tests/ -v
```

## Exclude Marks (fast, local-only tests)
```bash
pytest tests/ -m "not large_files and not large_benchmarks and not network_access"
```

## Run Only Specific Marks

### large_files — Tests requiring large local files
```bash
pytest tests/ -m large_files
```

### large_benchmarks — Heavy benchmarking tests
```bash
pytest tests/ -m large_benchmarks
```

### network_access — Tests requiring network access
```bash
pytest tests/ -m network_access
```

## Run by Test Module

```bash
# Models
pytest tests/test_models/test_sources.py
pytest tests/test_models/test_reference_definitions.py

# Abstract classes
pytest tests/test_abstract_classes/test_file_helper.py
pytest tests/test_abstract_classes/test_dataset_loader.py

# File helpers
pytest tests/test_utils/test_files/test_he5_helper.py
pytest tests/test_utils/test_files/test_tif_helper.py

# Data transformations
pytest tests/test_utils/test_data_transformations/test_lc09_ls2p_st_conversion_mock.py
pytest tests/test_utils/test_data_transformations/test_lc09_l2sp_st_conversion_actual.py
pytest tests/test_utils/test_data_transformations/test_prs_l2d_dn_to_surface_reflectance_transformer_mock.py
pytest tests/test_utils/test_data_transformations/test_prs_l2d_dn_to_surface_reflectance_transformer_actual.py

# Image transformation
pytest tests/test_utils/test_image_transformation/test_image_cube_operations.py
pytest tests/test_utils/test_image_transformation/test_advanced_image_cube_operations.py

# STAC utils
pytest tests/test_utils/test_stac/test_stac_utils/test_landsat_bounding_box_creation.py
pytest tests/test_utils/test_stac/test_stac_utils/test_prisma_bounding_box_creation.py
pytest tests/test_utils/test_stac/test_stac_utils/test_filename_parsers.py
pytest tests/test_utils/test_stac/test_stac_utils/test_stac_item_creation.py

# Dataset builders
pytest tests/test_utils/test_dataset_builder/test_prisma_dataset_building.py
pytest tests/test_utils/test_dataset_builder/test_landsat_dataset_builder.py

# Patch generation
pytest tests/test_utils/test_patch_generation/test_patch_plan_generation.py
pytest tests/test_utils/test_patch_generation/test_landsat_patcher.py

# General utils
pytest tests/test_utils/test_general_utils/test_paginated_s3_listing.py
pytest tests/test_utils/test_general_utils/test_shard_pipe_expression_builder.py

# Statistical models
pytest tests/test_statistical_models/test_b10_adaptive_cloud_masker.py

# PyTorch utils
pytest tests/test_utils/test_pytorch/test_device_selection.py
```

## Run with Coverage

### Generate coverage XML for Coverage Gutters (VSCode extension)
Coverage Gutters reads `coverage.xml` from the workspace root by default.
```bash
pytest tests/ --cov=app --cov-report=xml:coverage.xml
```

### Exclude heavy marks but still generate coverage
```bash
pytest tests/ -m "not large_files and not large_benchmarks and not network_access" --cov=app --cov-report=xml:coverage.xml
```

### With HTML report (for browsing in a browser)
```bash
pytest tests/ --cov=app --cov-report=xml:coverage.xml --cov-report=html:htmlcov
```

After running any of the above, toggle Coverage Gutters in VSCode:
- `Cmd+Shift+P` -> **Coverage Gutters: Display Coverage**
- It will read `coverage.xml` from the workspace root automatically

## Run Only Mock/Unit Tests (no large files, no network, no benchmarks)
```bash
pytest tests/ -m "not large_files and not large_benchmarks and not network_access" -v
```
