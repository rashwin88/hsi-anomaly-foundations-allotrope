"""
Tests for the EnMAP file helper.
"""

import pytest
import numpy as np

from app.models.file_processing.sources import FileSourceConfig
from app.models.file_processing.enmap_metadata import EnmapMetadata
from app.models.hyperspectral_concepts.file_components import EnmapFileComponents
from app.utils.files.enmap_helper import EnmapHelper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier


SCENE_FOLDER = "tests/test_payloads/ENMAP01-____L2A-DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z"


@pytest.fixture
def enmap_helper() -> EnmapHelper:
    config = FileSourceConfig(source_path=SCENE_FOLDER)
    return EnmapHelper(
        file_source_config=config,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.ENMAP_HYPERSPECTRAL),
    )


@pytest.mark.large_files
def test_initialization(enmap_helper):
    """Verify helper loads and metadata is populated."""
    assert isinstance(enmap_helper.file_metadata, EnmapMetadata)
    assert enmap_helper.masked_pixel_value == -32768
    assert len(enmap_helper.file_metadata.band_characterisation) == 224


@pytest.mark.large_files
def test_extract_all_bands(enmap_helper):
    """Verify full spectral cube shape is (224, H, W)."""
    cube = enmap_helper.extract_specific_bands(bands=[], mode="all")
    assert cube.ndim == 3
    assert cube.shape[0] == 224
    assert cube.dtype == np.int16


@pytest.mark.large_files
def test_extract_specific_bands(enmap_helper):
    """Verify subset extraction returns correct number of bands."""
    cube = enmap_helper.extract_specific_bands(bands=[1, 2, 3], mode="specific")
    assert cube.shape[0] == 3


@pytest.mark.large_files
def test_extract_pixel_mask(enmap_helper):
    """Verify pixel mask shape matches spectral cube and values are 0/1."""
    mask = enmap_helper.extract_pixel_mask()
    assert mask.ndim == 3
    assert mask.shape[0] == 224
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


@pytest.mark.large_files
def test_extract_quality_mask_cloud(enmap_helper):
    """Verify cloud mask shape is (H, W) and dtype is uint8."""
    mask = enmap_helper.extract_quality_mask(EnmapFileComponents.QUALITY_CLOUD)
    assert mask.ndim == 2
    assert mask.dtype == np.uint8


@pytest.mark.large_files
def test_extract_quality_mask_snow(enmap_helper):
    """Verify snow mask reads correctly."""
    mask = enmap_helper.extract_quality_mask(EnmapFileComponents.QUALITY_SNOW)
    assert mask.ndim == 2
    assert mask.dtype == np.uint8
