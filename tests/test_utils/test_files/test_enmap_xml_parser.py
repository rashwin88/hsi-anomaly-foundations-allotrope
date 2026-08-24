"""
Tests for the EnMAP XML metadata parser.
"""

import os
import pytest

from app.utils.files.enmap_xml_parser import EnmapXmlParser
from app.models.file_processing.enmap_metadata import (
    EnmapMetadata,
    EnmapBandCharacterisation,
    EnmapDetectorBoundary,
)

SCENE_FOLDER = "tests/test_payloads/ENMAP01-____L2A-DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z"
SCENE_NAME = os.path.basename(SCENE_FOLDER)
XML_PATH = os.path.join(SCENE_FOLDER, SCENE_NAME + "-METADATA.XML")

# Every test here parses a real EnMAP scene XML out of tests/test_payloads/,
# which is gitignored. Gate the whole module rather than each test.
pytestmark = pytest.mark.large_files


@pytest.fixture
def parsed_metadata() -> EnmapMetadata:
    parser = EnmapXmlParser(XML_PATH)
    return parser.parse()


def test_parse_returns_enmap_metadata(parsed_metadata):
    assert isinstance(parsed_metadata, EnmapMetadata)


def test_parse_band_characterisation(parsed_metadata):
    """Verify all 224 bands are parsed with correct first and last wavelengths."""
    bands = parsed_metadata.band_characterisation
    assert len(bands) == 224
    assert isinstance(bands[0], EnmapBandCharacterisation)

    # First band should be ~418nm
    assert 415 < bands[0].wavelength_center < 425
    assert bands[0].band_number == 1

    # Last band should be ~2445nm
    assert 2440 < bands[-1].wavelength_center < 2450
    assert bands[-1].band_number == 224

    # Gain and offset should be uniform
    assert all(b.gain == 0.0001 for b in bands)
    assert all(b.offset == 0 for b in bands)


def test_parse_detector_boundary(parsed_metadata):
    """Verify VNIR=91 channels, SWIR=133 channels."""
    db = parsed_metadata.detector_boundary
    assert isinstance(db, EnmapDetectorBoundary)
    assert db.num_vnir_bands == 91
    assert db.num_swir_bands == 133
    assert len(db.vnir_expected_channels) == 91
    assert len(db.swir_expected_channels) == 133
    # Total should be 224
    assert db.num_vnir_bands + db.num_swir_bands == 224


def test_vnir_swir_interleave(parsed_metadata):
    """Verify that channels in the overlap region are interleaved."""
    db = parsed_metadata.detector_boundary
    vnir_set = set(db.vnir_expected_channels)
    swir_set = set(db.swir_expected_channels)

    # No overlap — each channel belongs to exactly one detector
    assert len(vnir_set & swir_set) == 0
    # Union should cover 1-224
    assert vnir_set | swir_set == set(range(1, 225))

    # VNIR has channels above 78 (interleaved region)
    high_vnir = [ch for ch in db.vnir_expected_channels if ch > 78]
    assert len(high_vnir) > 0
    # SWIR has channels below 102 (interleaved region)
    low_swir = [ch for ch in db.swir_expected_channels if ch < 102]
    assert len(low_swir) > 0


def test_parse_bounding_polygon(parsed_metadata):
    """Verify bounding polygon has 4 corner points."""
    polygon = parsed_metadata.spatial_info.bounding_polygon
    # 4 corners (upper_left appears twice in XML, but we deduplicate by taking unique frames)
    assert len(polygon) >= 4
    for point in polygon:
        assert "latitude" in point
        assert "longitude" in point


def test_parse_quality_flags(parsed_metadata):
    """Verify quality flag values are parsed as numbers."""
    qf = parsed_metadata.quality_flags
    assert qf.cloud_cover >= 0
    assert qf.haze_cover >= 0
    assert qf.cirrus_cover >= 0
    assert qf.dead_pixels_vnir >= 0
    assert qf.dead_pixels_swir >= 0


def test_parse_spatial_info(parsed_metadata):
    """Verify scene dimensions and background value."""
    si = parsed_metadata.spatial_info
    assert si.width > 0
    assert si.height > 0
    assert si.background_value == -32768


def test_parse_mission_info(parsed_metadata):
    """Verify mission-level metadata fields."""
    assert parsed_metadata.mission == "EnMAP"
    assert parsed_metadata.satellite_id == "01"
    assert parsed_metadata.processing_level == "L2A"
    assert parsed_metadata.tile_id == "18"
    assert parsed_metadata.datatake_id == "0000059367"
