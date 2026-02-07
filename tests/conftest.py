"""
Test configurations
"""

from typing import Dict, List
import pytest
from app.models.file_processing.sources import FileSourceConfig


@pytest.fixture
def mock_source() -> FileSourceConfig:
    """
    Mocks a file source configuration
    """
    return FileSourceConfig(source_path="test.he5")


@pytest.fixture
def mock_tif_source() -> FileSourceConfig:
    """
    Mocks a tif source
    """
    return FileSourceConfig(source_path="test.tif")


@pytest.fixture
def live_source_data() -> Dict[str, FileSourceConfig]:
    """
    Provides a dictionary of file sources. Can add to this if there are
    more tests to be done on large files.
    """
    return {
        "hyperspectral_1": FileSourceConfig(
            source_path="tests/test_payloads/Hyper_1/PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
        ),
        "hyperspectral_3": FileSourceConfig(
            source_path="tests/test_payloads/Hyper_3/PRS_L2D_STD_20210516050459_20210516050503_0001.he5"
        ),
        "thermal_1": FileSourceConfig(
            source_path="tests/test_payloads/thermal_1/LC09_L2SP_150044_20251009_20251010_02_T1_ST_B10.TIF"
        ),
        "thermal_2": FileSourceConfig(
            source_path="tests/test_payloads/thermal_2/LC09_L2SP_147049_20251121_20251122_02_T1_ST_B10.TIF"
        ),
        "phase_2_hyperspectral_1": FileSourceConfig(
            source_path="tests/test_payloads/phase_2/Set-1/Hypersepctral Datasets/PRS_L2D_STD_20201214060713_20201214060717_0001.he5"
        ),
        "phase_2_thermal_1": FileSourceConfig(
            source_path="tests/test_payloads/phase_2/Set-1/Thermal Anomaly Datasets/LC09_L2SP_141045_20250604_20250605_02_T1_ST_B10.TIF"
        ),
    }


@pytest.fixture
def hyperspectral_band_numbers() -> List[int]:
    """
    Returns a standard set of band numbers to test against
    """

    return [11, 12, 35, 50, 49, 30, 32]
