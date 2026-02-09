"""
Actual tests of surface temperature conversion on real data
"""

import pytest
import numpy as np

from app.utils.data_transformations.l2sp_dn_to_temperature_transformer import (
    Lc09L2spStTransformer,
)
from app.models.units.surface_temperature import Temperature
from app.models.file_processing.sources import FileSourceConfig
from app.utils.files.tif_helper import TIFHelper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier


@pytest.fixture
def transformer() -> Lc09L2spStTransformer:
    """
    The actual data transformer
    """
    return Lc09L2spStTransformer()


@pytest.mark.large_files
def test_conversion_on_real_data(live_source_data, transformer):
    """
    Performs a conversion test on actual large dataset
    """
    dataset = live_source_data.get("phase_2_thermal_1")

    helper = TIFHelper(
        file_source_config=dataset,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    input_data: np.ma.MaskedArray = helper.extract_specific_bands(
        masking_needed=True, mode="all"
    )

    output_data: np.ma.MaskedArray = transformer.transform(
        input_data=input_data, unit=Temperature.KELVIN
    )

    assert isinstance(output_data, np.ma.MaskedArray)
    assert np.array_equal(output_data.mask, input_data.mask)


@pytest.mark.large_files
def test_transform_benchmark(benchmark, live_source_data, transformer):
    """
    Benchmarks performance
    """
    dataset = live_source_data.get("phase_2_thermal_1")

    helper = TIFHelper(
        file_source_config=dataset,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    input_data: np.ma.MaskedArray = helper.extract_specific_bands(
        masking_needed=True, mode="all"
    )

    # Run the benchmark
    result = benchmark(transformer.transform, input_data)
    assert result is not None
