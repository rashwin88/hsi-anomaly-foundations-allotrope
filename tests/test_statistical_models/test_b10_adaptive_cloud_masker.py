"""
Performs tests on the adaptive cloud masker
"""

import pytest
import numpy as np
from app.statistical_models.b10_adaptive_cloud_masker import B10AdaptiveCloudMasker
from app.utils.data_transformations.l2sp_dn_to_temperature_transformer import (
    Lc09L2spStTransformer,
)
from app.models.units.surface_temperature import Temperature
from app.utils.files.tif_helper import TIFHelper
from app.models.intermediate_concepts.adaptive_cloud_masker_response import (
    AdaptiveCloudMaskerResponse,
)
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier


@pytest.fixture
def base_data(live_source_data) -> np.ma.MaskedArray:
    """
    Creates a simple celsius scale masked cube as input to the model
    """

    helper = TIFHelper(
        file_source_config=live_source_data.get("phase_2_thermal_1"),
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    raw_data = helper.extract_specific_bands(
        bands=[1], masking_needed=True, mode="numpy"
    )

    temperature_converter = Lc09L2spStTransformer()
    temperature_data = temperature_converter.transform(
        input_data=raw_data, unit=Temperature.CELSIUS
    )

    return temperature_data


@pytest.mark.large_files
def test_adaptive_cloud_masker(base_data):
    """
    Tests the adaptove cloud masker on actual data
    Where there is some cloud cover.
    """

    # intiialize the model
    model = B10AdaptiveCloudMasker()
    model.configure(sampling_ratio=0.1)
    model.train(base_data)
    output = model.predict(base_data)

    assert isinstance(output, AdaptiveCloudMaskerResponse)
    assert output.pixels_masked > 1000
    assert isinstance(output.cloud_mask, np.ndarray)


@pytest.mark.large_files
@pytest.mark.large_benchmarks
def test_adaptive_b10_cloud_masker_benchmark(benchmark, base_data):
    """
    Benchmarks performance
    """
    model = B10AdaptiveCloudMasker()
    model.configure(sampling_ratio=0.1)
    # Run the benchmark
    result = benchmark(model.train, base_data)
    assert result is None
