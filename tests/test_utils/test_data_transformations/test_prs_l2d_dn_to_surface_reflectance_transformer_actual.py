"""
Tests the reflectance transformer on actual data
"""

import pytest
import numpy as np

from app.utils.files.he5_helper import HE5Helper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.images.cube_representation import CubeRepresentation
from app.utils.data_transformations.prs_l2d_dn_to_surface_reflectance_transformer import (
    PrsL2dDnToSurfaceReflectanceTransformer,
)


@pytest.mark.large_files
def test_sr_transformation_on_actual_data(benchmark, live_source_data):
    """
    Tests reflectance conversion on actual data
    """
    # Create the helper
    helper = HE5Helper(
        file_source_config=live_source_data.get("hyperspectral_1"),
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.PRISMA_HYPERSPECTRAL),
    )

    # Now extract the bands one by one
    swir_bands = helper.extract_specific_bands(
        bands=[1], spectral_family=SpectralFamily.SWIR, masking_needed=True, mode="all"
    )
    vnir_bands = helper.extract_specific_bands(
        bands=[1], spectral_family=SpectralFamily.VNIR, masking_needed=True, mode="all"
    )
    # Concatentate the things.
    full_bands = np.ma.concatenate([swir_bands, vnir_bands], axis=1)

    # Just assert once - being defensive here
    assert full_bands.shape[0] == swir_bands.shape[0]

    # Create the band_mappings
    band_mappings = [SpectralFamily.SWIR] * swir_bands.shape[1] + [
        SpectralFamily.VNIR
    ] * vnir_bands.shape[1]

    # We now have everything we need
    # Initalize the transformer
    converter = PrsL2dDnToSurfaceReflectanceTransformer()
    # Perform the transformation
    output = converter.transform(
        input_data=full_bands,
        cube_representation=CubeRepresentation.BIL,
        band_mapping=band_mappings,
        file_metadata=helper.file_metadata,
        masking_indicator=0.0,
    )

    assert isinstance(output, np.ma.MaskedArray)
    assert output.shape == full_bands.shape
    assert output.data.max() <= 1.0
    assert output.data.min() >= 0.0

    # Perform benchmarking
    result = benchmark(
        converter.transform,
        full_bands,
        CubeRepresentation.BIL,
        band_mappings,
        helper.file_metadata,
        0.0,
    )
    assert result is not None
