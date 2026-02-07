"""
Tests whether basic TIF file handling works as expected
"""

import pytest
import numpy as np
from app.utils.files.tif_helper import TIFHelper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier
from app.models.file_processing.file_categories import FileCategory
from app.models.file_processing.file_metadata_models import TIFMetadata, TIFProperty
from app.models.hyperspectral_concepts.file_components import ThermalComponents


PHASE1 = "thermal_1"
PHASE2 = "phase_2_thermal_1"


@pytest.mark.large_files
def test_class_initialization(live_source_data):
    """
    Tests whether the class can be initialized without errors
    """
    source = live_source_data.get(PHASE2)

    # Initialize the helper
    helper = TIFHelper(
        file_source_config=source,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    # Run the requisite tests
    assert helper.file_source_config.file_category == FileCategory.TIF
    assert isinstance(helper.file_metadata, TIFMetadata)
    assert len(helper.file_metadata.metadata.keys()) > 0
    assert isinstance(helper.file_metadata.metadata.get("driver"), TIFProperty)

    # Make sure that the templates are loaded
    assert len(helper.template.keys()) > 0


@pytest.mark.large_files
def test_full_band_extraction(live_source_data):
    """
    Tests whether all the bands can be etxracted at once and properly
    """
    source = live_source_data.get(PHASE2)

    # Initialize the helper
    helper = TIFHelper(
        file_source_config=source,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    # Collect all the bands
    thermal_cube = helper.extract_specific_bands(masking_needed=False, mode="all")
    assert isinstance(thermal_cube, np.ndarray)
    # This usually vends data in the BSQ format
    assert thermal_cube.shape[0] == 1
    assert (
        thermal_cube.shape[1]
        == helper.file_metadata.metadata.get(
            helper.template.get(ThermalComponents.HEIGHT).property_name
        ).value
    )
    assert (
        thermal_cube.shape[2]
        == helper.file_metadata.metadata.get(
            helper.template.get(ThermalComponents.WIDTH).property_name
        ).value
    )


@pytest.mark.large_files
def test_masked_band_extraction(live_source_data):
    """
    Tests whether masked bands can be extracted properly
    """
    source = live_source_data.get(PHASE2)

    # Initialize the helper
    helper = TIFHelper(
        file_source_config=source,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    # Collect all the bands
    thermal_cube = helper.extract_specific_bands(masking_needed=True, mode="all")
    assert isinstance(thermal_cube, np.ma.MaskedArray)


@pytest.mark.large_files
def test_specific_band_extraction(live_source_data):
    """
    Tests whether masked bands can be extracted properly
    """
    source = live_source_data.get(PHASE1)

    # Initialize the helper
    helper = TIFHelper(
        file_source_config=source,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    # Collect one of the bands
    thermal_cube = helper.extract_specific_bands(
        bands=[1], masking_needed=False, mode="specific"
    )
    assert isinstance(thermal_cube, np.ndarray)
    # This usually vends data in the BSQ format
    assert thermal_cube.shape[0] == 1
    assert (
        thermal_cube.shape[1]
        == helper.file_metadata.metadata.get(
            helper.template.get(ThermalComponents.HEIGHT).property_name
        ).value
    )
    assert (
        thermal_cube.shape[2]
        == helper.file_metadata.metadata.get(
            helper.template.get(ThermalComponents.WIDTH).property_name
        ).value
    )  # Pylint : disable=no-member


@pytest.mark.large_files
def test_exceptions(live_source_data):
    """
    Tests whether masked bands can be extracted properly
    """
    source = live_source_data.get(PHASE1)

    # Initialize the helper
    helper = TIFHelper(
        file_source_config=source,
        template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
    )

    with pytest.raises(Exception):
        # Collect one of the bands which is invalid
        thermal_cube = helper.extract_specific_bands(
            bands=[4], masking_needed=False, mode="specific"
        )
        del thermal_cube
