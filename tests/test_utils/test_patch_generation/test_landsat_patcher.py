"""
Tests on the landsat patcher
"""

import pytest


from app.utils.dataset_builder.landsat_dataset_builder import LandsatDataBuilder
from app.utils.patch_generation.landsat_patcher import patch_landsat_vendable
from app.utils.patch_generation.generate_patch_plan import (
    PatchPlanGenerator,
    PatchRequest,
)


@pytest.mark.large_files
def test_landsat_patching_works_as_expected(live_source_data):
    """
    Tests to make sure that the landsat patcher works as expected.
    """
    builder = LandsatDataBuilder(
        file_source_configuration=live_source_data.get("thermal_usgs")
    )
    vendable = builder.vend_dataset(
        provider_qa_pixel_source=live_source_data.get("thermal_usgs_qa")
    )

    # Now create the patching plan
    generator = PatchPlanGenerator()
    plan = generator.generate_patching_plan(
        request=PatchRequest(
            input_cube=vendable.normalized_thermal_cube.shape,
            height=1024,
            width=1024,
            stride=20,
        )
    )

    patch_pipe = patch_landsat_vendable(
        vendable=vendable, patching_plan=plan, stac_item=builder.stac_item
    )

    patch = next(patch_pipe)

    assert isinstance(patch, dict)
    assert patch.get("meta.json").get("patch_height") == 1024
    assert patch.get("meta.json").get("patch_width") == 1024
    assert patch.get("meta.json").get("patch_stride") == 20
    assert patch.get("meta.json").get("bands") == 1

    # Assert individual elements
    for key, value in patch.items():
        if key.endswith("npy"):
            assert value.shape == (1, 1024, 1024)
