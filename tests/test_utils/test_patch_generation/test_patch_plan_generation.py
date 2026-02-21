"""
Ensures that patch plans are generated properly
"""

from typing import Dict

import pytest
import numpy as np


from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator
from app.models.patches.patching_response import PatchingPlan, PatchRequest

# Creating fixtures


@pytest.fixture
def patch_examples() -> Dict[str, np.ndarray]:
    """
    Fixtures for testing patch plans
    """

    return {
        "simple_1_band": np.random.rand(1, 100, 100),
        "simple_1_band_variation": np.random.rand(1, 100, 101),
        "100_band_pure_horizontal": np.random.rand(100, 10, 1400),
    }


def test_patch_plan_works_on_divisible_patches(patch_examples):
    """
    Tests to see if the patch plan works on a set of patches such that the patch size is divisible
    by the size of the image
    """

    # Initialize the patch generator
    patch_generator = PatchPlanGenerator()

    # Create a patch request
    request_1 = PatchRequest(
        input_cube=patch_examples.get("simple_1_band"), width=10, height=10, stride=10
    )

    patch_plan_1 = patch_generator.generate_patching_plan(request_1)
    # Perform checks
    assert isinstance(patch_plan_1, PatchingPlan)
    assert len(patch_plan_1.patch_coordinates) == 100

    request_2 = PatchRequest(
        input_cube=patch_examples.get("simple_1_band_variation"),
        width=10,
        height=10,
        stride=10,
    )

    patch_plan_2 = patch_generator.generate_patching_plan(request_2)
    # Perform checks
    assert len(patch_plan_2.patch_coordinates) == 110

    ### Test out some errors
    with pytest.raises(ValueError):
        request_3 = PatchRequest(
            input_cube=patch_examples.get("simple_1_band_variation"),
            width=10,
            height=10,
            stride=0,
        )
        patch_plan_3 = patch_generator.generate_patching_plan(request_3)

    ### Pure horizontal patching
    request_4 = PatchRequest(
        input_cube=patch_examples.get("100_band_pure_horizontal"),
        width=10,
        height=10,
        stride=10,
    )
    patch_plan_4 = patch_generator.generate_patching_plan(request_4)
    assert len(patch_plan_4.patch_coordinates) == 140
