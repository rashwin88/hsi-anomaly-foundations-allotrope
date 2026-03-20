"""
Tests for the EnMAP L2A DN-to-surface-reflectance transformer.
"""

import numpy as np
import pytest

from app.utils.data_transformations.enmap_l2a_dn_to_surface_reflectance_transformer import (
    EnmapL2aDnToSurfaceReflectanceTransformer,
)
from app.models.dataset.transformations import Transformation


@pytest.fixture
def transformer():
    return EnmapL2aDnToSurfaceReflectanceTransformer()


def test_transformation_category(transformer):
    assert transformer.transformation_category == Transformation.ENMAP_L2A_DN_TO_SR


def test_transform_basic(transformer):
    """DN=10000 should yield SR=1.0 (10000 * 0.0001)."""
    input_data = np.array([[[10000, 5000], [2500, 0]]], dtype=np.int16)
    output = transformer.transform(input_data)
    assert output.dtype == np.float32
    np.testing.assert_almost_equal(output[0, 0, 0], 1.0, decimal=5)
    np.testing.assert_almost_equal(output[0, 0, 1], 0.5, decimal=5)
    np.testing.assert_almost_equal(output[0, 1, 0], 0.25, decimal=5)


def test_transform_nodata(transformer):
    """Nodata pixels (DN=-32768) should be zeroed out."""
    input_data = np.array([[[10000, -32768]]], dtype=np.int16)
    output = transformer.transform(input_data, nodata_value=-32768)
    np.testing.assert_almost_equal(output[0, 0, 0], 1.0, decimal=5)
    assert output[0, 0, 1] == 0.0


def test_transform_shape_preserved(transformer):
    """Output shape should match input shape."""
    input_data = np.random.randint(0, 10000, size=(10, 50, 50), dtype=np.int16)
    output = transformer.transform(input_data)
    assert output.shape == input_data.shape


def test_transform_masked_array(transformer):
    """Transformer should handle masked arrays."""
    data = np.array([[[10000, 5000]]], dtype=np.int16)
    mask = np.array([[[False, True]]])
    input_data = np.ma.MaskedArray(data, mask=mask)
    output = transformer.transform(input_data)
    assert output.dtype == np.float32
    np.testing.assert_almost_equal(output[0, 0, 0], 1.0, decimal=5)
