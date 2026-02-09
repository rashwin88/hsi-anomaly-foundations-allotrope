"""
Uses mock data to test the lc09_l2sp_temperature conversion from DN
"""

import pytest
import numpy as np

from app.utils.data_transformations.l2sp_dn_to_temperature_transformer import (
    Lc09L2spStTransformer,
)
from app.models.units.surface_temperature import Temperature


@pytest.fixture
def transformer() -> Lc09L2spStTransformer:
    """
    The actual transformer
    """
    return Lc09L2spStTransformer()


@pytest.fixture
def sample_array() -> np.ndarray:
    """
    Sample test frame
    """
    return np.array([[44177, 44177], [44177, 44177]], dtype=np.uint16)


def test_full_temperature_suite(transformer, sample_array):
    """
    Full test of the temperature suite
    """
    k = transformer.transform(input_data=sample_array, unit=Temperature.KELVIN)
    c = transformer.transform(input_data=sample_array, unit=Temperature.CELSIUS)
    f = transformer.transform(input_data=sample_array, unit=Temperature.FAHRENHEIT)

    assert np.isclose(k[0, 0], 300.0, atol=0.01)
    assert np.isclose(c[0, 0], 26.85, atol=0.01)
    assert np.isclose(f[0, 0], 80.33, atol=0.01)

    assert isinstance(k, np.ndarray)
    assert isinstance(c, np.ndarray)
    assert isinstance(f, np.ndarray)

    # We can also mask the data and check once
    mask = np.array([[True, False], [True, False]], dtype=np.bool)
    masked_input = np.ma.masked_array(sample_array, mask=mask)

    k = transformer.transform(input_data=masked_input, unit=Temperature.KELVIN)
    c = transformer.transform(input_data=masked_input, unit=Temperature.CELSIUS)
    f = transformer.transform(input_data=masked_input, unit=Temperature.FAHRENHEIT)

    assert np.isclose(k[0, 1], 300.0, atol=0.01)
    assert np.isclose(c[0, 1], 26.85, atol=0.01)
    assert np.isclose(f[0, 1], 80.33, atol=0.01)

    assert isinstance(k, np.ma.MaskedArray)
    assert isinstance(c, np.ma.MaskedArray)
    assert isinstance(f, np.ma.MaskedArray)

    assert np.array_equal(k.mask, masked_input.mask)
    assert np.array_equal(c.mask, masked_input.mask)
    assert np.array_equal(f.mask, masked_input.mask)
