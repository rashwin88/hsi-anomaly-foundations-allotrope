"""Band-delta + zstd codec for int16 shard cubes: exact, self-describing, smaller."""

import numpy as np
import pytest

from app.utils.shard_codec import MAGIC, decode_i16dz, encode_i16dz


def _spectral_cube(seed: int = 0) -> np.ndarray:
    """(202, 32, 32) smooth-ish spectra like real reflectance, with some nodata."""
    rng = np.random.default_rng(seed)
    base = rng.integers(500, 4000, (1, 32, 32))
    drift = np.cumsum(rng.integers(-40, 41, (202, 32, 32)), axis=0)
    cube = np.clip(base + drift, 0, 10_000).astype(np.int16)
    cube[:, :3, :3] = -32768                                     # off-swath corner
    cube[100:110, 5, 5] = -32768                                 # dead bands in one pixel
    return cube


def test_round_trip_is_exact():
    cube = _spectral_cube()
    assert np.array_equal(decode_i16dz(encode_i16dz(cube)), cube)


def test_int16_overflow_between_bands_wraps_exactly():
    # -32768 next to 10,000 is a difference of 42,768: outside int16, so it must wrap and unwrap.
    cube = np.array([[[10_000]], [[-32768]], [[10_000]], [[32767]], [[-32768]]], np.int16)
    assert np.array_equal(decode_i16dz(encode_i16dz(cube)), cube)


def test_header_carries_shape_and_the_result_is_smaller():
    cube = _spectral_cube(1)
    blob = encode_i16dz(cube)
    assert blob.startswith(MAGIC)
    assert decode_i16dz(blob).shape == (202, 32, 32)
    assert len(blob) < cube.nbytes / 1.5


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((), np.int16)])
def test_rejects_non_int16_or_scalar(bad):
    with pytest.raises(ValueError):
        encode_i16dz(bad)


def test_rejects_foreign_or_truncated_data():
    blob = encode_i16dz(_spectral_cube(2))
    with pytest.raises(ValueError):
        decode_i16dz(b"NOTIT!" + blob[6:])
    with pytest.raises(Exception):
        decode_i16dz(blob[:-100])
