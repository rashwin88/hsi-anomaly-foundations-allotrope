"""
Characterisation tests for the frequency-domain destriper.

Written BEFORE refactoring it, not after. These do not claim the destriper is
correct - they pin down what it currently does, so a 124-line function can be
decomposed safely in a codebase with no other behavioural coverage of this path.

The fixture matters more than it looks. An early attempt used a clean sinusoid
on a smooth background and detected nothing at all, at any amplitude. The
detector scores a candidate angle as sigma above the background of the
power-vs-angle curve, and a synthetic scene has almost no background variance -
so the divisor collapses and the `bg_std > 0` guard rejects everything.
Counter-intuitively, the CLEANER the scene, the less likely a stripe is found.

So the fixture models what a real cube looks like: spatially correlated texture,
plus a per-detector column gain, which is the physical origin of pushbroom
striping. That reliably clears the threshold at ~3.4-5.3 sigma.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from app.utils.data_transformations.frequency_domain_destriper import (
    FrequencyDomainDestriper,
)

SIZE = 128
STRIPE_PERIOD_PX = 3.0


def _scene(bands: int = 4, stripe_amplitude: float = 0.12, seed: int = 3):
    """
    A realistic cube: correlated texture plus a per-detector column gain.

    Returns (cube, validity) in BSQ. `stripe_amplitude=0.0` gives a clean scene
    with no stripe, which the destriper should leave untouched.
    """
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.normal(size=(SIZE, SIZE)), sigma=4.0)
    texture = 0.4 + 0.15 * (texture / np.abs(texture).max())

    columns = np.arange(SIZE)
    col_gain = stripe_amplitude * np.sin(2 * np.pi * columns / STRIPE_PERIOD_PX)
    col_gain = col_gain[None, :]

    cube = np.stack(
        [
            (texture + col_gain + 0.005 * rng.normal(size=(SIZE, SIZE))).astype(
                np.float32
            )
            for _ in range(bands)
        ]
    )
    return cube, np.ones_like(cube, dtype=np.int8)


def test_transform_preserves_shape_and_dtype():
    cube, validity = _scene()
    out = FrequencyDomainDestriper().transform(cube, validity_mask=validity)
    assert out.shape == cube.shape
    assert out.dtype == cube.dtype


def test_transform_is_deterministic():
    """Same input twice must give bit-identical output - no RNG, no device drift."""
    cube, validity = _scene()
    a = FrequencyDomainDestriper().transform(cube, validity_mask=validity)
    b = FrequencyDomainDestriper().transform(cube, validity_mask=validity)
    assert np.array_equal(a, b)


def test_detects_the_injected_stripe_angle():
    """Column-wise striping must be found near 0 degrees."""
    cube, validity = _scene(stripe_amplitude=0.12)
    d = FrequencyDomainDestriper()
    d.transform(cube, validity_mask=validity)
    found = d._last_detected_angles
    assert found, "no stripe angle detected in a deliberately striped cube"
    # 0 and 180 describe the same orientation.
    assert any(min(a, 180.0 - a) <= 6.0 for a in found), found


def test_striped_cube_is_actually_modified():
    cube, validity = _scene(stripe_amplitude=0.12)
    out = FrequencyDomainDestriper().transform(cube, validity_mask=validity)
    assert not np.array_equal(out, cube)


def test_destriping_reduces_stripe_energy():
    """
    The point of the module. Measured against the input's own stripe energy so
    the assertion does not depend on absolute scale.
    """
    cube, validity = _scene(stripe_amplitude=0.12)
    out = FrequencyDomainDestriper().transform(cube, validity_mask=validity)

    def stripe_energy(band: np.ndarray) -> float:
        # The column-gain pattern lives in the column-mean profile.
        profile = band.mean(axis=0)
        spectrum = np.abs(np.fft.rfft(profile - profile.mean()))
        return float(spectrum.max())

    assert stripe_energy(out[0]) < stripe_energy(cube[0])


def test_no_detected_angles_returns_the_cube_unchanged(monkeypatch):
    """
    When no stripe is found, transform returns a copy rather than round-tripping
    through the FFT and perturbing every value.

    The empty-detection branch is forced rather than provoked with a clean
    scene, because a clean scene does NOT reliably produce one: correlated
    texture alone can clear the 3-sigma gate and trigger filtering. That
    false-positive tendency is recorded in docs/09-known-issues.md; this test is
    about the branch, not the detector.
    """
    cube, validity = _scene(stripe_amplitude=0.0, seed=11)
    d = FrequencyDomainDestriper()
    monkeypatch.setattr(d, "_find_stripe_angles", lambda *a, **k: [])
    out = d.transform(cube, validity_mask=validity)
    assert np.array_equal(out, cube)
    assert out is not cube, "must return a copy, not the caller's array"


def test_validity_mask_is_required():
    cube, _ = _scene()
    with pytest.raises(ValueError, match="validity_mask"):
        FrequencyDomainDestriper().transform(cube)


def test_invalid_pixels_do_not_break_detection():
    """A partially masked band must still be processed, not crash."""
    cube, validity = _scene(stripe_amplitude=0.12)
    validity[:, :20, :] = 0
    out = FrequencyDomainDestriper().transform(cube, validity_mask=validity)
    assert out.shape == cube.shape
    assert np.isfinite(out).all()
