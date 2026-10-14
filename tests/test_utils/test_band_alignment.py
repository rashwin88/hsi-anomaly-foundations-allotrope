"""Band alignment: samples with different band lists onto one target list, by wavelength."""

import json

import numpy as np
import pytest

from app.utils.band_alignment import NODATA, align_bands, load_band_list


def test_reorders_selects_and_fills_by_wavelength():
    px = np.arange(4 * 2 * 2, dtype=np.int16).reshape(4, 2, 2)
    wl = np.array([500.0, 400.0, 700.0, 600.0])                 # not monotonic, like EnMAP's overlap
    out = align_bands(px, wl, np.array([400.0, 500.0, 650.0, 700.0]))
    assert np.array_equal(out[0], px[1]) and np.array_equal(out[1], px[0]) and np.array_equal(out[3], px[2])
    assert (out[2] == NODATA).all()                              # 650 nm has no source band
    assert out.dtype == np.int16 and out.shape == (4, 2, 2)


def test_tolerance_absorbs_rounding_but_never_pairs_neighbours():
    px = np.ones((2, 1, 1), np.int16)
    assert (align_bands(px, np.array([1518.87, 1530.29]), np.array([1518.9]))[0] == 1).all()
    assert (align_bands(px, np.array([1518.87, 1530.29]), np.array([1524.0]))[0] == NODATA).all()


def test_mismatched_lengths_refused_and_band_list_loaded(tmp_path):
    with pytest.raises(ValueError):
        align_bands(np.zeros((3, 1, 1), np.int16), np.array([1.0, 2.0]), np.array([1.0]))
    f = tmp_path / "stats.json"
    f.write_text(json.dumps({"wavelengths_nm": [418.4, 2445.3]}))
    assert load_band_list(f).tolist() == pytest.approx([418.4, 2445.3])


def test_trainer_transform_unifies_raw_and_compressed_samples():
    from app.foundation_models.trainers.spectral_stem_mae_trainer import align_sample
    target = np.array([400.0, 500.0], np.float32)
    raw = {"pixels.npy": np.ones((3, 2, 2), np.int16), "wavelengths.npy": np.array([400.0, 450.0, 500.0], np.float32)}
    comp = {"pixels.i16dz": np.ones((2, 2, 2), np.int16), "wavelengths.npy": np.array([500.0, 400.0], np.float32)}
    a, b = align_sample(raw, target), align_sample(comp, target)
    assert set(a) == set(b) == {"pixels.npy", "wavelengths.npy"}
    assert a["pixels.npy"].shape == b["pixels.npy"].shape == (2, 2, 2)
