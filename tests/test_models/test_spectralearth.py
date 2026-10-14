"""SpectralEarth band table and patch decoding (app/models/dataset/spectralearth.py)."""

import numpy as np
import pytest
from rasterio.io import MemoryFile

from app.models.dataset.spectralearth import (
    SpectralEarthPatch,
    load_band_table,
    parse_member_path,
)
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

DROPPED = set(range(127, 142)) | set(range(161, 168))


def _tif_bytes(cube: np.ndarray) -> bytes:
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", count=cube.shape[0], height=cube.shape[1],
                      width=cube.shape[2], dtype=cube.dtype.name, nodata=-32768) as dst:
            dst.write(cube)
        return mem.read()


def test_band_table_is_enmap_minus_the_published_drop_list():
    t = load_band_table()
    assert t.count == 202
    assert set(t.enmap_bands) == set(range(1, 225)) - DROPPED
    assert list(t.enmap_bands) == sorted(t.enmap_bands)
    assert t.reflectance_scale == 0.0001 and t.nodata == -32768


def test_families_split_at_the_detector_boundary():
    t = load_band_table()
    vnir = [w for w, f in zip(t.wavelengths_nm, t.families) if f is SpectralFamily.VNIR]
    swir = [w for w, f in zip(t.wavelengths_nm, t.families) if f is SpectralFamily.SWIR]
    assert (len(vnir), len(swir)) == (91, 111)       # all 22 drops are SWIR bands
    assert vnir == sorted(vnir) and swir == sorted(swir)
    assert swir[0] < vnir[-1]                          # the detectors overlap near 900-1000 nm


def test_authors_rgb_indices_land_on_red_green_blue():
    # spectral_earth's own loader uses rgb_indices = [43, 28, 10] for EnMAP
    w = load_band_table().wavelengths_nm
    assert 620 < w[43] < 660 and 540 < w[28] < 570 and 455 < w[10] < 480


def test_parse_member_path():
    assert parse_member_path("enmap/0019846/01121.tif") == ("0019846", "01121")
    with pytest.raises(ValueError):
        parse_member_path("enmap/0019846/01121.npy")


def test_decode_scales_reflectance_and_marks_nodata():
    cube = np.full((202, 128, 128), 1234, dtype=np.int16)
    cube[5, 0, 0] = -32768
    p = SpectralEarthPatch.from_geotiff_bytes(_tif_bytes(cube), "enmap/0000001/00007.tif")
    assert (p.site_id, p.patch_id, p.key) == ("0000001", "00007", "se_0000001_00007")
    r, v = p.reflectance(), p.validity()
    assert r.dtype == np.float32 and v.dtype == np.int8
    assert r[0, 1, 1] == pytest.approx(0.1234)
    assert r[5, 0, 0] == 0.0 and v[5, 0, 0] == 0 and v.sum() == 202 * 128 * 128 - 1


def test_decode_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match="expected int16"):
        SpectralEarthPatch.from_geotiff_bytes(
            _tif_bytes(np.zeros((224, 128, 128), dtype=np.int16)), "enmap/1/1.tif")
