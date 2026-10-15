"""PRISMA native-band shards: the HE5 reader, the tiler and the site split (docs/lld/prisma-shards.md)."""

import h5py
import numpy as np
import pytest

from app.utils.prisma.scene import NODATA, SWATH, read_scene
from app.utils.prisma.split import BENCH_SITES, sites, split_scenes
from app.utils.prisma.tiles import tiles

H, W = 260, 300                                  # 2 x 2 full 128 px tiles


@pytest.fixture
def he5(tmp_path):
    """A tiny PRISMA L2D: 3 VNIR bands (one flagged off) and 2 SWIR bands, BIL cubes."""
    path = tmp_path / "PRS_L2D_STD_20240101050000_20240101050004_0001.he5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs.update({"List_Cw_Vnir": [700.0, 600.0, 500.0], "List_Cw_Vnir_Flags": [1, 0, 1],
                        "List_Cw_Swir": [1000.0, 2000.0], "List_Cw_Swir_Flags": [1, 1],
                        "L2ScaleVnirMin": 0.0, "L2ScaleVnirMax": 0.5, "L2ScaleSwirMin": 0.0, "L2ScaleSwirMax": 1.0,
                        "Sun_zenith_angle": 40.0, "Sun_azimuth_angle": 150.0, "Epsg_Code": 32643})
        d = f.create_group(SWATH + "/Data Fields")
        vn = rng.integers(1, 65535, (H, 3, W), dtype=np.uint16)
        vn[:, :, :150] = 0                          # left half outside the swath
        vn[5, 0, 200] = 13107                       # 13107 / 65535 x 0.5 = 0.1 reflectance
        sw = rng.integers(1, 65535, (H, 2, W), dtype=np.uint16)
        sw[:, :, :150] = 0
        err = np.zeros((H, 3, W), np.uint8); err[10, 2, 220] = 1
        d["VNIR_Cube"], d["SWIR_Cube"] = vn, sw
        d["VNIR_PIXEL_L2_ERR_MATRIX"], d["SWIR_PIXEL_L2_ERR_MATRIX"] = err, np.zeros((H, 2, W), np.uint8)
        g = f.create_group(SWATH + "/Geolocation Fields")
        g["Latitude"] = np.full((H, W), 12.0); g["Longitude"] = np.full((H, W), 77.0)
    return path


def test_reader_scales_drops_flagged_bands_and_marks_nodata(he5):
    s = read_scene(he5)
    assert s.pixels.shape == (4, H, W) and s.pixels.dtype == np.int16
    assert s.wavelengths_nm.tolist() == [700.0, 500.0, 1000.0, 2000.0]     # file order, 600 nm dropped
    assert s.pixels[0, 5, 200] == 1000                                        # 0.1 reflectance x 10,000
    assert (s.pixels[:, :, :150] == NODATA).all()                             # DN 0
    assert s.pixels[1, 10, 220] == NODATA and s.pixels[0, 10, 220] != NODATA  # error matrix, one band only
    assert s.facts["date"] == "20240101" and s.centre == (12.0, 77.0)


def test_tiles_keep_the_shard_layout_and_skip_empty_tiles(he5):
    out = dict(tiles(read_scene(he5)))
    assert sorted(out) == ["pr_20240101050000_r0c1", "pr_20240101050000_r1c1"]  # column 0 is all nodata
    t = out["pr_20240101050000_r0c1"]
    assert t["pixels.npy"].shape == (4, 128, 128) and t["wavelengths.npy"].dtype == np.float32
    assert t["meta.json"]["lat"] == 12.0 and t["meta.json"]["sun_zenith"] == 40.0
    assert not any(k.startswith("label_") for k in t)                        # L2D carries no labels


def test_split_groups_sites_and_holds_out_benchmark_sites():
    lat, lon = BENCH_SITES["set1"]
    centres = {"a": (lat + 0.05, lon), "b": (10.00, 77.00), "c": (10.05, 77.02), "d": (15.0, 75.0), "e": (28.0, 80.0)}
    assert ["b", "c"] in sites(centres)
    out = split_scenes(centres, test_fraction=0.3)
    assert out["a"] == "excluded" and out["b"] == out["c"] != "excluded"
    assert sum(v == "test" for v in out.values()) >= 1
    assert split_scenes(centres, test_fraction=0.3) == out                  # deterministic
    with pytest.raises(ValueError):
        split_scenes(centres, test_fraction=1.0)
