"""AVIRIS-NG native-band shards: the strip reader and the tiler (docs/lld/aviris-shards.md)."""

import numpy as np
import pytest

from app.utils.aviris.scene import NODATA, open_scene
from app.utils.aviris.tiles import tiles
from app.utils.shard_codec import decode_i16dz
from app.utils.spectralearth.sharding import BandStats

LINES, SAMPLES, BANDS = 260, 300, 5              # 2 x 2 full 128 px tiles


@pytest.fixture
def line(tmp_path):
    """A tiny BIL float32 flight line: band 3 marked bad (bbl 0), a nodata strip on the left."""
    folder = tmp_path / "ang20180318t060114"; folder.mkdir()
    cube = np.full((LINES, BANDS, SAMPLES), 0.25, np.float32)       # BIL: (lines, bands, samples)
    cube[:, :, :140] = -9999.0                                     # left strip outside the swath
    cube[10, 0, 200] = 0.1234
    cube.tofile(folder / "ang20180318t060114_corr_v2s1_img.bin")
    (folder / "ang20180318t060114_corr_v2s1_img.hdr").write_text(
        "ENVI\ndescription = {AVIRIS-NG Measured Radiances}\n"
        f"samples = {SAMPLES}\nlines = {LINES}\nbands = {BANDS}\nheader offset = 0\nfile type = ENVI\n"
        "data type = 4\ninterleave = bil\nbyte order = 0\ndata ignore value = -9999\n"
        "map info = { UTM , 1 , 1 , 400000.0 , 1300000.0 , 4.0 , 4.0 , 44 , North , WGS-84 , units=Meters , rotation=0.0 }\n"
        "wavelength units = Nanometers\nwavelength = { 500.0 , 600.0 , 1400.0 , 1600.0 , 2200.0 }\n"
        "fwhm = { 5.0 , 5.0 , 5.0 , 5.0 , 5.0 }\nbbl = { 1 , 1 , 0 , 1 , 1 }\n")
    (folder / "ang20180318t060114_h2o_v2s1_img.hdr").write_text("ENVI\n")   # must not be picked up
    return folder


def test_reader_drops_bad_bands_scales_and_marks_nodata(line):
    s = open_scene(line)
    assert (s.scene_id, s.facts["product"], s.facts["pixel_size_m"]) == ("ang20180318t060114", "v2s1", 4.0)
    assert s.wavelengths_nm.tolist() == [500.0, 600.0, 1600.0, 2200.0]
    px = s.strip(0, 128)
    assert px.shape == (4, 128, SAMPLES) and px.dtype == np.int16
    assert px[0, 10, 200] == 1234 and px[1, 10, 200] == 2500                # x 10,000
    assert (px[:, :, :140] == NODATA).all()


def test_tiles_compress_exactly_count_stats_and_skip_empty(line):
    stats = BandStats()
    out = dict(tiles(open_scene(line), stats))
    assert sorted(out) == ["av_20180318t060114_r0c1", "av_20180318t060114_r1c1"]  # column 0 is mostly nodata
    t = out["av_20180318t060114_r0c1"]
    px = decode_i16dz(t["pixels.i16dz"])
    assert px.shape == (4, 128, 128) and px[0, 10, 72] == 1234                 # x0 = 128: column 200 -> 72
    assert t["meta.json"]["product"] == "v2s1" and "map" not in t["meta.json"]
    assert not any(k.startswith("label_") for k in t)
    assert int(stats.count[1]) > 0                                             # stats from the uncompressed tile


def test_centre_comes_from_the_utm_map_info(line):
    lat, lon = open_scene(line).centre
    assert 11.0 < lat < 12.5 and 79.0 < lon < 80.5                             # UTM 44N, E 400 km, N 1300 km


def test_a_folder_without_the_reflectance_pair_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_scene(tmp_path)
