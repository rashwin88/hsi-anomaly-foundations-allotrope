"""
The scoring rasters switch to BigTIFF when classic TIFF could overflow (v1.0.1).

A native-band AVIRIS-NG reconstruction.tif ((C, H, W) float32) is over 4 GB uncompressed and classic TIFF aborted
with "Maximum TIFF file size exceeded". The writers now pass BIGTIFF=IF_SAFER: GDAL picks BigTIFF (header magic 43)
when the uncompressed size passes its ~2 GB safety line, and keeps small files classic (magic 42) so every existing
reader of the small rasters is unaffected. The large case writes zeros (np.zeros pages are not committed until
touched, and LZW shrinks them to a few MB), so it needs neither the RAM nor the disk its declared size suggests.
"""
from __future__ import annotations

import shutil
import struct

import numpy as np
import pytest


def _magic(path) -> int:
    with open(path, "rb") as fh:
        head = fh.read(4)
    return struct.unpack("<H" if head[:2] == b"II" else ">H", head[2:4])[0]


def test_small_rasters_stay_classic(tmp_path):
    from allotrope.action_types._anomaly_scoring_render import _save_raster_2d, _save_raster_3d

    _save_raster_2d(np.random.rand(20, 30), tmp_path / "a.tif", "float32")
    _save_raster_3d(np.random.rand(3, 20, 30), tmp_path / "b.tif", "float32")
    assert _magic(tmp_path / "a.tif") == 42 and _magic(tmp_path / "b.tif") == 42


def test_a_raster_over_the_classic_limit_is_bigtiff(tmp_path):
    from allotrope.action_types._anomaly_scoring_render import _save_raster_3d

    if shutil.disk_usage(tmp_path).free < 2 << 30:
        pytest.skip("under 2 GB free for the temporary BigTIFF")
    arr = np.zeros((2, 16384, 16384), np.float32)                # 2.1 GB uncompressed
    _save_raster_3d(arr, tmp_path / "big.tif", "float32")
    assert _magic(tmp_path / "big.tif") == 43
