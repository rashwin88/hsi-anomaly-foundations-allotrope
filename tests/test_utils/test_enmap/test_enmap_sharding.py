"""EnMAP sharding pieces that need no real raster: split, file lookup, band selection, sample writer."""

import io
import json
import os
import tarfile

import numpy as np
import pytest

from app.utils.enmap.scene import SceneError, find_member, select_bands
from app.utils.enmap.split import split_datatakes
from app.utils.spectralearth.sharding import write_samples


def _scenes(n: int, cloudy_every: int = 6) -> list[dict]:
    return [{"datatake": f"DT{i:03d}", "cover_pct": {"cloud": 30.0 if i % cloudy_every == 0 else 0.0}} for i in range(n)]


def test_split_is_deterministic_stratified_and_whole_datatakes():
    scenes = _scenes(60) + [{"datatake": "DT000", "cover_pct": {"cloud": 0.0}}]   # a second tile of DT000
    test = split_datatakes(scenes)
    assert test == split_datatakes(list(reversed(scenes)))
    cloudy = {s["datatake"] for s in scenes if s["cover_pct"]["cloud"] > 5}
    assert test & cloudy and test - cloudy                     # both strata represented
    assert len(test) == round(0.1 * 10) + round(0.1 * 50)      # 10% of each stratum
    assert split_datatakes(scenes, seed="other") != test


def test_split_gives_a_rare_stratum_at_least_one_test_datatake():
    scenes = _scenes(40, cloudy_every=1000)                   # only DT000 cloudy
    assert "DT000" in split_datatakes(scenes)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_split_rejects_bad_fraction(bad):
    with pytest.raises(ValueError):
        split_datatakes(_scenes(10), bad)


def test_find_member_is_case_insensitive_and_refuses_ambiguity(tmp_path):
    for name in ("S-METADATA.xml", "S-SPECTRAL_IMAGE.TIF", "S-QL_QUALITY_CLOUD_COG.tiff", "S-QL_QUALITY_CLOUDSHADOW_COG.tiff"):
        (tmp_path / name).touch()
    assert os.path.basename(find_member(str(tmp_path), "METADATA")) == "S-METADATA.xml"
    assert os.path.basename(find_member(str(tmp_path), "SPECTRAL_IMAGE")) == "S-SPECTRAL_IMAGE.TIF"
    assert os.path.basename(find_member(str(tmp_path), "QL_QUALITY_CLOUD")) == "S-QL_QUALITY_CLOUD_COG.tiff"
    (tmp_path / "S-METADATA.XML").touch()
    if len(os.listdir(tmp_path)) == 5:                         # case-sensitive filesystem: two files
        with pytest.raises(SceneError):
            find_member(str(tmp_path), "METADATA")


def test_select_bands_by_wavelength_and_refuse_missing():
    scene = np.array([400.0, 410.0, 1518.87, 1530.29, 2445.0])
    assert select_bands(scene, np.array([1518.87, 400.0])).tolist() == [2, 0]
    with pytest.raises(SceneError):
        select_bands(scene, np.array([1525.0]))


def test_write_samples_keeps_member_order_types_and_refuses_dotted_keys():
    px = np.arange(2 * 4 * 4, dtype=np.int16).reshape(2, 4, 4)
    buf = io.BytesIO()
    keys, stats = write_samples([("en_DT1_01_r0c0", {"pixels.npy": px, "label_cloud.npy": np.zeros((1, 4, 4), np.uint8),
                                                      "meta.json": {"lat": 13.25}, "raw.bin": b"\x01\x02"})], buf)
    assert keys == ["en_DT1_01_r0c0"] and int(stats.count[0]) == 16
    with tarfile.open(fileobj=io.BytesIO(buf.getvalue())) as tar:
        assert [m.name.split(".", 1)[1] for m in tar.getmembers()] == ["pixels.npy", "label_cloud.npy", "meta.json", "raw.bin"]
        assert json.loads(tar.extractfile("en_DT1_01_r0c0.meta.json").read()) == {"lat": 13.25}
    with pytest.raises(ValueError):
        write_samples([("bad.key", {"pixels.npy": px})], io.BytesIO())
