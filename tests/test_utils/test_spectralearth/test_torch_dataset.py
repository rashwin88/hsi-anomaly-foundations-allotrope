"""SpectralEarthTorchDataset: both modes, validity semantics, batching, worker safety."""

import json
import os
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from app.models.dataset.spectralearth import SpectralEarthPatch, load_band_table
from app.utils.spectralearth.catalog import SpectralEarthCatalog
from app.utils.spectralearth.source import SpectralEarthSource
from app.utils.spectralearth.torch_dataset import SpectralEarthTorchDataset

NODATA = -32768


class _StubSource:
    """Hands back a synthetic patch for any entry; pixel (0,0) loses one band."""

    def load(self, entry):
        bands = load_band_table()
        raw = np.full((bands.count, 128, 128), 2000, dtype=np.int16)
        raw[50, 0, 0] = NODATA
        return SpectralEarthPatch(site_id=entry.site_id, patch_id=entry.patch_id, raw=raw, bands=bands)


@pytest.fixture
def catalog():
    lines = [json.dumps({"relative_path": f"enmap/{s:07d}/00001.tif", "size_bytes": 1, "sha256": "0" * 64})
             for s in range(6)]
    return SpectralEarthCatalog.from_manifest_lines(lines, "bkt", "coll")


def test_native_mode_shapes_and_complete_spectrum_rule(catalog):
    ds = SpectralEarthTorchDataset(catalog, _StubSource(), mode="native")
    s = ds[0]
    assert s["pixels"].shape == (202, 128, 128) and s["pixels"].dtype == torch.float32
    assert s["validity"].shape == (1, 128, 128) and s["validity"].dtype == torch.uint8
    assert s["validity"][0, 0, 0] == 0 and not s["pixels"][:, 0, 0].any()   # one band missing -> out
    assert s["pixels"][0, 1, 1] == pytest.approx(0.2)
    assert s["key"] == "se_0000000_00001" and len(ds.wavelengths_nm) == 202


def test_common_grid_mode_fills_the_gap_instead(catalog):
    ds = SpectralEarthTorchDataset(catalog, _StubSource(), mode="common_grid")
    s = ds[0]
    assert s["pixels"].shape == (165, 128, 128) and len(ds.wavelengths_nm) == 165
    assert s["validity"][0, 0, 0] == 1                                       # interpolated, kept
    assert s["pixels"][:, 0, 0].numpy() == pytest.approx(np.full(165, 0.2), abs=1e-6)


def test_batches_through_a_dataloader(catalog):
    batch = next(iter(DataLoader(SpectralEarthTorchDataset(catalog, _StubSource()), batch_size=4)))
    assert batch["pixels"].shape == (4, 202, 128, 128)
    assert batch["validity"].shape == (4, 1, 128, 128) and len(batch["key"]) == 4


def test_rejects_unknown_mode(catalog):
    with pytest.raises(ValueError):
        SpectralEarthTorchDataset(catalog, _StubSource(), mode="raw")


def test_source_pickles_without_its_client():
    src = SpectralEarthSource(workers=4)
    src._client = object()                    # stands in for a live, unpicklable client
    clone = pickle.loads(pickle.dumps(src))
    assert clone._client is None and clone.workers == 4


@pytest.mark.network_access
def test_real_patches_from_the_inventory():
    uri = os.environ.get("SPECTRALEARTH_COLLECTION_URI",
                         "s3://allotrope-raw-inventory-india/enmap/SpectralEarth/ds_0b1abe70-8c20-4248-a3da-f0905027e8fa/")
    cat = SpectralEarthCatalog.from_files_manifest(uri)
    assert len(cat) == 539_755
    ds = SpectralEarthTorchDataset(cat.sample(2, seed=0), SpectralEarthSource(workers=2), mode="native")
    s = ds[0]
    assert s["pixels"].shape == (202, 128, 128) and 0.0 <= float(s["pixels"].max()) < 2.0
