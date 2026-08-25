"""
Tests on the hyperspectral patcher's segmentation label emission.

Unlike test_landsat_patcher.py these need no payloads - a synthetic vendable
is a handful of numpy arrays - so they are not gated behind `large_files`
and run on any machine.
"""

import numpy as np
import pytest

from app.models.dataset.vendables import VendableEnmapHyperspectralDataset
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.utils.patch_generation.generate_patch_plan import (
    PatchPlanGenerator,
    PatchRequest,
)
from app.utils.patch_generation.hyperspectral_patcher import (
    _LABEL_LAYERS,
    patch_hyperspectral_vendable,
)

BANDS, H, W, PATCH = 4, 32, 32, 16


class _StubVendable:
    """Only the attributes the patcher actually touches."""

    def __init__(self, with_labels: bool):
        self.normalized_hyperspectral_cube = np.zeros((BANDS, H, W), np.float32)
        self.validity_cube = np.ones((BANDS, H, W), np.int8)
        self.band_cw_order = [500.0, 600.0, 700.0, 800.0]
        self.spectral_family_order = [SpectralFamily.VNIR] * BANDS
        if with_labels:
            self.cloud_mask = np.zeros((H, W), np.uint8)
            self.cirrus_mask = np.full((H, W), 3, np.uint8)
            self.haze_mask = np.zeros((H, W), np.uint8)
            self.cloud_shadow_mask = np.zeros((H, W), np.uint8)
            self.snow_mask = np.zeros((H, W), np.uint8)
            self.quality_classes_mask = np.full((H, W), 2, np.uint8)


def _patches(vendable, include_labels, **kwargs):
    plan = PatchPlanGenerator().generate_patching_plan(
        PatchRequest(input_cube=(BANDS, H, W), width=PATCH, height=PATCH, stride=PATCH)
    )
    return list(
        patch_hyperspectral_vendable(
            vendable=vendable,
            patching_plan=plan,
            scene_id="SCENE",
            sensor="enmap",
            include_labels=include_labels,
            **kwargs,
        )
    )


def test_pixels_default_to_float32():
    """The reconstruction lane's format. Indradhanu trained on float32 shards;
    changing this default would silently alter them."""
    patch = _patches(_StubVendable(with_labels=False), include_labels=False)[0]
    assert patch["pixels.npy"].dtype == np.float32


def test_pixel_dtype_is_opt_in():
    """Segmentation sharding halves storage with float16; a trainer reading
    these must cast back to float32 before the model."""
    patch = _patches(
        _StubVendable(with_labels=False), include_labels=False, pixel_dtype=np.float16
    )[0]
    assert patch["pixels.npy"].dtype == np.float16
    assert patch["validity_cube.npy"].dtype == np.int8, "validity is unaffected"


def test_labels_absent_by_default():
    """The reconstruction path must be untouched by this feature."""
    patch = _patches(_StubVendable(with_labels=True), include_labels=False)[0]
    assert not [k for k in patch if k.startswith("label_")]


def test_all_six_labels_emitted():
    patch = _patches(_StubVendable(with_labels=True), include_labels=True)[0]
    assert {k for _, k in _LABEL_LAYERS} <= set(patch)


@pytest.mark.parametrize("key", [k for _, k in _LABEL_LAYERS])
def test_label_is_channel_first_uint8(key):
    """Every other shard array is (C, H, W); labels must match."""
    patch = _patches(_StubVendable(with_labels=True), include_labels=True)[0]
    assert patch[key].shape == (1, PATCH, PATCH)
    assert patch[key].dtype == np.uint8


def test_cirrus_grading_survives():
    """EnMAP grades cirrus 0-3 by thickness. Collapsing it to 0/1 here would
    silently discard that, and the trainer could never recover it."""
    patch = _patches(_StubVendable(with_labels=True), include_labels=True)[0]
    assert patch["label_cirrus.npy"].max() == 3


def test_classes_codes_not_remapped():
    """0 and 3 are both no-data, 1=land, 2=water. Remapping is the trainer's
    job; the shard records what the provider said."""
    patch = _patches(_StubVendable(with_labels=True), include_labels=True)[0]
    assert set(np.unique(patch["label_classes.npy"]).tolist()) <= {0, 1, 2, 3}


def test_vendable_without_masks_emits_nothing():
    """PRISMA ships no provider masks; include_labels must be a no-op there."""
    patch = _patches(_StubVendable(with_labels=False), include_labels=True)[0]
    assert not [k for k in patch if k.startswith("label_")]


def test_label_attrs_exist_on_the_real_vendable():
    """Guard against drift: the stub above is a fake, so assert the attribute
    names _LABEL_LAYERS looks up are really fields on the EnMAP vendable."""
    fields = set(VendableEnmapHyperspectralDataset.model_fields)
    assert {attr for attr, _ in _LABEL_LAYERS} <= fields
