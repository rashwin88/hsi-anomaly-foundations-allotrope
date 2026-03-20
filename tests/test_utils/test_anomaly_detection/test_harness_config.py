"""
Tests for InferenceHarnessConfig and PatchConfig validation.
"""

import pytest
from pydantic import ValidationError

from app.models.anomaly_detection.harness_config import (
    PatchConfig,
    InferenceHarnessConfig,
)


# ------------------------------------------------------------------
# PatchConfig
# ------------------------------------------------------------------


def test_patch_config_valid():
    pc = PatchConfig(patch_height=64, patch_width=64, stride=32)
    assert pc.patch_height == 64
    assert pc.patch_width == 64
    assert pc.stride == 32


def test_patch_config_rejects_zero_height():
    with pytest.raises(ValidationError):
        PatchConfig(patch_height=0, patch_width=64, stride=32)


def test_patch_config_rejects_negative_stride():
    with pytest.raises(ValidationError):
        PatchConfig(patch_height=64, patch_width=64, stride=-1)


def test_patch_config_rejects_zero_width():
    with pytest.raises(ValidationError):
        PatchConfig(patch_height=64, patch_width=0, stride=32)


# ------------------------------------------------------------------
# InferenceHarnessConfig — defaults
# ------------------------------------------------------------------


def test_default_config_is_full_scene():
    config = InferenceHarnessConfig()
    assert config.patch_config is None
    assert config.batch_size == 1
    assert config.fit_on_full_scene is True


def test_config_with_patch_config():
    pc = PatchConfig(patch_height=32, patch_width=32, stride=16)
    config = InferenceHarnessConfig(patch_config=pc, batch_size=8)
    assert config.patch_config is pc
    assert config.batch_size == 8


# ------------------------------------------------------------------
# InferenceHarnessConfig — validation
# ------------------------------------------------------------------


def test_batch_size_greater_than_one_requires_patching():
    with pytest.raises(ValidationError, match="batch_size"):
        InferenceHarnessConfig(batch_size=4)


def test_batch_size_one_without_patching_is_valid():
    config = InferenceHarnessConfig(batch_size=1)
    assert config.batch_size == 1


def test_batch_size_rejects_zero():
    with pytest.raises(ValidationError):
        InferenceHarnessConfig(
            patch_config=PatchConfig(patch_height=32, patch_width=32, stride=16),
            batch_size=0,
        )


def test_fit_on_full_scene_can_be_disabled():
    config = InferenceHarnessConfig(fit_on_full_scene=False)
    assert config.fit_on_full_scene is False
