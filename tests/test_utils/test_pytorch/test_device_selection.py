"""
Tests if device selection is working properly
"""

import torch
from app.utils.torch_helpers.device_selection import get_device


def test_get_device():
    """
    Tests whether the device is recognized correctly
    """

    # Simple test and nothing fancy here
    device: torch.device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda", "mps"]
