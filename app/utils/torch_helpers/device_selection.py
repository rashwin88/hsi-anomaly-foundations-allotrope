"""
Device setter across mac and other linux devices
"""

import logging
import torch

logger = logging.getLogger("DeviceSelection")
logger.setLevel(logging.INFO)


def get_device():
    """
    Automatically selects the best available device.
    Priority: CUDA (Nvidia) -> MPS (Mac Silicon) -> CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Using device: %s", device.upper())
    return torch.device(device)
