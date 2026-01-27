import torch
import logging

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

    logger.info(f"Using device: {device.upper()}")
    return torch.device(device)


# testing routine
if __name__ == "__main__":
    device = get_device()
    print(device)
