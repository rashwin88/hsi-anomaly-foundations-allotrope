"""
Abstract base class for foundation model inference.

Handles device setup, model instantiation, and checkpoint loading.
Concrete inferencers implement build_model() and infer().
"""

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from app.models.training.inference_config import InferenceConfig
from app.utils.torch_helpers.device_selection import get_device

logger = logging.getLogger("FoundationInferencer")
logger.setLevel(logging.INFO)


class FoundationInferencer(ABC):
    """
    Contract for running inference with a foundation model.

    Concrete inferencers implement:
      - build_model()  → instantiate nn.Module from config
      - infer()        → model-specific inference logic

    The base class provides:
      - predict()      → public entry point, takes a tensor and mask
      - _load_weights() → loads model weights from checkpoint
    """

    def __init__(self, config: InferenceConfig):
        self.config = config

        # Device
        if config.device is not None:
            self.device = torch.device(config.device)
        else:
            self.device = get_device()

        # Model
        self.model = self.build_model().to(self.device)
        self._load_weights(config.checkpoint_path)
        self.model.eval()

        logger.info(
            f"Inferencer ready: {config.foundation_model_name.value} "
            f"on {self.device}, patch_size={config.patch_size}"
        )

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Instantiate the nn.Module from self.config.model_config_."""
        ...

    @abstractmethod
    def infer(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Model-specific inference logic.

        Args:
            tensor: Input tensor (B, C, H, W) on self.device.
            mask: Validity mask (B, 1, H, W) on self.device, same spatial size as tensor.

        Returns:
            Model output tensor.
        """
        ...

    def predict(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Run inference on a tensor with a validity mask.

        Moves inputs to device, calls infer() under no_grad.

        Args:
            tensor: (B, C, H, W) input patches.
            mask: (B, 1, H, W) validity mask, same spatial size as tensor.

        Returns:
            Model output tensor.
        """
        tensor = tensor.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            return self.infer(tensor, mask)

    def _load_weights(self, path: str) -> None:
        """Load model weights from a checkpoint."""
        logger.info(f"Loading weights from: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"]
        if getattr(self.config, "pixel_stats_override", None) is not None:
            # The override (e.g. HotSat-1's per-scene DN z-score) lives in the PixelNormalize / PixelDenormalize
            # buffers build_model set; the checkpoint's own stats would overwrite it on load (found 2026-10-31:
            # runs with and without the override were identical). Keep the override's buffers.
            from app.foundation_models.components.pixel_normalization import PixelDenormalize, PixelNormalize
            keep = {f"{name}.{b}" for name, mod in self.model.named_modules()
                    if isinstance(mod, (PixelNormalize, PixelDenormalize)) for b in ("mean", "std")}
            state = {k: v for k, v in state.items() if k not in keep}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected or set(missing) - keep:
                raise RuntimeError(f"checkpoint mismatch: missing {sorted(set(missing) - keep)}, "
                                   f"unexpected {unexpected}")
            logger.info(f"  pixel_stats_override kept: {sorted(keep)}")
        else:
            self.model.load_state_dict(state)
        logger.info(
            f"  Loaded weights from epoch {ckpt.get('epoch', '?')}, "
            f"avg_val_loss: {ckpt.get('avg_val_loss', 'N/A')}"
        )
