"""
Abstract base class for foundation model training.

Provides the training loop, dataloader construction, LR scheduling,
checkpointing, and device management. Concrete trainers implement
build_model(), compute_loss(), and validation_step().
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import webdataset as wds

from app.models.training.training_config import (
    TrainingConfig,
    LRScheduleConfig,
)
from app.utils.general_utils.shard_pipe_expression_builder import (
    shard_pipe_expression_builder,
)
from app.utils.torch_helpers.device_selection import get_device

logger = logging.getLogger("FoundationTrainer")
logger.setLevel(logging.INFO)


class FoundationTrainer(ABC):
    """
    Contract for training a foundation model.

    Concrete trainers implement:
      - build_model()       → instantiate nn.Module from config
      - compute_loss()      → loss for one batch
      - validation_step()   → eval for one batch (returns loss value)

    The base class provides:
      - train()             → full training loop across all sizes per epoch
      - _build_dataloader() → webdataset from S3 for a given split + size
      - _build_scheduler()  → LR scheduler from config
      - _save_checkpoint()  → save model + optimizer + config
      - _cleanup_checkpoints() → keep only top K by val loss
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Device
        if config.device is not None:
            self.device = torch.device(config.device)
        else:
            self.device = get_device()

        # Model + optimizer (built once, shared across all sizes)
        self.model = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = self._build_scheduler()

    # ------------------------------------------------------------------
    # Abstract methods — concrete trainers must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Instantiate the nn.Module from self.config.model_config_."""
        ...

    @abstractmethod
    def compute_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the scalar training loss for one batch.
        Returns (loss, num_valid_samples) — num_valid_samples reflects
        how many samples survived any filtering (e.g. min valid pixel threshold).
        """
        ...

    @abstractmethod
    def validation_step(
        self, batch: dict, model: nn.Module
    ) -> tuple[float, int]:
        """
        Run one validation batch.
        Returns (loss_value, num_valid_samples).
        """
        ...

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Full training loop.

        Each epoch:
          1. Train on all patch sizes (each capped at train_samples_per_epoch)
          2. Validate on all patch sizes (each capped at test_samples_per_epoch)
          3. Step LR scheduler
          4. Checkpoint if due
        """
        data = self.config.data
        patch_sizes = data.patch_sizes

        for epoch in range(data.num_epochs):
            # --- Training ---
            self.model.train()
            epoch_train_loss = 0.0
            epoch_train_samples = 0

            for size in patch_sizes:
                cap = data.train_samples_per_epoch[size]
                loader = self._build_dataloader(split="train", size=size)
                size_loss, size_samples = self._run_train_pass(loader, cap)
                epoch_train_loss += size_loss
                epoch_train_samples += size_samples

            avg_train_loss = epoch_train_loss / max(epoch_train_samples, 1)

            # --- Validation ---
            self.model.eval()
            val_losses = {}

            for size in patch_sizes:
                cap = data.test_samples_per_epoch[size]
                loader = self._build_dataloader(split="test", size=size)
                val_loss = self._run_val_pass(loader, cap)
                val_losses[size] = val_loss

            avg_val_loss = sum(val_losses.values()) / len(val_losses)

            # --- LR schedule ---
            if self.scheduler is not None:
                if self.config.lr_schedule.scheduler_type == "plateau":
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # --- Logging ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            val_str = ", ".join(
                f"{s}px: {l:.6f}" for s, l in sorted(val_losses.items())
            )
            logger.info(
                f"Epoch {epoch + 1}/{data.num_epochs} | "
                f"train_loss: {avg_train_loss:.6f} | "
                f"val_loss: [{val_str}] | "
                f"avg_val: {avg_val_loss:.6f} | "
                f"lr: {current_lr:.2e}"
            )

            # --- Checkpointing ---
            ckpt = self.config.checkpoint
            if (epoch + 1) % ckpt.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1, avg_train_loss, val_losses)
                self._cleanup_checkpoints()

    def _run_train_pass(
        self, loader, sample_cap: int
    ) -> tuple[float, int]:
        """
        Train on one patch size, stop after sample_cap valid samples.

        Only samples that survive filtering (e.g. min valid pixel check)
        count toward the cap. Discarded patches don't contribute gradients
        and don't count toward the epoch sample budget.
        """
        total_loss = 0.0
        valid_samples = 0

        for batch in loader:
            if valid_samples >= sample_cap:
                break

            self.optimizer.zero_grad()
            loss, num_kept = self.compute_loss(batch, self.model)

            if num_kept == 0:
                continue  # entire batch filtered out, skip

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * num_kept
            valid_samples += num_kept

        return total_loss, valid_samples

    def _run_val_pass(self, loader, sample_cap: int) -> float:
        """
        Validate on one patch size, stop after sample_cap valid samples.

        Same filtering logic as training — only valid samples count.
        """
        total_loss = 0.0
        valid_samples = 0

        with torch.no_grad():
            for batch in loader:
                if valid_samples >= sample_cap:
                    break

                loss, num_kept = self.validation_step(batch, self.model)

                if num_kept == 0:
                    continue

                total_loss += loss * num_kept
                valid_samples += num_kept

        return total_loss / max(valid_samples, 1)

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _build_dataloader(self, split: str, size: int):
        """Build a webdataset dataloader for a given split and patch size."""
        shard_key = self.config.data.resolve_shard_key(split=split, size=size)
        pipe_expr = shard_pipe_expression_builder(
            data_key=shard_key,
            bucket_name=self.config.data.bucket_name,
            region_name=self.config.data.region_name,
        )
        dataset = wds.WebDataset(
            pipe_expr, shardshuffle=self.config.data.shardshuffle
        ).decode()

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
        )

    # ------------------------------------------------------------------
    # LR Scheduler
    # ------------------------------------------------------------------

    def _build_scheduler(self):
        """Build the LR scheduler from config."""
        cfg = self.config.lr_schedule
        stype = cfg.scheduler_type

        if stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.data.num_epochs,
                eta_min=cfg.min_lr,
            )
        elif stype == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.step_size,
                gamma=cfg.step_gamma,
            )
        elif stype == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=cfg.step_gamma,
                patience=cfg.step_size,
                min_lr=cfg.min_lr,
            )
        else:
            logger.warning(f"Unknown scheduler type '{stype}', no scheduling applied")
            return None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, epoch: int, train_loss: float, val_losses: dict[int, float]
    ) -> None:
        """Save a checkpoint with full reproducibility info."""
        ckpt_dir = Path(self.config.checkpoint.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        name = self.config.foundation_model_name.value
        version = self.config.version
        filename = f"{name}_v{version}_epoch{epoch}.pt"
        path = ckpt_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_losses": val_losses,
            "avg_val_loss": sum(val_losses.values()) / len(val_losses),
            "config": self.config.model_dump(mode="json", by_alias=True),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def _cleanup_checkpoints(self) -> None:
        """Keep only the top K checkpoints by average validation loss."""
        ckpt_dir = Path(self.config.checkpoint.checkpoint_dir)
        keep_top_k = self.config.checkpoint.keep_top_k
        name = self.config.foundation_model_name.value
        version = self.config.version
        prefix = f"{name}_v{version}_"

        # Find all checkpoints for this model + version
        ckpt_files = sorted(ckpt_dir.glob(f"{prefix}*.pt"))
        if len(ckpt_files) <= keep_top_k:
            return

        # Load avg_val_loss from each and sort
        scored = []
        for f in ckpt_files:
            ckpt = torch.load(f, weights_only=False)
            scored.append((f, ckpt.get("avg_val_loss", float("inf"))))

        scored.sort(key=lambda x: x[1])

        # Delete everything past top K
        for f, _ in scored[keep_top_k:]:
            f.unlink()
            logger.info(f"Removed checkpoint: {f}")
