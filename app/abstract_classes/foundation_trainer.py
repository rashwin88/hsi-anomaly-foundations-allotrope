"""
Abstract base class for foundation model training.

Provides the training loop, dataloader construction, LR scheduling,
checkpointing, device management, and hot storage (local shard caching
with rotation). Concrete trainers implement build_model(), compute_loss(),
and validation_step().
"""

import logging
import random
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
import webdataset as wds

from app.models.training.training_config import TrainingConfig
from app.utils.general_utils.paginated_s3_listing import get_all_objects_paginated
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

        If hot storage is enabled, shards are synced to local disk in
        rotating windows. Test shards are synced once and reused.
        If disabled, streams directly from S3 (original behavior).

        Each epoch:
          1. Train on all patch sizes (each capped at train_samples_per_epoch)
          2. Validate on all patch sizes (each capped at test_samples_per_epoch)
          3. Step LR scheduler
          4. Checkpoint if due
        """
        data = self.config.data
        patch_sizes = data.patch_sizes
        hot = self.config.hot_storage

        if hot.enabled:
            self._train_with_hot_storage()
        else:
            self._train_streaming()

    def _train_streaming(self) -> None:
        """Original S3 streaming training loop."""
        data = self.config.data
        patch_sizes = data.patch_sizes

        # Build all dataloaders once upfront
        train_loaders = {
            size: self._build_dataloader(split="train", size=size)
            for size in patch_sizes
        }
        test_loaders = {
            size: self._build_dataloader(split="test", size=size)
            for size in patch_sizes
        }

        for epoch in range(data.num_epochs):
            self._run_epoch(epoch, train_loaders, test_loaders)

    def _train_with_hot_storage(self) -> None:
        """
        Sync-once hot storage training loop.

        1. Sync train and test shards to local disk (skipped if already present
           and skip_sync_if_exists=True — avoids repeated S3 egress costs)
        2. Build local dataloaders
        3. Train all epochs from local disk (shardshuffle provides data diversity)
        4. Shards are NOT deleted at the end — kept for future runs
        """
        data = self.config.data
        hot = self.config.hot_storage
        patch_sizes = data.patch_sizes

        # Sync shards (one-time unless skip_sync_if_exists is False)
        self._ensure_local_shards(
            split="test", sizes=patch_sizes, num_shards=hot.test_shards_per_size
        )
        self._ensure_local_shards(
            split="train", sizes=patch_sizes, num_shards=hot.train_shards_per_size
        )

        # Build local dataloaders
        train_loaders = {
            size: self._build_local_dataloader(split="train", size=size)
            for size in patch_sizes
        }
        test_loaders = {
            size: self._build_local_dataloader(split="test", size=size)
            for size in patch_sizes
        }

        # Train all epochs from local disk
        for epoch in range(data.num_epochs):
            self._run_epoch(epoch, train_loaders, test_loaders)

        logger.info("Training complete. Local shards kept for future runs.")

    def _run_epoch(
        self,
        epoch: int,
        train_loaders: dict,
        test_loaders: dict,
    ) -> None:
        """Run a single epoch: train all sizes, validate all sizes, schedule, checkpoint."""
        data = self.config.data
        patch_sizes = data.patch_sizes

        # --- Training ---
        self.model.train()
        epoch_train_loss = 0.0
        epoch_train_samples = 0

        for size in patch_sizes:
            cap = data.train_samples_per_epoch[size]
            size_loss, size_samples = self._run_train_pass(
                train_loaders[size], cap
            )
            epoch_train_loss += size_loss
            epoch_train_samples += size_samples

        avg_train_loss = epoch_train_loss / max(epoch_train_samples, 1)

        # --- Validation ---
        self.model.eval()
        val_losses = {}

        for size in patch_sizes:
            cap = data.test_samples_per_epoch[size]
            val_loss = self._run_val_pass(test_loaders[size], cap)
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
        """Build a webdataset dataloader streaming from S3."""
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

    def _build_local_dataloader(self, split: str, size: int):
        """Build a webdataset dataloader reading from local cached shards."""
        local_dir = self._local_shard_dir(split, size)
        shard_paths = sorted(local_dir.glob("*.tar"))

        if not shard_paths:
            raise FileNotFoundError(
                f"No shards found in {local_dir}. Did _sync_shards run?"
            )

        urls = [str(p) for p in shard_paths]
        dataset = wds.WebDataset(
            urls, shardshuffle=self.config.data.shardshuffle
        ).decode()

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
        )

    # ------------------------------------------------------------------
    # Hot storage helpers
    # ------------------------------------------------------------------

    def _local_shard_dir(self, split: str, size: int) -> Path:
        """Local directory for cached shards of a given split and size."""
        data = self.config.data
        subdir = f"{split}/{data.stage}/w{size}_h{size}_s{size // 2}"
        return Path(self.config.hot_storage.local_cache_dir) / subdir

    def _ensure_local_shards(
        self, split: str, sizes: list[int], num_shards: int
    ) -> None:
        """
        Ensure local shards exist for all sizes. If skip_sync_if_exists
        is True and shards are already present, skip the download entirely.
        """
        hot = self.config.hot_storage

        if hot.skip_sync_if_exists:
            # Check if all sizes already have shards locally
            all_present = all(
                any(self._local_shard_dir(split, size).glob("*.tar"))
                for size in sizes
                if self._local_shard_dir(split, size).exists()
            )
            if all_present and all(
                self._local_shard_dir(split, size).exists() for size in sizes
            ):
                total = sum(
                    len(list(self._local_shard_dir(split, size).glob("*.tar")))
                    for size in sizes
                )
                logger.info(
                    f"  Local {split} shards already exist ({total} shards). "
                    f"Skipping S3 sync."
                )
                return

        logger.info(f"Syncing {split} shards to local disk (one-time)...")
        self._sync_shards_all_sizes(split=split, sizes=sizes, num_shards=num_shards)

    def _sync_shards_all_sizes(
        self, split: str, sizes: list[int], num_shards: int
    ) -> None:
        """
        Sync shards for ALL patch sizes in one parallel pool.

        Flattens all download jobs across all sizes into a single thread pool.
        Skips files that already exist locally (partial resume support).
        """
        data = self.config.data
        max_parallel = self.config.hot_storage.max_parallel_downloads

        # Build the full download list across all sizes
        download_jobs = []  # list of (s3_key, local_path)

        for size in sizes:
            shard_key = data.resolve_shard_key(split=split, size=size)
            all_keys = get_all_objects_paginated(
                bucket_name=data.bucket_name,
                prefix_key=shard_key,
                region_name=data.region_name,
                page_size=500,
            )
            tar_keys = [k for k in all_keys if k.endswith(".tar")]

            if not tar_keys:
                raise FileNotFoundError(
                    f"No shards found at s3://{data.bucket_name}/{shard_key}"
                )

            selected = random.sample(tar_keys, min(num_shards, len(tar_keys)))

            local_dir = self._local_shard_dir(split, size)
            local_dir.mkdir(parents=True, exist_ok=True)

            for key in selected:
                filename = key.split("/")[-1]
                local_path = local_dir / filename
                # Skip files that already exist (partial resume)
                if not local_path.exists():
                    download_jobs.append((key, str(local_path)))

        if not download_jobs:
            logger.info(f"  All {split} shards already downloaded. Nothing to sync.")
            return

        def _download_one(job: tuple) -> None:
            key, local_path = job
            s3_url = f"s3://{data.bucket_name}/{key}"
            subprocess.run(
                ["aws", "s3", "cp", s3_url, local_path,
                 "--region", data.region_name],
                check=True,
                capture_output=True,
            )

        logger.info(
            f"  Downloading {len(download_jobs)} shards across "
            f"{len(sizes)} sizes ({max_parallel} parallel)..."
        )
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(_download_one, job): job for job in download_jobs}
            with tqdm(
                total=len(download_jobs),
                desc=f"Syncing {split} shards",
                unit="shard",
            ) as pbar:
                for future in as_completed(futures):
                    future.result()  # raises if download failed
                    pbar.update(1)

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
