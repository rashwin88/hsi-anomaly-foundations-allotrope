"""
Trainer for SpectralStemSegFormerMAE - the SpectralEarth pretraining experiment's A/B/C variants.

Registered in trainer_factory for `spectral_stem_mae`; reads the shards made by
scripts/shard_spectralearth.py and normalises with their stats.json.

Extends HyperspectralSegFormerMAETrainer and keeps its token masking, erosion, SAM ramp and
two-pass validation, so the only thing that differs from today's Indradhanu training is what
the experiment is about. What changes is the data contract: shards carry raw int16 with a
per-voxel nodata value and their own wavelengths, so validity is per band and the loss counts
only valid bands (design decision 6, docs/proposals/spectral-attention-bottleneck.md).

Gotchas:
  - pixels.npy is int16, x 0.0001 = reflectance, -32768 = dead voxel.
  - A pixel is spatially valid if ANY band is valid; its dead bands are skipped, not the pixel.
  - When every sample in a batch has the same wavelengths (always, for SpectralEarth), one
    (N,) row is passed instead of (B, N), which lets attention share its queries.
"""

import csv
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from app.foundation_models.components.sam_loss import SAMLoss
from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.pixel_stats import resolve_wavelength_stats
from app.foundation_models.components.token_masking import TokenMasking
from app.foundation_models.trainers.hyperspectral_segformer_mae_trainer import (
    MIN_VALID_PIXEL_FRACTION, STAGE1_KERNEL_SIZE, STAGE1_PADDING, STAGE1_STRIDE,
    HyperspectralSegFormerMAETrainer)

logger = logging.getLogger("SpectralStemMAETrainer")

# What the validation score is; stored with the early-stop state so a checkpoint scored another way
# cannot seed it (E003's epochs 1-3 were scored with the ramped SAM weight).
VAL_SCORE = "l1+full_sam"

# SpectralEarth's encoding (docs/lld/spectralearth.md): reflectance x 10,000 as int16.
REFLECTANCE_SCALE, NODATA = 0.0001, -32768


METRICS_COLUMNS = ("epoch", "train_l1", "train_sam_deg", "val_l1", "val_sam_deg", "sam_weight", "lr",
                   "train_patches", "seconds", "patches_per_s", "peak_gpu_gb")


def decode_i16dz_member(key: str, data: bytes):
    """webdataset handler: `.pixels.i16dz` (band-delta + zstd, app/utils/shard_codec.py) -> int16
    ndarray. Module-level so DataLoader workers can pickle it on spawn (Windows, macOS)."""
    if key.endswith(".i16dz"):
        from app.utils.shard_codec import decode_i16dz
        return decode_i16dz(data)
    return None


class _EarlyStop(Exception):
    """Raised after the epoch whose checkpoint met the early-stop rule; caught by train()."""


class SpectralStemMAETrainer(HyperspectralSegFormerMAETrainer):

    def __init__(self, config):
        self._val_l1_sum, self._val_sam_sum, self._val_samples = 0.0, 0.0, 0
        # Early-stop state; saved in every checkpoint so a resumed run remembers it.
        self._best_val, self._bad_epochs, self._lowest_val, self._stop = math.inf, 0, math.inf, False
        super().__init__(config)                 # loads config.resume_from, if set
        cfg = config.model_config_
        if cfg.auto_resume and config.resume_from is None:
            path = self._find_resume_checkpoint()
            if path is not None:
                self._load_checkpoint(str(path))
            else:
                logger.info("auto_resume: no last.pt locally or in S3 - starting fresh")

    def train(self) -> None:
        try:
            super().train()
        except _EarlyStop as stop:
            logger.info(f"Early stop: {stop}")

    # --- checkpoints: last.pt + best.pt, mirrored to S3, carrying early-stop state ---------

    def _s3_target(self) -> tuple[str, str] | None:
        uri = self.config.model_config_.checkpoint_s3_uri
        if not uri:
            return None
        bucket, _, prefix = uri.removeprefix("s3://").partition("/")
        return bucket, prefix.rstrip("/")

    def _find_resume_checkpoint(self) -> Path | None:
        local = Path(self.config.checkpoint.checkpoint_dir) / "last.pt"
        if local.exists():
            return local
        target = self._s3_target()
        if target is None:
            return None
        from app.utils.general_utils import s3_config
        client, (bucket, prefix) = s3_config.client(), target
        found = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/last.pt").get("KeyCount", 0)
        if not found:
            return None
        local.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, f"{prefix}/last.pt", str(local))
        logger.info(f"auto_resume: fetched s3://{bucket}/{prefix}/last.pt")
        return local

    def _load_checkpoint(self, path: str) -> None:
        super()._load_checkpoint(path)
        state = torch.load(path, map_location="cpu", weights_only=False).get("early_stop")
        if state and state.get("score") != VAL_SCORE:
            logger.info(f"  early-stop state ignored: scored as {state.get('score', 'l1+ramped_sam')}, now {VAL_SCORE}")
            state = None
        if state and self.config.resume_mode == "resume":
            self._best_val, self._bad_epochs, self._lowest_val = state["best_val"], state["bad_epochs"], state["lowest_val"]
            logger.info(f"  early-stop state: best {self._best_val:.6f}, {self._bad_epochs} flat epochs")

    def _update_early_stop(self, epoch: int, avg_val: float) -> None:
        """An epoch counts as improvement only if it beats the best by early_stop_min_delta
        (relative); smaller gains do not reset the count - otherwise a slow creep never stops."""
        cfg = self.config.model_config_
        if self._best_val == math.inf or avg_val < self._best_val * (1 - cfg.early_stop_min_delta):
            self._best_val, self._bad_epochs = avg_val, 0
        else:
            self._bad_epochs += 1
        self._stop = bool(cfg.early_stop_patience) and epoch >= cfg.early_stop_min_epochs             and self._bad_epochs >= cfg.early_stop_patience

    def _save_checkpoint(self, epoch: int, train_loss: float, val_losses: dict[int, float]) -> None:
        avg_val = sum(val_losses.values()) / len(val_losses)
        self._update_early_stop(epoch, avg_val)
        super()._save_checkpoint(epoch, train_loss, val_losses)
        ckpt_dir = Path(self.config.checkpoint.checkpoint_dir)
        ckpt = torch.load(ckpt_dir / f"{self.config.foundation_model_name.value}_v{self.config.version}_epoch{epoch}.pt",
                          map_location="cpu", weights_only=False)
        ckpt["early_stop"] = {"score": VAL_SCORE, "best_val": self._best_val, "bad_epochs": self._bad_epochs,
                              "lowest_val": min(self._lowest_val, avg_val), "stopped": self._stop}
        names = ["last.pt"]
        if avg_val < self._lowest_val:
            self._lowest_val = avg_val
            names.append("best.pt")
        target = self._s3_target()
        client = None
        if target is not None:
            from app.utils.general_utils import s3_config
            client = s3_config.client()
        for name in names:
            # Write-then-rename: an instance killed mid-write never leaves a truncated last.pt.
            tmp = ckpt_dir / f".{name}.part"
            torch.save(ckpt, tmp)
            os.replace(tmp, ckpt_dir / name)
            if client is not None:
                client.upload_file(str(ckpt_dir / name), target[0], f"{target[1]}/{name}")
        logger.info(f"  saved {', '.join(names)}" + (f" -> s3://{target[0]}/{target[1]}/" if target else "")
                    + f" | early stop: best {self._best_val:.6f}, flat {self._bad_epochs}")

    def _run_epoch(self, epoch, train_loaders, test_loaders):
        """The parent's epoch, plus one metrics.csv row: the experiment's record of the run."""
        self._val_l1_sum, self._val_sam_sum, self._val_samples = 0.0, 0.0, 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.time()
        super()._run_epoch(epoch, train_loaders, test_loaders)
        path = self.config.model_config_.metrics_csv
        if path is None:
            if self._stop:
                raise _EarlyStop(f"best avg val {self._best_val:.6f} (best.pt)")
            return
        seconds, n = time.time() - start, self._epoch_loss_samples
        deg = 180 / math.pi
        row = {
            "epoch": epoch + 1,
            "train_l1": self._epoch_l1_sum / max(n, 1), "train_sam_deg": self._epoch_sam_sum / max(n, 1) * deg,
            "val_l1": self._val_l1_sum / max(self._val_samples, 1),
            "val_sam_deg": self._val_sam_sum / max(self._val_samples, 1) * deg,
            "sam_weight": self._get_sam_weight(epoch), "lr": self.optimizer.param_groups[0]["lr"],
            "train_patches": n, "seconds": round(seconds, 1), "patches_per_s": round(n / seconds, 2),
            "peak_gpu_gb": (round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2)
                            if self.device.type == "cuda" else ""),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        new = not Path(path).exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        if self._stop:
            raise _EarlyStop(f"no {self.config.model_config_.early_stop_min_delta:.0%} improvement in "
                             f"{self._bad_epochs} epochs; best avg val {self._best_val:.6f} (best.pt)")

    def _sample_decoders(self) -> list:
        return [decode_i16dz_member]

    def build_model(self) -> nn.Module:
        cfg = self.config.model_config_
        wl, mean, std = resolve_wavelength_stats(self.config.data.pixel_stats_path)
        self._sam_loss = SAMLoss()
        return SpectralStemSegFormerMAE(
            stem=cfg.stem, stats_wavelengths_nm=wl, stats_mean=mean, stats_std=std,
            bands=cfg.bands, compressed_channels=cfg.compressed_channels, token_dim=cfg.token_dim,
            attention_heads=cfg.attention_heads, gate_hidden=cfg.gate_hidden, bin_nm=cfg.bin_nm,
            embed_dims=cfg.embed_dims, num_heads=cfg.num_heads, reduction_ratios=cfg.reduction_ratios,
            num_blocks=cfg.num_blocks, decoder_dim=cfg.decoder_dim,
            expansion_ratio=cfg.expansion_ratio, drop_rate=cfg.drop_rate)

    def _decode_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (reflectance (B, N, H, W) float32, band_valid (B, N, H, W) bool,
        wavelengths (N,) or (B, N), pixel mask (B, 1, H, W) float)."""
        # Raw shards carry pixels.npy; recompressed ones pixels.i16dz, already decoded to the
        # same int16 array by decode_i16dz_member in the workers.
        raw = torch.as_tensor(batch["pixels.npy"] if "pixels.npy" in batch else batch["pixels.i16dz"]).to(self.device)
        band_valid = raw != NODATA
        pixels = torch.where(band_valid, raw.float() * REFLECTANCE_SCALE, torch.zeros((), device=self.device))
        wl = torch.as_tensor(batch["wavelengths.npy"]).to(self.device).float()      # (B, N)
        if (wl == wl[:1]).all():
            wl = wl[0]
        mask = band_valid.any(dim=1, keepdim=True).float()
        return pixels, band_valid, wl, mask

    def _forward(self, model, pixels, wl, band_valid, keep_mask) -> torch.Tensor:
        """The model under bf16 autocast when amp_bf16 is set (CUDA only); returns fp32 so the
        loss - small differences of reflectance - is always computed in full precision."""
        bf16 = self.config.model_config_.amp_bf16 and self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(pixels, wl, band_valid, keep_mask)
        return out.float()

    def _keep_samples(self, pixels, band_valid, wl, mask):
        """Drop samples below MIN_VALID_PIXEL_FRACTION, keeping all four tensors aligned."""
        keep = mask.flatten(1).mean(dim=1) >= MIN_VALID_PIXEL_FRACTION
        wl = wl if wl.dim() == 1 else wl[keep]
        return pixels[keep], band_valid[keep], wl, mask[keep], int(keep.sum())

    def _masked_l1_sam(self, x_hat, pixels, band_valid, loss_mask) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and SAM over valid bands only, averaged over pixels where loss_mask (B, 1, H, W)
        is 1. A dead band has no truth to compare with, so it adds nothing to either term."""
        bv = band_valid.float()
        per_pixel_l1 = ((x_hat - pixels).abs() * bv).sum(1, keepdim=True) / bv.sum(1, keepdim=True).clamp(min=1)
        l1 = (per_pixel_l1 * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        # Zeroing dead bands in both spectra removes them from SAM's dot product and norms.
        sam = self._sam_loss(x_hat * bv, pixels * bv, loss_mask)
        return l1, sam

    def compute_loss(self, batch: dict, model: nn.Module) -> tuple[torch.Tensor, int]:
        """L1 + lambda(t) * SAM on hidden, valid, eroded pixels and valid bands."""
        cfg = self.config.model_config_
        pixels, band_valid, wl, mask, n = self._keep_samples(*self._decode_batch(batch))
        if n == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        keep_mask, pred_mask = self._build_prediction_mask(mask, cfg.mask_ratio)
        x_hat = self._forward(model, pixels, wl, band_valid, keep_mask)
        loss_mask = (self._pred_mask_to_pixel_mask(pred_mask, *pixels.shape[2:])
                     * TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size))
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        l1, sam = self._masked_l1_sam(x_hat, pixels, band_valid, loss_mask)
        self._epoch_l1_sum += l1.item() * n
        self._epoch_sam_sum += sam.item() * n
        self._epoch_loss_samples += n
        return l1 + self._get_sam_weight(self._current_epoch) * sam, n

    def compute_validation_loss(self, batch: dict, model: nn.Module) -> tuple[torch.Tensor, int]:
        """Two complementary random token masks, so every valid pixel is scored once while
        hidden - the parent's regime, with valid-band L1 and SAM. Also accumulates the
        validation L1 and SAM that metrics.csv reports."""
        cfg = self.config.model_config_
        pixels, band_valid, wl, mask, n = self._keep_samples(*self._decode_batch(batch))
        if n == 0:
            return torch.tensor(0.0, device=self.device), 0
        H, W = pixels.shape[2:]
        token_valid = TokenMasking.pixel_mask_to_token_mask(
            mask, kernel_size=STAGE1_KERNEL_SIZE, stride=STAGE1_STRIDE, padding=STAGE1_PADDING)
        half = (torch.rand(1, token_valid.shape[1], device=self.device) > 0.5).float()
        x_hat = torch.zeros_like(pixels)
        for hidden in (token_valid * (1.0 - half), token_valid * half):
            x_hat = x_hat + self._forward(model, pixels, wl, band_valid, 1.0 - hidden) * self._pred_mask_to_pixel_mask(hidden, H, W)
        loss_mask = TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size)
        l1, sam = self._masked_l1_sam(x_hat, pixels, band_valid, loss_mask)
        self._val_l1_sum += l1.item() * n
        self._val_sam_sum += sam.item() * n
        self._val_samples += n
        # Full SAM weight, not the training ramp: the validation score ranks epochs for early
        # stopping and best.pt, so it must be on one scale. With the ramp, epoch 1 (lambda 0) scored
        # ~0.055 and every later epoch ~0.1 - early stop would have fired at the first allowed epoch
        # whatever the model did (found in E003 on 2026-10-14; the trial's lost checkpoint, again).
        return l1 + cfg.sam_weight * sam, n
