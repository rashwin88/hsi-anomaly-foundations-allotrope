"""
Fine-tuning a pretrained spectral-stem model on several sensors at once, with a segmentation head
(docs/proposals/fine-tuning-and-segmentation.md).

Registered in trainer_factory for `spectral_stem_finetune`. Extends SpectralStemMAETrainer and keeps
its reconstruction loss (MAE token masking, band-list augmentation, L1 + ramped SAM), validation at
full SAM weight, early stopping, checkpoints and auto-resume. What it adds:

  - **Sources.** One WebDataset pipeline per sensor (`model_config.sources`), each reading its own staged
    shard folders, aligned onto its own band list, EnMAP's label layers turned into segmentation targets
    (app/utils/segmentation/targets.py). Every training batch comes from one sensor - the stem takes one
    band list per batch - drawn at random by the sources' shares.
  - **Modes.** probe: the segmentation head alone, backbone frozen (its BatchNorm statistics too).
    adapt: reconstruction only. joint: reconstruction + seg_weight x segmentation loss on labelled batches.
  - **Start point.** `init_from` loads a pretrained checkpoint's weights (the head starts fresh). The model's
    normalisation curve comes from that checkpoint - frozen at pretraining, never recomputed.
  - **Trained-wavelength record.** Each new band list a batch carries is added to the model's record, so
    `trained_band_mask` accepts what fine-tuning has taught.
  - **Records.** Besides metrics.csv, `finetune_metrics_csv` gets one row per epoch: validation L1 / SAM per
    sensor and per-class IoU on labelled test data.

Gotchas:
  - Validation runs every test source to exhaustion, in order; `test_samples_per_epoch` caps the total.
  - The epoch's training budget is `train_samples_per_epoch[128]` samples across all sources together.
  - meta.json and label_* members are consumed in the loader workers and dropped before collation (nested,
    sometimes null metadata does not collate).
"""

import csv
import functools
import json
import logging
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import webdataset as wds

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.trainers.hyperspectral_segformer_mae_trainer import MIN_VALID_PIXEL_FRACTION
from app.foundation_models.trainers.spectral_stem_mae_trainer import (
    NODATA, SpectralStemMAETrainer, TokenMasking, align_sample, sample_band_subset)
from app.utils.segmentation.targets import CLASSES, IoUTally, LabelPolicy, enmap_targets, masked_bce_dice

logger = logging.getLogger("FinetuneTrainer")
LABEL_KEYS = ("label_cloud", "label_cirrus", "label_haze", "label_cloud_shadow", "label_snow", "label_classes")


def prepare_sample(sample: dict, target_nm=None, labels: str | None = None, policy: LabelPolicy = LabelPolicy()) -> dict:
    """Loader-worker transform: align onto a band list (optional), turn EnMAP labels into seg_targets /
    seg_known, drop meta.json and label_* so batches collate. Module-level so workers can pickle it."""
    if target_nm is not None:
        sample = align_sample(sample, target_nm)
    meta = sample.pop("meta.json", None) or {}
    if labels == "enmap":
        t, k = enmap_targets({key: sample[f"{key}.npy"] for key in LABEL_KEYS}, meta, policy)
        sample["seg_targets.npy"], sample["seg_known.npy"] = t, k
    for key in LABEL_KEYS:
        sample.pop(f"{key}.npy", None)
    return sample


def gradient_l1(x_hat: torch.Tensor, x: torch.Tensor, band_valid: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Edge term: mean |d x_hat - d x| over horizontal and vertical neighbour differences, per band, on pairs whose
    two pixels are both in loss_mask (B, 1, H, W) and valid in that band. Blur shows up as a gradient mismatch."""
    total, count = x.new_zeros(()), x.new_zeros(())
    for dim in (2, 3):
        d_hat, d = torch.diff(x_hat, dim=dim), torch.diff(x, dim=dim)
        ok = (torch.diff(loss_mask, dim=dim) == 0) * loss_mask.narrow(dim, 1, x.shape[dim] - 1)   # both in mask
        ok = ok * band_valid.narrow(dim, 1, x.shape[dim] - 1) * band_valid.narrow(dim, 0, x.shape[dim] - 1)
        total, count = total + ((d_hat - d).abs() * ok).sum(), count + ok.sum()
    return total / count.clamp(min=1)


def keep_sample(sample: dict, skip: frozenset = frozenset()) -> bool:
    """Loader-worker filter: drop samples of skipped scenes (meta.json scene_id)."""
    return not skip or (sample.get("meta.json") or {}).get("scene_id") not in skip


class _MixedLoader:
    """Endless batches drawn from several loaders by weight; each batch is tagged with its source."""

    def __init__(self, loaders: dict, shares: dict, seed: int = 0):
        self.loaders, self.names = loaders, list(loaders)
        self.weights = [shares[n] for n in self.names]
        self.rng, self.iters = random.Random(seed), {}

    def __iter__(self):
        while True:
            name = self.rng.choices(self.names, self.weights)[0]
            if name not in self.iters:
                self.iters[name] = iter(self.loaders[name])
            try:
                batch = next(self.iters[name])
            except StopIteration:                       # that sensor's shards are done: start them again
                self.iters[name] = iter(self.loaders[name])
                batch = next(self.iters[name])
            batch["_source"] = name
            yield batch


class _AllSourcesLoader:
    """Endless mixed batches: one sub-batch from every loader per step, as {"_groups": [(name, batch), ...]}."""

    def __init__(self, loaders: dict):
        self.loaders, self.iters = loaders, {}

    def __iter__(self):
        while True:
            groups = []
            for name, loader in self.loaders.items():
                if name not in self.iters:
                    self.iters[name] = iter(loader)
                try:
                    batch = next(self.iters[name])
                except StopIteration:                   # that sensor's shards are done: start them again
                    self.iters[name] = iter(loader)
                    batch = next(self.iters[name])
                groups.append((name, batch))
            yield {"_groups": groups}


class _SequentialLoader:
    """Every batch of every loader once, in order, tagged with its source (validation)."""

    def __init__(self, loaders: dict):
        self.loaders = loaders

    def __iter__(self):
        for name, loader in self.loaders.items():
            for batch in loader:
                batch["_source"] = name
                yield batch


class FinetuneTrainer(SpectralStemMAETrainer):

    def __init__(self, config):
        self._seen_band_lists: set = set()
        self._val_by_source: dict = {}
        self._iou = IoUTally()
        super().__init__(config)
        cfg = config.model_config_
        if cfg.mode == "probe":                          # only the head learns
            for name, p in self.model.named_parameters():
                p.requires_grad = name.startswith("seg_head.")
            self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config.learning_rate)
            self.scheduler = self._build_scheduler()
        self._policy = LabelPolicy(cfg.plateau_elevation_m, cfg.low_water_vapour, cfg.cirrus_min_level)

    # --- model ---------------------------------------------------------------------------------

    def build_model(self) -> nn.Module:
        cfg = self.config.model_config_
        state = self._load_init_state(cfg.init_from) if cfg.init_from else None
        if state is not None:                            # the frozen curve travels with the weights
            wl, mean, std = (state[f"stats.{k}"].tolist() for k in ("wl", "mean", "std"))
        else:
            from app.foundation_models.pixel_stats import resolve_wavelength_stats
            wl, mean, std = resolve_wavelength_stats(self.config.data.pixel_stats_path)
        from app.foundation_models.components.sam_loss import SAMLoss
        self._sam_loss = SAMLoss()
        model = SpectralStemSegFormerMAE(
            stem=cfg.stem, stats_wavelengths_nm=wl, stats_mean=mean, stats_std=std, bands=cfg.bands,
            compressed_channels=cfg.compressed_channels, token_dim=cfg.token_dim, attention_heads=cfg.attention_heads,
            gate_hidden=cfg.gate_hidden, bin_nm=cfg.bin_nm, band_weighting=cfg.band_weighting,
            seg_classes=cfg.seg_classes, seg_decoder_dim=cfg.seg_decoder_dim, embed_dims=cfg.embed_dims,
            num_heads=cfg.num_heads, reduction_ratios=cfg.reduction_ratios, num_blocks=cfg.num_blocks,
            decoder_dim=cfg.decoder_dim, expansion_ratio=cfg.expansion_ratio, drop_rate=cfg.drop_rate)
        if state is not None:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or any(not k.startswith("seg_head.") for k in missing):
                raise ValueError(f"init_from does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
            logger.info(f"init_from {cfg.init_from}: loaded; {len(missing)} new segmentation-head tensors")
        return model

    def _load_init_state(self, uri: str) -> dict:
        path = Path(uri)
        if uri.startswith("s3://"):
            from app.utils.general_utils import s3_config
            bucket, _, key = uri.removeprefix("s3://").partition("/")
            path = Path(self.config.checkpoint.checkpoint_dir) / "init.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                s3_config.client().download_file(bucket, key, str(path))
        return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]

    # --- data ----------------------------------------------------------------------------------

    def _source_loader(self, src, split: str, batch_size: int | None = None):
        from app.utils.band_alignment import load_band_list
        folder = Path(src.train_dir if split == "train" else src.test_dir)
        shards = sorted(folder.glob("*.tar"))
        if not shards:
            raise FileNotFoundError(f"{src.name}: no shards in {folder}")
        target = load_band_list(src.band_list_path) if src.band_list_path else None
        transform = functools.partial(prepare_sample, target_nm=target, labels=src.labels, policy=self._policy)
        skip = frozenset(json.loads(Path(src.skip_scenes_path).read_text())) if src.skip_scenes_path else frozenset()
        urls = [f"file:{p.as_posix()}" if os.name == "nt" else str(p) for p in shards]   # as the base trainer
        dataset = (wds.WebDataset(urls, shardshuffle=self.config.data.shardshuffle if split == "train" else False,
                                  handler=wds.warn_and_continue)
                   .decode(*self._sample_decoders()).select(functools.partial(keep_sample, skip=skip)).map(transform))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size or self.config.data.batch_size,
                                           num_workers=self.config.data.num_workers)

    def _train_with_hot_storage(self) -> None:
        cfg = self.config.model_config_
        if cfg.mixed_batches:                            # every batch holds every sensor, share x batch_size each
            size = self.config.data.batch_size
            train = _AllSourcesLoader({s.name: self._source_loader(s, "train", max(1, round(s.share * size)))
                                       for s in cfg.sources})
        else:
            train = _MixedLoader({s.name: self._source_loader(s, "train") for s in cfg.sources},
                                 {s.name: s.share for s in cfg.sources}, seed=self.start_epoch)
        test = _SequentialLoader({s.name: self._source_loader(s, "test") for s in cfg.sources})
        sizes = self.config.data.patch_sizes
        for epoch in range(self.start_epoch, self.config.data.num_epochs):
            self._run_epoch(epoch, {sizes[0]: train}, {sizes[0]: test})
        logger.info("Fine-tuning complete.")

    # --- losses --------------------------------------------------------------------------------

    def _note_band_list(self, wl: torch.Tensor) -> None:
        key = tuple(round(float(v), 1) for v in wl.flatten()[:512])
        if key not in self._seen_band_lists:
            self._seen_band_lists.add(key)
            added = self.model.extend_trained_wavelengths(wl)
            if added:
                logger.info(f"trained-wavelength record +{added} ({len(self.model.trained_wl)} total)")

    def _seg_batch(self, batch: dict):
        """(pixels, band_valid, wl, targets, known) for the batch's usable samples, or None if unlabelled."""
        if "seg_targets.npy" not in batch or self.model.seg_head is None:
            return None
        pixels, band_valid, wl, mask = self._decode_batch(batch)
        keep = mask.flatten(1).mean(dim=1) >= MIN_VALID_PIXEL_FRACTION
        if not keep.any():
            return None
        t = torch.as_tensor(batch["seg_targets.npy"]).to(self.device)[keep]
        k = torch.as_tensor(batch["seg_known.npy"]).to(self.device).bool()[keep] & mask[keep].bool()
        return pixels[keep], band_valid[keep], (wl if wl.dim() == 1 else wl[keep]), t, k

    def _segment(self, pixels, band_valid, wl):
        bf16 = self.config.model_config_.amp_bf16 and self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            return self.model.segment(pixels, wl, band_valid).float()

    def compute_loss(self, batch: dict, model: nn.Module):
        cfg = self.config.model_config_
        if "_groups" in batch:
            return self._mixed_loss(batch["_groups"], model)
        wl = torch.as_tensor(batch["wavelengths.npy"])
        self._note_band_list(wl[0] if wl.dim() == 2 else wl)
        seg = self._seg_batch(batch) if cfg.mode in ("probe", "joint") else None
        if cfg.mode == "probe":
            if seg is None:
                return torch.tensor(0.0, device=self.device, requires_grad=True), 0
            pixels, bv, w, t, k = seg
            return masked_bce_dice(self._segment(pixels, bv, w), t, k), len(pixels)
        loss, n = super().compute_loss(batch, model)
        if seg is not None and n:
            pixels, bv, w, t, k = seg
            loss = loss + cfg.seg_weight * masked_bce_dice(self._segment(pixels, bv, w), t, k)
        return loss, n

    def _mixed_loss(self, groups: list, model: nn.Module):
        """The parent's reconstruction loss for a mixed batch: per sensor, its own MAE mask and band-list
        augmentation; one shared pass (forward_mixed); L1 + ramped SAM per sensor, weighted by its samples."""
        cfg = self.config.model_config_
        inputs, truth = [], []
        for _, batch in groups:
            wl = torch.as_tensor(batch["wavelengths.npy"])
            self._note_band_list(wl[0] if wl.dim() == 2 else wl)
            pixels, band_valid, wl, mask, n = self._keep_samples(*self._decode_batch(batch))
            if n == 0:
                continue
            if wl.dim() != 1:
                raise ValueError("mixed batches need one band list per source - set band_list_path")
            keep_mask, pred_mask = self._build_prediction_mask(mask, cfg.mask_ratio)
            x, wl_in, bv, out_wl = pixels, wl, band_valid, None
            if cfg.band_augment_prob > 0 and torch.rand(()).item() < cfg.band_augment_prob:
                idx = sample_band_subset(wl)
                wl_in = wl[idx] + (torch.rand(len(idx), device=wl.device) * 2 - 1) * cfg.band_jitter_nm
                x, bv, out_wl = pixels[:, idx], band_valid[:, idx], wl
            loss_mask = (self._pred_mask_to_pixel_mask(pred_mask, *pixels.shape[2:])
                         * TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size))
            inputs.append((x, wl_in, bv, keep_mask, out_wl))
            truth.append((pixels, band_valid, loss_mask, n))
        if not inputs:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        bf16 = cfg.amp_bf16 and self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            outs = model.forward_mixed(inputs)
        total, kept = 0.0, 0
        for x_hat, (pixels, band_valid, loss_mask, n) in zip(outs, truth):
            if loss_mask.sum() == 0:
                continue
            l1, sam = self._masked_l1_sam(x_hat.float(), pixels, band_valid, loss_mask)
            edge = 0.0
            if cfg.grad_loss_weight:                     # kept out of train L1 / metrics.csv; own column
                grad = gradient_l1(x_hat.float(), pixels, band_valid, loss_mask)
                self._epoch_grad_sum, self._epoch_grad_n = self._epoch_grad_sum + grad.item() * n, self._epoch_grad_n + n
                edge = cfg.grad_loss_weight * grad
            self._epoch_l1_sum += l1.item() * n
            self._epoch_sam_sum += sam.item() * n
            self._epoch_loss_samples += n
            total, kept = total + (l1 + self._get_sam_weight(self._current_epoch) * sam + edge) * n, kept + n
        if not kept:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        return total / kept, kept

    def compute_validation_loss(self, batch: dict, model: nn.Module):
        cfg = self.config.model_config_
        source = batch.get("_source", "?")
        seg = self._seg_batch(batch)
        seg_loss = None
        if seg is not None:
            pixels, bv, w, t, k = seg
            logits = self._segment(pixels, bv, w)
            self._iou.add(logits, t, k)
            seg_loss = masked_bce_dice(logits, t, k)
        if cfg.mode == "probe":
            n = 0 if seg is None else len(seg[0])
            return (seg_loss if seg_loss is not None else torch.tensor(0.0, device=self.device)), n
        l1_before, sam_before, n_before = self._val_l1_sum, self._val_sam_sum, self._val_samples
        loss, n = super().compute_validation_loss(batch, model)
        acc = self._val_by_source.setdefault(source, [0.0, 0.0, 0])
        acc[0] += self._val_l1_sum - l1_before; acc[1] += self._val_sam_sum - sam_before; acc[2] += self._val_samples - n_before
        return loss, n

    # --- per-epoch records -----------------------------------------------------------------------

    def _train_pass_mode(self) -> None:
        if self.config.model_config_.mode == "probe":    # frozen backbone: keep its BatchNorm statistics
            self.model.eval()
            self.model.seg_head.train()

    def _run_train_pass(self, loader, sample_cap: int):
        self._train_pass_mode()
        return super()._run_train_pass(loader, sample_cap)

    def _run_epoch(self, epoch, train_loaders, test_loaders):
        self._val_by_source, self._iou = {}, IoUTally()
        self._epoch_grad_sum, self._epoch_grad_n = 0.0, 0
        try:
            super()._run_epoch(epoch, train_loaders, test_loaders)
        finally:                                         # also when early stopping ends the run
            self._write_finetune_row(epoch)

    def _write_finetune_row(self, epoch: int) -> None:
        path = self.config.model_config_.finetune_metrics_csv
        if path is None:
            return
        deg = 180 / math.pi
        row = {"epoch": epoch + 1, "trained_wavelengths": len(self.model.trained_wl)}
        if self.config.model_config_.grad_loss_weight:
            row["train_grad_l1"] = round(self._epoch_grad_sum / max(self._epoch_grad_n, 1), 6)
        for s in self.config.model_config_.sources:
            l1, sam, n = self._val_by_source.get(s.name, [0.0, 0.0, 0])
            row[f"{s.name}_val_l1"] = round(l1 / n, 6) if n else ""
            row[f"{s.name}_val_sam_deg"] = round(sam / n * deg, 4) if n else ""
        ious = self._iou.iou()
        row.update({f"iou_{c}": ("" if ious[c] is None else round(ious[c], 4)) for c in CLASSES})
        known = [v for v in ious.values() if v is not None]
        row["iou_mean"] = round(sum(known) / len(known), 4) if known else ""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        new = not Path(path).exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        logger.info("finetune: " + ", ".join(f"{k} {v}" for k, v in row.items() if k != "epoch"))
