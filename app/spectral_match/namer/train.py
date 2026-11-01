"""
Training the spectral namer (docs/proposals/spectral-namer.md): real tiles from each sensor's shards, planted on the
GPU (generator.plant), one loss per level.

    loss = BCE(man-made logit, man-made)                        every sample
         + CE(class scores, class)                               man-made samples, own spectrum left out
         + fill_weight * L1(fill, a)                             every sample

The class scores read the library's train-split embeddings (test spectra and an open-set class never reach the loss),
with each sample's own planted spectrum left out, as in the separability check's leave-one-out - so the model is
rewarded for naming a class, not for recognising one spectrum.

Standalone loop rather than a FinetuneTrainer subclass: the reconstruction trainer's epoch, metrics and checkpoint
machinery assume a reconstruction output. The shard pipeline is the fine-tuner's (prepare_sample, decode_i16dz_member).

Config (JSON): see configs/namer/N001.json. Records: metrics.csv per epoch (train loss, per-sensor level-1 and
level-2 accuracy at each evaluation fill, open-set rates), last.pt / best.pt (best on mean level-2 accuracy at
fill 0.25).
"""

from __future__ import annotations

import csv
import functools
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from app.spectral_match.namer.bank import build_bank
from app.spectral_match.namer.classes import MANMADE_CLASSES
from app.spectral_match.namer.generator import plant
from app.spectral_match.namer.model import SpectralNamer

logger = logging.getLogger("namer")
C = len(MANMADE_CLASSES)


def _loader(src: dict, split: str, cfg: dict):
    import webdataset as wds
    from app.foundation_models.trainers.finetune_trainer import prepare_sample, single_thread_worker
    from app.foundation_models.trainers.spectral_stem_mae_trainer import decode_i16dz_member
    from app.utils.band_alignment import load_band_list
    shards = sorted(Path(src[f"{split}_dir"]).glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"{src['name']}: no shards in {src[f'{split}_dir']}")
    target = load_band_list(src["band_list_path"]) if src.get("band_list_path") else None
    urls = [f"file:{p.as_posix()}" if os.name == "nt" else str(p) for p in shards]
    ds = (wds.WebDataset(urls, shardshuffle=100 if split == "train" else False, handler=wds.warn_and_continue)
          .decode(decode_i16dz_member).map(functools.partial(prepare_sample, target_nm=target)))
    if split == "train":
        ds = ds.shuffle(64)
    return torch.utils.data.DataLoader(ds, batch_size=cfg["tiles_per_batch"], num_workers=cfg["loader_workers"],
                                       worker_init_fn=single_thread_worker)


def _endless(loader):
    while True:
        yield from loader


def decode(batch: dict, device):
    """(reflectance (B, N, H, W), band_valid, wavelengths (N,)) - a per-sample list is averaged (PRISMA's
    per-scene centres differ by ~1 nm; the bank and the stem read one list per batch)."""
    raw = torch.as_tensor(batch["pixels.npy"] if "pixels.npy" in batch else batch["pixels.i16dz"]).to(device)
    valid = raw != -32768
    pixels = torch.where(valid, raw.float() * 1e-4, torch.zeros((), device=device))
    wl = torch.as_tensor(batch["wavelengths.npy"]).to(device).float()
    return pixels, valid, (wl.mean(0) if wl.dim() == 2 else wl)


class NamerRun:
    def __init__(self, cfg: dict):
        self.cfg, self.out = cfg, Path(cfg["run_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)["model_state_dict"]
        self.model = SpectralNamer.from_spectral_stem(state, **cfg.get("model", {})).to(self.device)
        stem = [p for n, p in self.model.named_parameters() if n.startswith(("stem.", "norm."))]
        rest = [p for n, p in self.model.named_parameters() if not n.startswith(("stem.", "norm."))]
        self.opt = torch.optim.AdamW([{"params": stem, "lr": cfg["lr"] * cfg["stem_lr_scale"]},
                                      {"params": rest, "lr": cfg["lr"]}], weight_decay=cfg["weight_decay"])
        total, warm = cfg["epochs"] * cfg["steps_per_epoch"], cfg["warmup_steps"]
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / total))))
        self.banks, self.bank_wl = {}, {}
        self.start_epoch, self.best = 0, -1.0
        if (self.out / "last.pt").exists():
            ck = torch.load(self.out / "last.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(ck["model"]); self.opt.load_state_dict(ck["opt"])
            self.sched.load_state_dict(ck["sched"]); self.start_epoch, self.best = ck["epoch"] + 1, ck["best"]
            logger.info(f"resumed at epoch {self.start_epoch}")

    def bank(self, name: str, wl: torch.Tensor):
        """One bank per sensor, on its first batch's band list (later batches of the sensor read the same list)."""
        if name not in self.banks:
            self.banks[name] = build_bank(wl.cpu().numpy(), self.out / "lib_cache", self.cfg["library_root"],
                                          self.cfg.get("holdout_class")).to(self.device)
            self.bank_wl[name] = wl
            b = self.banks[name]
            logger.info(f"{name}: bank {len(b.names)} spectra on {len(wl)} bands, {int(b.manmade.sum())} man-made, "
                        f"{int(b.test.sum())} test")
        return self.banks[name], self.bank_wl[name]

    def _amp(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    def forward(self, name, batch, split, fixed_fill=None, kinds=None, generator=None):
        pixels, valid, wl = decode(batch, self.device)
        bank, wl = self.bank(name, wl)
        if pixels.shape[1] != len(wl):
            return None
        keep = valid.float().mean(dim=(0, 2, 3)) > 0.5               # the batch's live bands (no water bands)
        if int(keep.sum()) < 20:
            return None
        pixels, valid, wl, bank = pixels[:, keep], valid[:, keep], wl[keep], bank.bands(keep)
        s = plant(pixels, valid, bank, self.cfg["pixels_per_tile"], self.cfg["a_min"], self.cfg["gain"],
                  kinds or self.cfg["kinds"], split, fixed_fill, generator, wl, self.cfg.get("lib_aug"))
        if s is None:
            return None
        train_lib = bank.manmade & ~bank.test
        if bank.holdout_class >= 0:
            train_lib &= bank.cls != bank.holdout_class
        lib_cls = torch.where(train_lib, bank.cls, torch.full_like(bank.cls, -1))
        with self._amp():
            z, logit, fill = self.model.forward_pixel(s["x"], s["r"], wl)
            lib = bank.refl
            la = self.cfg.get("lib_aug") or {}
            if split == "train" and la.get("library_side"):            # N003: the library tower sees variants too
                from app.spectral_match.namer.generator import augment_spectra
                lib = augment_spectra(lib, wl, la)
            e = self.model.forward_library(lib, wl)
            if self.cfg.get("level1", "head") == "matching":       # natural entries as one more class
                lib_cls = torch.where(bank.manmade, lib_cls, torch.full_like(lib_cls, C))
                scores = self.model.class_scores(z.float(), e.float(), lib_cls, C + 1, exclude=s["entry"])
                logit, scores = self.model.level1_logit(scores), scores[:, :C]
            else:
                scores = self.model.class_scores(z.float(), e.float(), lib_cls, C, exclude=s["entry"])
        return s, logit.float(), fill.float(), scores

    def loss(self, s, logit, fill, scores):
        l1 = F.binary_cross_entropy_with_logits(logit, s["manmade"])
        mm = s["cls"] >= 0
        l2 = F.cross_entropy(scores[mm], s["cls"][mm]) if mm.any() else logit.sum() * 0
        lf = (fill - s["fill"]).abs().mean()
        return l1 + l2 + self.cfg["fill_weight"] * lf, (l1.item(), l2.item(), lf.item())

    def train(self):
        srcs = self.cfg["sources"]
        train = {s["name"]: _endless(_loader(s, "train", self.cfg)) for s in srcs}
        for epoch in range(self.start_epoch, self.cfg["epochs"]):
            self.model.train()
            t0, sums, n = time.time(), [0.0, 0.0, 0.0, 0.0], 0
            for step in range(self.cfg["steps_per_epoch"]):
                self.opt.zero_grad(set_to_none=True)
                total = 0.0
                for src in srcs:
                    out = self.forward(src["name"], next(train[src["name"]]), "train")
                    if out is None:
                        continue
                    loss, parts = self.loss(*out)
                    (loss * src.get("weight", 1.0) / len(srcs)).backward()
                    total += loss.item()
                    for i, v in enumerate(parts):
                        sums[i + 1] += v
                    n += 1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step(); self.sched.step()
                sums[0] += total
                if step % 100 == 0:
                    logger.info(f"epoch {epoch + 1} step {step} loss {total / len(srcs):.4f}")
            row = {"epoch": epoch + 1, "train_loss": sums[0] / max(n, 1),
                   "train_bce": sums[1] / max(n, 1), "train_ce": sums[2] / max(n, 1), "train_fill_l1": sums[3] / max(n, 1),
                   "tau": round(float(self.model.log_tau.exp()), 4), "lr": self.opt.param_groups[1]["lr"]}
            row.update(self.evaluate())
            row["seconds"] = round(time.time() - t0, 1)
            score = sum(v for k, v in row.items() if k.endswith("_l2_acc_0.25")) / len(srcs)
            self._save(epoch, score)
            self._write(row)

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Per sensor and evaluation fill: level-1 accuracy over planted spectra (man-made and natural), level-2
        accuracy over man-made (test-split spectra, own spectrum left out); with an open-set class, how often it is
        called man-made and how often it gets a confident (P > 0.5) wrong class."""
        self.model.eval()
        row, g = {}, torch.Generator(device=self.device)
        for src in self.cfg["sources"]:
            name = src["name"]
            for a in self.cfg["eval_fills"]:
                g.manual_seed(0)
                hit1 = n1 = hit2 = n2 = os_mm = os_wrong = n_os = 0
                mm_logits, nat_logits = [], []
                for i, batch in enumerate(_loader(src, "test", self.cfg)):
                    if i >= self.cfg["eval_batches"]:
                        break
                    out = self.forward(name, batch, "test", fixed_fill=a, kinds=(0.5, 0.5, 0.0), generator=g)
                    if out is None:
                        continue
                    s, logit, _, scores = out
                    bank = self.banks[name]
                    planted = s["entry"] >= 0
                    truth = s["manmade"] > 0.5
                    hit1 += int(((logit > 0) == truth)[planted].sum()); n1 += int(planted.sum())
                    mm_logits.append(logit[truth & (s["cls"] != bank.holdout_class)])
                    nat_logits.append(logit[planted & ~truth])
                    pred = scores.argmax(1)
                    known = truth & (s["cls"] != bank.holdout_class)
                    hit2 += int((pred == s["cls"])[known].sum()); n2 += int(known.sum())
                    if bank.holdout_class >= 0:
                        held = truth & (s["cls"] == bank.holdout_class)
                        conf = scores.softmax(1).max(1).values > 0.5
                        os_mm += int((logit > 0)[held].sum()); os_wrong += int(((logit > 0) & conf)[held].sum())
                        n_os += int(held.sum())
                row[f"{name}_l1_acc_{a}"] = round(hit1 / max(n1, 1), 4)
                row[f"{name}_l2_acc_{a}"] = round(hit2 / max(n2, 1), 4)
                fa = self._ace_false_alarm(name, a)                 # man-made recall at ACE's natural false alarms
                if fa is not None and mm_logits and torch.cat(nat_logits).numel():
                    cut = torch.quantile(torch.cat(nat_logits), 1 - fa)
                    row[f"{name}_l1_recall_at_ace_fa_{a}"] = round(float((torch.cat(mm_logits) > cut).float().mean()), 4)
                if n_os:
                    row[f"{name}_open_mm_{a}"] = round(os_mm / n_os, 4)
                    row[f"{name}_open_wrong_{a}"] = round(os_wrong / n_os, 4)
        return row

    def _ace_false_alarm(self, name: str, fill: float) -> float | None:
        """The natural false-alarm rate of the classical ACE check (separability_<sensor>.json) at that fill."""
        path = Path(self.cfg.get("ace_baseline_dir", "")) / f"separability_{name}.json"
        if not self.cfg.get("ace_baseline_dir") or not path.exists():
            return None
        row = json.loads(path.read_text()).get(str(fill))
        return None if row is None else 1 - row["level1_natural_called_natural"]

    def _save(self, epoch: int, score: float):
        ck = {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
              "epoch": epoch, "best": max(self.best, score), "config": self.cfg, "classes": MANMADE_CLASSES}
        torch.save(ck, self.out / "last.pt")
        torch.save(ck, self.out / f"ep{epoch + 1}.pt")
        if score > self.best:
            self.best = score
            torch.save(ck, self.out / "best.pt")

    def _write(self, row: dict):
        path = self.out / "metrics.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        logger.info("epoch " + ", ".join(f"{k} {v}" for k, v in row.items()))


def main(config_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    NamerRun(json.loads(Path(config_path).read_text())).train()
