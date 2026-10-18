"""
Segmentation targets for fine-tuning: EnMAP's provider quality layers turned into per-class targets
with an explicit "no label" mask, plus the masked losses and the IoU tally used to train and judge
the segmentation head (docs/proposals/fine-tuning-and-segmentation.md).

Used by the fine-tuning trainer, per sample in the loader workers (`enmap_targets`) and per batch on
the GPU (`masked_bce_dice`, `IoUTally`).

Classes, one independent logit each (they overlap - thin cloud over water):

    0 water        label_classes == 2 (land = 1)
    1 cloud        label_cloud == 1, which includes every label_classes == 0 pixel (EnMAP's cloud class)
    2 thin_cloud   label_cirrus >= cirrus_min_level (2), or label_haze == 1
    3 shadow       label_cloud_shadow == 1
    4 snow         label_snow == 1

Each class also has a "known" mask; unknown pixels are left out of the loss. The rules come from a label
QA over all 15,492 EnMAP tiles (2026-10-16; docs/proposals/fine-tuning-and-segmentation.md):
  - label_classes == 3 is provider no-data: unknown for every class. label_classes == 0 is NOT no-data -
    it is EnMAP's cloud class (79% of all cloud-labelled pixels, mean 450-650 nm reflectance 0.30).
    There cloud is known (= 1) and the other four classes are unknown.
  - Dry scenes (scene_wv below 750, i.e. 0.75 g/cm2 - meta stores g/cm2 x 1000): cloud and thin_cloud
    unknown for the whole scene. Below 0.75 g/cm2 the cirrus labels are 90-100% surface-driven (dry ground
    seen through thin air, not cloud); the rule removes 63% of suspect labels for 14% of plausible ones.
    Elevation >= 3500 m is kept as a guard; no scene in native-v1 reaches it (max 2776 m).
  - cirrus level 1 is unknown for thin_cloud, not negative (over water it is no brighter than clear water).
  - Snow wins over cloud: where both are set, cloud is unknown (122,815 such pixels).
  - Water is unknown where cloud or cirrus >= 2 is set (those "water" pixels are land-bright in the NIR).

Gotchas:
  - Policy values are run configuration (LabelPolicy), fixed from the QA before training, never tuned.
  - Targets are float32 0/1 and known bool, both (5, H, W); the loss averages over known pixels per class.
  - Snow is too rare in EnMAP's test split to measure (1 tile); shadow is rare (0.09% of pixels).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

CLASSES = ("water", "cloud", "thin_cloud", "shadow", "snow")


@dataclass(frozen=True)
class LabelPolicy:
    plateau_elevation_m: float | None = 3500.0   # mean ground elevation at or above: cloud, thin_cloud unknown
    low_water_vapour: float | None = 750.0        # scene_wv (g/cm2 x 1000) below: cloud, thin_cloud unknown
    cirrus_min_level: int = 2                      # cirrus levels 0-3; at or above = thin cloud; 1 = unknown


def enmap_targets(labels: dict[str, np.ndarray], meta: dict, policy: LabelPolicy = LabelPolicy()
                  ) -> tuple[np.ndarray, np.ndarray]:
    """labels: label_* arrays as stored ((1, H, W) uint8). meta: the tile's meta.json.
    Returns (targets (5, H, W) float32, known (5, H, W) bool)."""
    cls = labels["label_classes"][0]
    cloud_class = cls == 0                               # EnMAP's own cloud class
    cloud = (labels["label_cloud"][0] == 1) | cloud_class
    cirrus, haze = labels["label_cirrus"][0], labels["label_haze"][0] == 1
    snow = labels["label_snow"][0] == 1
    t = np.stack([cls == 2, cloud, (cirrus >= policy.cirrus_min_level) | haze,
                  labels["label_cloud_shadow"][0] == 1, snow]).astype(np.float32)
    known = np.repeat((cls != 3)[None], len(CLASSES), axis=0)
    known[[0, 2, 3, 4]] &= ~cloud_class                  # at cloud-class pixels only "cloud" is known
    known[2] &= ~((cirrus > 0) & (cirrus < policy.cirrus_min_level) & ~haze)   # cirrus 1: unsure, not clear
    known[1] &= ~snow                                    # snow wins over cloud
    known[0] &= ~(cloud | (cirrus >= policy.cirrus_min_level))                # no water truth under cloud
    elev, wv = meta.get("mean_ground_elevation_m"), meta.get("scene_wv")
    dry = ((policy.plateau_elevation_m is not None and elev is not None and elev >= policy.plateau_elevation_m)
           or (policy.low_water_vapour is not None and wv is not None and wv < policy.low_water_vapour))
    if dry:
        known[1] = known[2] = False
    return t, known


def masked_bce_dice(logits: torch.Tensor, targets: torch.Tensor, known: torch.Tensor,
                    dice_weight: float = 1.0, class_weights=None) -> torch.Tensor:
    """Per-class binary cross-entropy + soft Dice over known pixels, averaged over classes that have any
    known pixel in the batch - weighted by class_weights (K,) if given (rare classes, FT04).
    logits, targets: (B, K, H, W); known: (B, K, H, W) bool."""
    k = known.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    n = k.sum(dim=(0, 2, 3))                                        # (K,) known pixels per class
    has = n > 0
    if not has.any():
        return logits.sum() * 0.0
    bce_c = (bce * k).sum(dim=(0, 2, 3)) / n.clamp(min=1)
    p = torch.sigmoid(logits) * k
    inter, denom = (p * targets).sum(dim=(0, 2, 3)), (p + targets * k).sum(dim=(0, 2, 3))
    dice_c = 1 - (2 * inter + 1) / (denom + 1)
    per = bce_c + dice_weight * dice_c
    if class_weights is None:
        return per[has].mean()
    w = torch.as_tensor(class_weights, dtype=per.dtype, device=per.device)
    return (w * per)[has].sum() / w[has].sum()


PSEUDO_CONF = 0.9          # a pseudo-label counts only where the teacher's probability is > 0.9 or < 0.1


def band_mean(refl: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (wl >= lo) & (wl <= hi)
    return np.nanmean(refl[sel], axis=0) if sel.any() else np.full(refl.shape[1:], np.nan)


def pseudo_targets(probs: np.ndarray, refl: np.ndarray, wl: np.ndarray, valid: np.ndarray,
                   conf: float = PSEUDO_CONF) -> tuple[np.ndarray, np.ndarray]:
    """Teacher probabilities (K, H, W) uint8 (0-255) or float -> (targets float32, known bool) for a sensor with no
    labels (FT04, PRISMA / AVIRIS-NG). Only water and cloud are taught, where the teacher is confident (> conf or
    < 1 - conf); a confident water with NIR (800-900 nm) above 0.15 becomes a known NOT-water. Thin cloud, shadow and
    snow stay unknown (EnMAP's labels teach them).

    No cloud guard: cloud_plausible (bright and flat) rejected 69% of EnMAP's labelled clouds on a test shard, while the
    teacher's confident clouds matched them at IoU 0.82; and PRISMA set4's "11% cloud", first read as bright desert, is
    haze and cloud on inspection (2026-10-18)."""
    p = probs.astype(np.float32) / (255.0 if probs.dtype == np.uint8 else 1.0)
    targets = (p > conf).astype(np.float32)
    known = ((p > conf) | (p < 1 - conf)) & valid[None]
    known[2:] = False
    not_water = (targets[0] > 0) & (band_mean(refl, wl, 800, 900) > 0.15)
    targets[0][not_water] = 0.0
    return targets, known


class IoUTally:
    """Accumulates per-class intersection and union of (sigmoid > 0.5) against targets, known pixels only."""

    def __init__(self, n_classes: int = len(CLASSES)):
        self.inter = torch.zeros(n_classes, dtype=torch.float64)
        self.union = torch.zeros(n_classes, dtype=torch.float64)

    def add(self, logits: torch.Tensor, targets: torch.Tensor, known: torch.Tensor) -> None:
        pred, tgt, k = (logits > 0), targets > 0.5, known
        self.inter += ((pred & tgt & k).sum(dim=(0, 2, 3))).double().cpu()
        self.union += (((pred | tgt) & k).sum(dim=(0, 2, 3))).double().cpu()

    def iou(self) -> dict[str, float | None]:
        return {c: (float(i / u) if u > 0 else None) for c, i, u in zip(CLASSES, self.inter, self.union)}
