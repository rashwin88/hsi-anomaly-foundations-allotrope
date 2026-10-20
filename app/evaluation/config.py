"""
EvaluationConfig: everything one evaluation run needs, as one JSON file (docs/lld/evaluation.md).

    {"name": ..., "models": [...], "datasets": [...], "scorers": [...], "metrics": [...], "group_by": [...],
     "output": "<dir>"}

Scorers are named by `name` or, if absent, by their settings, e.g. "residual_mahalanobis(d=30,w=21x61)" - the
name is the column the report uses and part of the score-map cache key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CLASS_IDS = {"land": 0, "water": 1, "cloud": 2, "shadow": 3, "snow": 4}   # segmentation classes (FT03 head)


class ModelSpec(BaseModel):
    name: str
    checkpoint: str = Field(description="Local path or s3:// URI of a training checkpoint.")
    reading: Literal["auto", "native", "product165", "se202"] = Field(
        default="auto", description="auto: native for wavelength stems, se202 for fixed stems, product165 for "
        "hyperspectral_segformer_mae (v0.2.0).")
    adapt_steps: int = Field(default=0, ge=0, description="Gated per-scene adaptation steps before reconstructing "
                             "(app/foundation_models/scene_adaptation.py); 0 = off. Wavelength stems only.")
    adapt_lr: float = Field(default=1e-5, gt=0)
    adapt_gate_pct: float = Field(default=1.0, ge=0, lt=100, description="Top % of the unadapted model's residual "
                                  "Mahalanobis score left out of the adaptation loss; 0 = no gating.")
    adapt_halo: int = Field(default=2, ge=0, description="Gate dilation, px.")


class DatasetSpec(BaseModel):
    kind: Literal["prisma_bench", "implant"]
    path: str = Field(description="prisma_bench: the stage1_datasets root; implant: the bench directory.")
    scenes: list[str] | None = Field(default=None, description="Subset of scene / host names; default all.")


class ScorerSpec(BaseModel):
    kind: Literal["residual_mahalanobis", "combined", "L1", "SAM", "class_aware", "exclude", "pca_rx", "fuse",
                  "mixture", "scaled_rm"]
    name: str | None = None
    d: int = 30
    local_window: tuple[int, int] | None = (21, 61)
    base: "ScorerSpec | None" = Field(default=None, description="exclude: the scorer whose output is masked.")
    classes: list[Literal["water", "cloud", "shadow", "snow"]] = Field(default_factory=list,
                                                                      description="exclude: classes scored 0.")
    segmenter: str | None = Field(default=None, description="class_aware / exclude: a model name with a seg head.")
    fit_on_kept: bool = Field(default=True, description="exclude: fit the base scorer on the kept pixels only.")
    spatial_sigma: float = Field(default=0.0, ge=0, description="residual_mahalanobis / pca_rx / mixture: Gaussian blur (px) of the residual before whitening - a spatial matched filter for PSF-blurred targets.")
    parts: list["ScorerSpec"] = Field(default_factory=list, description="fuse: the scorers combined.")
    rule: Literal["rank_mean", "rank_max"] = Field(default="rank_mean", description="fuse: how ranks combine.")
    k: int = Field(default=6, ge=2, description="mixture: number of background types (k-means on the cube).")
    scale_power: float = Field(default=1.0, gt=0, description="scaled_rm: divide the residual by b ** scale_power.")
    scale_floor_pct: float = Field(default=5.0, ge=0, lt=100, description="scaled_rm: floor b at this percentile.")

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "exclude" and (self.base is None or not self.classes or not self.segmenter):
            raise ValueError("exclude needs base, classes and segmenter")
        if self.kind == "fuse" and len(self.parts) < 2:
            raise ValueError("fuse needs at least two parts")
        if self.kind == "class_aware" and not self.segmenter:
            raise ValueError("class_aware needs segmenter")
        return self

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        w = "global" if self.local_window is None else f"{self.local_window[0]}x{self.local_window[1]}"
        blur = f",s={self.spatial_sigma:g}" if self.spatial_sigma else ""
        if self.kind == "scaled_rm":
            return f"scaled_rm(p={self.scale_power:g},floor={self.scale_floor_pct:g}%,d={self.d},w={w}{blur})"
        if self.kind in ("residual_mahalanobis", "class_aware", "pca_rx"):
            return f"{self.kind}(d={self.d},w={w}{blur})"
        if self.kind == "mixture":
            return f"mixture(k={self.k},d={self.d},w={w}{blur})"
        if self.kind == "fuse":
            return f"fuse[{self.rule}](" + "; ".join(p.label for p in self.parts) + ")"
        if self.kind == "exclude":
            return f"exclude({'+'.join(self.classes)};{'fit' if self.fit_on_kept else 'mask'})[{self.base.label}]"
        return self.kind

    @property
    def needs_scale(self) -> bool:
        """True if the scorer reads the model's uncertainty head; the runner skips it for models without one."""
        return (self.kind == "scaled_rm" or (self.kind == "exclude" and self.base.needs_scale)
                or (self.kind == "fuse" and any(p.needs_scale for p in self.parts)))

    @property
    def needs_model(self) -> bool:
        if self.kind == "fuse":
            return any(p.needs_model for p in self.parts)
        return self.kind != "pca_rx" and (self.kind != "exclude" or self.base.needs_model)


class MetricSpec(BaseModel):
    kind: Literal["roc_auc", "pauc", "tpr_at", "objects_caught", "three_d_roc"]
    max_fpr: float = 0.01
    fpr: float = 0.001
    top_pct: list[float] = Field(default_factory=lambda: [1.0, 0.1])


class EvaluationConfig(BaseModel):
    name: str
    models: list[ModelSpec] = Field(default_factory=list)
    datasets: list[DatasetSpec] = Field(min_length=1)
    scorers: list[ScorerSpec] = Field(min_length=1)
    metrics: list[MetricSpec] = Field(default_factory=lambda: [
        MetricSpec(kind="roc_auc"), MetricSpec(kind="pauc"), MetricSpec(kind="tpr_at"),
        MetricSpec(kind="objects_caught")])
    group_by: list[str] = Field(default_factory=list, description="Object attributes to break results down by, "
                                "e.g. tier, source, sensor.")
    output: str
    cache: bool = Field(default=True, description="Cache score maps (float16) under <output>/cache.")

    @model_validator(mode="after")
    def _check(self):
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        labels = [s.label for s in self.scorers]
        if len(set(labels)) != len(labels):
            raise ValueError(f"scorer labels must be unique: {labels}")
        for s in self.scorers:
            if s.segmenter and s.segmenter not in names:
                raise ValueError(f"scorer {s.label}: segmenter {s.segmenter!r} is not a model in this config")
        if any(s.needs_model for s in self.scorers) and not self.models:
            raise ValueError("model-based scorers need at least one model")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationConfig":
        return cls(**json.loads(Path(path).read_text()))
