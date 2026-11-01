"""
What a story is told from: a spectral_library_match action's chain as directories, configs and summaries,
and the readers that pull the scene picture, its grid and its masks out of them.

Filled by the api (backend/allotrope/api/action_story.py, which walks the configs' input ids in the
database) or by a test from a directory tree; read by app/story/frames.py. No DB here.

Gotcha: rgb.png is not always on the scene grid. The upstream actions cap it at 4,096 px on its longer
side, so anything placed by scene (row, col) - masks, namer centroids - is scaled by ``scene_grid``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from app.story.render import resize

# Outside spatial validity in class_map.tif / class_codes.bin (docs/lld/segmentation-v2.md).
INVALID = 255


class StoryError(ValueError):
    """The story cannot be told (no scene picture to draw on). The api answers 422 with it."""


@dataclass
class Step:
    """One action of the chain: its output directory, configuration and output summary."""

    dir: Path
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


@dataclass
class StoryChain:
    """A spectral_library_match action and what it was built from; any upstream step may be None."""

    sensor: str
    scene_name: str
    match: Step
    acquired: date | datetime | None = None
    scene_bands: int | None = None
    prep: Step | None = None
    scoring: Step | None = None
    segmentation: Step | None = None
    band_filter: Step | None = None


@dataclass
class SelectionParams:
    """The class-driven selection, same meaning as the class-driven export's query parameters."""

    tokens: tuple[str, ...] = ("manmade",)
    confident_only: bool = False
    manmade_p: float | None = None          # None = the action's own cut (0.5)


def has_namer(chain: StoryChain) -> bool:
    return (chain.match.dir / "namer_objects.parquet").is_file()


def scene_grid(chain: StoryChain, rgb: np.ndarray) -> tuple[int, int]:
    """(rows, cols) of the scene grid the masks and namer centroids are on. rgb.png can be smaller: the
    upstream actions cap it at 4,096 px on its longer side (an AVIRIS-NG flight line is 4,118 rows)."""
    bf = chain.band_filter.summary if chain.band_filter else {}
    if bf.get("height") and bf.get("width"):
        return int(bf["height"]), int(bf["width"])
    shp = chain.prep.summary.get("composite_shape") if chain.prep else None
    return (int(shp[0]), int(shp[1])) if shp else rgb.shape[:2]


def scene_rgb(chain: StoryChain) -> np.ndarray:
    for step in (chain.match, chain.prep, chain.scoring, chain.segmentation):
        if step is not None and (step.dir / "rgb.png").is_file():
            return np.asarray(Image.open(step.dir / "rgb.png").convert("RGB"))
    raise StoryError("no rgb.png in this action or the actions upstream of it, so there is no scene to draw on")


def valid_mask(chain: StoryChain, rgb: np.ndarray, shape) -> np.ndarray:
    """Spatially valid pixels on the display grid: the segmentation's class map where there is one (255 =
    outside valid), else the RGB's non-black pixels (upstream actions write no-data as black)."""
    if chain.segmentation is not None:
        cm = read_tif(chain.segmentation.dir / "class_map.tif")
        if cm is not None:
            return resize((cm != INVALID).astype(np.uint8) * 255, shape) >= 128
    return resize((rgb.max(axis=2) > 0).astype(np.uint8) * 255, shape) >= 128


def read_tif(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    import warnings

    import rasterio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)  # identity, by design
        with rasterio.open(path) as src:
            return src.read(1)


def class_codes(seg: Step) -> np.ndarray | None:
    path, shp = seg.dir / "class_codes.bin", seg.summary.get("scene_shape")
    if path.is_file() and shp and path.stat().st_size == shp[0] * shp[1]:
        return np.fromfile(path, np.uint8).reshape(shp[0], shp[1])
    return read_tif(seg.dir / "class_map.tif")


def namer_labels(match_dir: Path) -> dict[str, str]:
    try:
        legend = json.loads((match_dir / "namer_legend.json").read_text())
        return {v["cls"]: v["label"] for v in legend.get("classes", {}).values() if "cls" in v}
    except (OSError, ValueError, KeyError, AttributeError):
        return {}
