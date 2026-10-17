"""
Evaluation datasets: each yields EvalScene objects one at a time, so a run holds one scene in memory
(docs/lld/evaluation.md).

    PrismaBench(root)    the 4 labelled PRISMA L2D scenes (909 objects), read the product's way per mode
    ImplantBench(path)   the implanted-target benchmark (scripts/build_implant_bench.py), 18 held-out hosts

An EvalScene carries the cube a wavelength-stem model reads (native bands after the product's exclusions),
its objects as an int id map, and per-object attributes for group_by. `read(mode)` gives the cube a fixed-grid
model needs: PRISMA scenes reload through the product's band_filter_apply path (exact); other scenes are linearly
resampled (close to, not identical with, the product).

Real bench objects get a contrast tier from the same function as the implants (app/evaluation/implant.py), so
tiers group both benchmarks alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from scipy import ndimage

# Relative to the stage1_datasets root (C:/Users/FITT/Documents/stage1_datasets on the dev laptop).
PRISMA_SCENES = {
    "set1": ("Set-1/Hypersepctral Datasets/PRS_L2D_STD_20201214060713_20201214060717_0001.he5",
             "Set-1/Hypersepctral Datasets/Ground truth/groundtruth_hyper.tif"),
    "set2": ("Set-2/Hypersepctral Dataset/PRS_L2D_STD_20241205050514_20241205050518_0001.he5",
             "Set-2/Hypersepctral Dataset/Ground truth/Hyp_Ground_Truth.tif"),
    "set3": ("Set-3/Hyper/PRS_L2D_STD_20210516050459_20210516050503_0001.he5",
             "Set-3/Hyper/Ground truth/Groundtruth hyper anomalies.tif"),
    "set4": ("Set-4/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5",
             "Set-4/Hyper/Ground truth/ground_truth_hyper.tif"),
}
MODES = ("native", "product165", "se202")
# SpectralEarth dropped EnMAP's water bands at 1295-1507 and 1728-1968 nm; only those gaps are excluded in se202
# so every SpectralEarth wavelength has PRISMA bands on both sides of it.
SE202_EXCLUSIONS = [(1300.0, 1512.0), (1722.0, 1972.0)]
PRODUCT_EXCLUSIONS = [(0, 450), (912, 978), (1131, 1152), (1350, 1450), (1800, 1950)]
NODATA, SCALE = -32768, 1e-4


@dataclass
class EvalScene:
    dataset: str
    name: str
    sensor: str
    cube: np.ndarray                 # (C, H, W) float32 reflectance, gaps filled
    wavelengths_nm: np.ndarray       # (C,)
    valid: np.ndarray                # (H, W) bool
    objects: np.ndarray              # (H, W) int32, 0 = background, k = object k
    attrs: dict = field(default_factory=dict)       # object id -> {attribute: value}
    units: str = "reflectance"
    reader: Callable | None = None   # mode -> (cube, wavelengths, valid), for an exact non-native reading

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.name}"

    def read(self, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cube, wavelengths, valid) in a loading mode: native, product165 or se202."""
        if mode == "native":
            return self.cube, self.wavelengths_nm, self.valid
        if self.reader is not None:
            return self.reader(mode)
        return resample(self.cube, self.wavelengths_nm, target_grid(mode)), target_grid(mode), self.valid


def target_grid(mode: str) -> np.ndarray:
    if mode == "product165":
        from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
        return np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, float)
    if mode == "se202":
        from app.models.dataset.spectralearth import load_band_table
        return np.sort(load_band_table().wavelengths_nm)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def resample(cube: np.ndarray, wl: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation along wavelength, (C, H, W) -> (G, H, W) float32."""
    M = np.stack([np.interp(grid, wl, np.eye(len(wl))[i]) for i in range(len(wl))], 1)
    return np.einsum("gb,bhw->ghw", M, cube, optimize=True).astype(np.float32)


def object_attributes(cube: np.ndarray, valid: np.ndarray, objects: np.ndarray) -> dict:
    """id -> {"size", "z", "tier"} for real labelled objects: contrast z against the diamond ring, windowed."""
    from app.evaluation.implant import contrast_z, ring_mask, tier_of
    out = {}
    for i, sl in enumerate(ndimage.find_objects(objects), 1):
        if sl is None:
            continue
        win = tuple(slice(max(0, s.start - 4), s.stop + 4) for s in sl)
        obj = objects[win] == i
        ring = ring_mask(obj, valid[win], exclude=objects[win] > 0)
        sub = cube[(slice(None),) + win]
        if ring.sum() < 8:
            out[i] = {"size": int(obj.sum())}
            continue
        z = contrast_z(sub, np.ones(sub.shape, bool), obj, ring)
        out[i] = {"size": int(obj.sum()), "z": round(z, 4), "tier": tier_of(z)}
    return out


# --- PRISMA bench ---------------------------------------------------------------------------------------------

def load_prisma_scene(root: str | Path, name: str, mode: str):
    """(cube (C, H, W) float32 nearest-valid filled, validity (C, H, W) int8, wavelengths, spatial_valid (H, W) uint8,
    gt (H, W) uint8) for one bench scene, read as the product's band_filter_apply would in `mode`."""
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID, BandFilterConfig
    from app.models.file_processing.sources import FileSourceConfig
    from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
    from app.utils.pixel_fill.nearest_valid_fill import nearest_valid_fill
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    he5, gt_rel = (Path(root) / p for p in PRISMA_SCENES[name])
    product = dict(exclusion_ranges=PRODUCT_EXCLUSIONS, edge_bands_to_trim=3, min_valid_pixel_pct=20.0,
                   max_invalid_voxel_fraction=0.4, quality_masks_to_apply=[])
    if mode == "product165":
        cfg = BandFilterConfig(**product, common_wavelength_grid=DEFAULT_COMMON_WAVELENGTH_GRID)
    elif mode == "se202":
        cfg = BandFilterConfig(**{**product, "exclusion_ranges": SE202_EXCLUSIONS, "edge_bands_to_trim": 0},
                               common_wavelength_grid=target_grid("se202"))
    else:
        cfg = BandFilterConfig(**product, common_wavelength_grid=None)
    v = PrismaDatasetBuilder(FileSourceConfig(source_path=str(he5))).vend_dataset(band_filter_config=cfg)
    validity = np.asarray(v.validity_cube, dtype=np.int8)
    spatial = (validity.sum(axis=0) > 0).astype(np.uint8)
    cube = nearest_valid_fill(v.normalized_hyperspectral_cube, spatial[None]).astype(np.float32)
    return cube, validity, np.asarray(v.band_cw_order, dtype=np.float64), spatial, load_gt(gt_rel, spatial.shape)


def load_gt(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) uint8 anomaly mask (> 0). Raises ValueError if it does not match the scene grid."""
    import rasterio
    with rasterio.open(path) as src:
        gt = (src.read(1) > 0).astype(np.uint8)
    if gt.shape != shape:
        raise ValueError(f"{path.name}: ground truth {gt.shape} vs scene {shape}")
    return gt


class PrismaBench:
    """The 4 labelled PRISMA scenes. Objects are the ground truth's connected components (4-neighbour, as every
    earlier count); each gets size, contrast z and tier. GT may sit ~1 px off the grid - objects_caught allows 1 px."""

    def __init__(self, root: str, scenes: list[str] | None = None):
        self.root, self.scenes = root, scenes or list(PRISMA_SCENES)

    def __iter__(self) -> Iterator[EvalScene]:
        for name in self.scenes:
            cube, _, wl, spatial, gt = load_prisma_scene(self.root, name, "native")
            objects, _ = ndimage.label(gt)
            valid = spatial.astype(bool)
            reader = lambda mode, n=name: (lambda c, _, w, sp, __: (c, w, sp.astype(bool)))(*load_prisma_scene(self.root, n, mode))
            yield EvalScene("prisma_bench", name, "prisma", cube, wl, valid, objects.astype(np.int32),
                            object_attributes(cube, valid, objects), reader=reader)


# --- implant bench ----------------------------------------------------------------------------------------------

class ImplantBench:
    """The implanted-target benchmark: per host the product's exclusions, bands mostly empty over valid pixels
    dropped, missing voxels filled per band from the nearest valid pixel. Attributes from objects.json."""

    def __init__(self, path: str, hosts: list[str] | None = None):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.hosts = [h for h in self.manifest["hosts"] if hosts is None or h["scene"] in hosts]

    def __iter__(self) -> Iterator[EvalScene]:
        for h in self.hosts:
            d = self.path / h["sensor"] / h["scene"]
            cube, wl, valid = load_implant_host(d)
            objs = json.loads((d / "objects.json").read_text())
            attrs = {o["id"]: {"tier": o["tier"], "source": o["source"], "size": o["px"], "sensor": h["sensor"]}
                     for o in objs}
            yield EvalScene("implant", h["scene"], h["sensor"], cube, wl, valid,
                            np.load(d / "gt.npy").astype(np.int32), attrs)


def load_implant_host(host: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cube = np.load(host / "cube.npy")
    wl = np.load(host / "wavelengths.npy")
    keep = np.array([not any(lo <= w <= hi for lo, hi in PRODUCT_EXCLUSIONS) for w in wl])
    cube, wl = cube[keep], wl[keep]
    finite = cube != NODATA
    valid = finite.mean(0) >= 0.9
    good = finite[:, valid].mean(1) >= 0.6                     # drop bands mostly empty over valid pixels
    cube, wl, finite = cube[good], wl[good], finite[good]
    refl = cube.astype(np.float32) * SCALE
    for b in range(len(wl)):                                     # per-band nearest-valid fill of missing voxels
        hole = ~finite[b] & valid
        if hole.any():
            idx = ndimage.distance_transform_edt(~finite[b], return_distances=False, return_indices=True)
            refl[b][hole] = refl[b][idx[0][hole], idx[1][hole]]
    refl[:, ~valid] = 0.0
    return refl, wl, valid


def make_dataset(spec):
    if spec.kind == "prisma_bench":
        return PrismaBench(spec.path, spec.scenes)
    if spec.kind == "implant":
        return ImplantBench(spec.path, spec.scenes)
    raise ValueError(f"unknown dataset kind {spec.kind!r}")
