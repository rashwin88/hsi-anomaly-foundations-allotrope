"""
SpectralEarth patches as a map-style torch Dataset, native or on the common grid.

The entry point training code uses: give it a catalogue (usually one side of
SpectralEarthCatalog.split), a SpectralEarthSource and a ScreenChain, wrap it in a
DataLoader. It sits on top of everything else in app/utils/spectralearth/ and decides
nothing about the model.

Each patch is already its own 6.6 MB S3 object, so this reads patches directly by index —
random access, no shards. Whether training also needs shards for throughput is a question
for the GPU setup; this keeps it open.

Two modes:
  native       202 bands, SpectralEarth's own EnMAP wavelengths (no resampling, ~free).
  common_grid  165 bands on the 460-2450 nm grid, through the EnMAP pipeline
               (~60 ms/patch on one core — likely too slow to run live in a loader).

Each sample is a dict:
  pixels        float32 (C, 128, 128) reflectance; rejected pixels are 0.0
  validity      uint8 (1, 128, 128), 1 where the pixel survived screening and has a
                complete spectrum
  keep_fraction float, share of pixels kept — the trainer's low-coverage filter reads this
  key           str, `se_<site>_<patch>`

Gotcha: wavelengths are the same for every sample, so they live on the dataset
(`.wavelengths_nm`), not in each sample, to keep them out of batch collation.
"""

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from app.models.dataset.spectralearth import load_band_table
from app.models.dataset.vendables import BandFilterConfig, DEFAULT_COMMON_WAVELENGTH_GRID
from app.utils.spectralearth.catalog import SpectralEarthCatalog
from app.utils.spectralearth.screens import NodataScreen, ScreenChain
from app.utils.spectralearth.source import SpectralEarthSource

Mode = Literal["native", "common_grid"]


class SpectralEarthTorchDataset(Dataset):
    def __init__(
        self,
        catalog: SpectralEarthCatalog,
        source: SpectralEarthSource,
        mode: Mode = "native",
        screens: ScreenChain | None = None,
        band_filter_config: BandFilterConfig | None = None,
    ):
        if mode not in ("native", "common_grid"):
            raise ValueError(f"mode must be 'native' or 'common_grid', got {mode!r}")
        self.catalog, self.source, self.mode = catalog, source, mode
        self.screens = screens if screens is not None else ScreenChain([NodataScreen()])
        self.band_filter_config = band_filter_config or BandFilterConfig(
            common_wavelength_grid=DEFAULT_COMMON_WAVELENGTH_GRID
        )

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Band centres of every sample's `pixels`, in channel order."""
        if self.mode == "native":
            return load_band_table().wavelengths_nm
        return np.asarray(self.band_filter_config.common_wavelength_grid)

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, i: int) -> dict:
        patch = self.source.load(self.catalog[i])
        report = self.screens.apply(patch)
        if self.mode == "native":
            pixels = patch.reflectance()
            valid = report.keep & patch.validity().all(axis=0)
            pixels[:, ~valid] = 0.0
        else:
            v = patch.to_vendable(self.band_filter_config, keep=report.keep)
            pixels = v.normalized_hyperspectral_cube
            valid = v.validity_cube.all(axis=0).astype(bool)
        return {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32)),
            "validity": torch.from_numpy(valid[None].astype(np.uint8)),
            "keep_fraction": float(valid.mean()),
            "key": patch.key,
        }
