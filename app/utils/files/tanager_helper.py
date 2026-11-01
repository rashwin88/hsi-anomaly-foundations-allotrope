"""
FileHelper for Planet Tanager ortho surface reflectance: HDF-EOS5 GRID `.h5`, or ENVI.

An ENVI `.hdr` (+ data file) goes to tanager_envi.py; everything below
describes the HDF5 product, which is what Planet delivers.

The physical-read layer beneath TanagerDatasetBuilder. The georef resolver
(app/georef/from_scene_file.py) and scene onboarding's duplicate pre-check
also use `read_tanager_metadata` so the three agree on scene id and grid.

Why its own helper rather than HE5Helper: HE5Helper is PRISMA's SWATH reader
(two detectors, BIL cubes, root attributes, template indirection). Tanager is
one GRID with a single (C, H, W) cube and its band table in that dataset's
own attributes, so there is nothing for a template to map; `template` is {}.

Layout (verified on 20250205_175845_00_4001, docs/lld/tanager.md):
  HDFEOS/GRIDS/HYP                       attrs epsg_code, strip_id
  HDFEOS/GRIDS/HYP/Data Fields/
    surface_reflectance (426, H, W) f32  0-1, _FillValue -9999; attrs
                                         wavelengths, fwhm, good_wavelengths
    nodata_pixels / beta_cloud_mask / beta_cirrus_mask (H, W) u8, fill 255
  HDFEOS INFORMATION/StructMetadata.0    grid size, corners, UTM zone

Gotchas:
  - Band order out of this helper is ascending wavelength, not file order
    (see TanagerMetadata). Every per-band list and every cube it returns
    uses the sorted order.
  - The cube is chunked 14 bands deep and gzip'd. Reading one band at a
    time decompresses each chunk 14 times, so reads go in chunk-sized blocks.
  - h5py only: the file is untrusted downloaded data.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional

import h5py
import numpy as np

from app.abstract_classes.file_helper import FileHelper
from app.models.file_processing.sources import FileSourceConfig
from app.models.file_processing.tanager_metadata import TanagerGrid, TanagerMetadata
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily


logger = logging.getLogger("TanagerHelper")
logger.setLevel(logging.INFO)


GRID_PATH = "HDFEOS/GRIDS/HYP"
FIELDS_PATH = f"{GRID_PATH}/Data Fields"
STRUCT_METADATA_PATH = "HDFEOS INFORMATION/StructMetadata.0"
CUBE_NAME = "surface_reflectance"
LAYER_NODATA = "nodata_pixels"
LAYER_CLOUD = "beta_cloud_mask"
LAYER_CIRRUS = "beta_cirrus_mask"

# Planet item ids start YYYYMMDD_HHMMSS (UTC); strip ids add a "_strip" suffix.
_ID_TIME_RE = re.compile(r"^(\d{8})_(\d{6})")
_STRIP_SUFFIX = "_strip"


TANAGER_SUFFIXES = (".h5", ".he5", ".hdf5")
# Planet names the asset ortho_sr; an upload may also carry a radiance .h5.
_SR_TOKEN_RE = re.compile(r"(^|[_\-.])sr([_\-.]|$)")


def pick_tanager_file(paths: List[Path]) -> Optional[Path]:
    """The surface-reflectance file among `paths`: an HDF5 if any, else an ENVI `.hdr` whose data file exists.

    Within each format an 'sr'-named file wins, then sorted order. HDF5 wins
    over ENVI because it carries Planet's nodata and cloud masks. Shared by
    backend/allotrope/sensors/source_path.py, scene onboarding and the georef
    resolver so all three read the same file. `paths` must exist on disk for
    the ENVI data-file check.
    """
    from app.utils.files.tanager_envi import envi_data_file   # circular: tanager_envi imports this module

    def best(found: List[Path]) -> Optional[Path]:
        sr = [p for p in found if _SR_TOKEN_RE.search(Path(p).stem.lower())]
        return (sr or found or [None])[0]

    h5 = sorted(p for p in paths if Path(p).suffix.lower() in TANAGER_SUFFIXES)
    if h5:
        return best(h5)
    return best(sorted(p for p in paths if Path(p).suffix.lower() == ".hdr" and envi_data_file(Path(p)) is not None))


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "O") and value.size == 1:
        return _decode(value.reshape(-1)[0])
    return value


def parse_struct_metadata(text: str, epsg: Optional[int] = None) -> TanagerGrid:
    """The HYP grid's geometry out of the ODL text in `StructMetadata.0`.

    Raises ValueError when a field is missing: a grid we cannot place is not
    something to guess at.
    """
    start = text.find('GridName="HYP"')
    body = text[start:] if start >= 0 else text

    def grab(pattern: str) -> re.Match:
        m = re.search(pattern, body)
        if m is None:
            raise ValueError(f"StructMetadata.0 has no {pattern!r}")
        return m

    num = r"(-?[\d.]+(?:[eE][-+]?\d+)?)"
    ul = grab(rf"UpperLeftPointMtrs=\({num},{num}\)")
    lr = grab(rf"LowerRightMtrs=\({num},{num}\)")
    zone = re.search(r"ZoneCode=(-?\d+)", body)
    return TanagerGrid(
        x_dim=int(grab(r"XDim=(\d+)").group(1)),
        y_dim=int(grab(r"YDim=(\d+)").group(1)),
        upper_left_m=(float(ul.group(1)), float(ul.group(2))),
        lower_right_m=(float(lr.group(1)), float(lr.group(2))),
        projection=grab(r"Projection=(\w+)").group(1),
        zone_code=int(zone.group(1)) if zone else None,
        epsg=epsg,
    )


def grid_epsg(grid: TanagerGrid) -> int:
    """EPSG of the grid: the group attribute, else WGS-84 UTM from ZoneCode.

    GCTP marks the southern hemisphere with a negative zone. Northing cannot
    tell the hemispheres apart (southern UTM northings are positive too, via
    the 10,000 km false northing), so it is not consulted.
    """
    if grid.epsg:
        return int(grid.epsg)
    if grid.projection != "HE5_GCTP_UTM" or not grid.zone_code:
        raise ValueError(f"no epsg_code and no UTM zone (projection={grid.projection}, zone={grid.zone_code})")
    return (32700 if grid.zone_code < 0 else 32600) + abs(grid.zone_code)


def _sorted_band_table(wavelengths, fwhm, good) -> tuple[list[int], list[float], list[float], list[int]]:
    """(file index, wavelength, fwhm, good) per band, ascending wavelength; FWHM forced > 0."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    fw = np.asarray(fwhm, dtype=np.float64)
    gd = np.asarray(good, dtype=np.int64).astype(bool)
    if not (wl.shape == fw.shape == gd.shape):
        raise ValueError(f"band table lengths differ: wavelengths {wl.shape}, fwhm {fw.shape}, good {gd.shape}")
    if not np.all(np.isfinite(wl)):
        raise ValueError("non-finite wavelength in the band table")
    order = np.argsort(wl, kind="stable")
    if np.any(order != np.arange(wl.size)):
        moved = np.flatnonzero(order != np.arange(wl.size))
        logger.warning("Tanager wavelengths are not ascending in the file; sorted %d bands (sorted positions %s)",
                       moved.size, moved[:10].tolist())
    wl, fw, gd = wl[order], fw[order], gd[order]
    if np.any(np.diff(wl) <= 0):
        raise ValueError("duplicate wavelengths in the band table")
    # splib07 resampling silently returns nothing for FWHM <= 0, so a band
    # without a usable width is flagged bad and given its band spacing.
    bad_fw = ~np.isfinite(fw) | (fw <= 0)
    if bad_fw.any():
        logger.warning("Tanager: %d bands with FWHM <= 0 flagged bad", int(bad_fw.sum()))
        fw[bad_fw] = np.gradient(wl)[bad_fw]
        gd &= ~bad_fw
    return order.tolist(), wl.tolist(), fw.tolist(), gd.astype(int).tolist()


def tanager_scene_id(strip_id: Optional[str], path: str | Path) -> str:
    """Planet item id: strip_id without '_strip'; the file stem if there is none."""
    if strip_id:
        return strip_id[: -len(_STRIP_SUFFIX)] if strip_id.endswith(_STRIP_SUFFIX) else strip_id
    return Path(path).stem


def _acquisition_time(scene_id: str) -> Optional[datetime]:
    m = _ID_TIME_RE.match(scene_id)
    if m is None:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def read_tanager_metadata(path: str | Path) -> TanagerMetadata:
    """Parse the band table, grid and identity of a Tanager file without reading the cube."""
    if Path(path).suffix.lower() == ".hdr":
        from app.utils.files.tanager_envi import read_tanager_envi_metadata

        return read_tanager_envi_metadata(path)
    with h5py.File(path, "r") as f:
        grid_attrs = {k: _decode(v) for k, v in f[GRID_PATH].attrs.items()}
        cube = f[f"{FIELDS_PATH}/{CUBE_NAME}"]
        attrs = cube.attrs
        bands, height, width = cube.shape
        fill = float(np.asarray(attrs.get("_FillValue", -9999.0)).reshape(-1)[0])
        order, wl, fw, good = _sorted_band_table(
            attrs["wavelengths"], attrs["fwhm"], attrs.get("good_wavelengths", np.ones(bands)),
        )
        struct = _decode(f[STRUCT_METADATA_PATH][()])
    if len(wl) != bands:
        raise ValueError(f"band table has {len(wl)} entries for a {bands}-band cube")
    epsg_attr = grid_attrs.get("epsg_code")
    grid = parse_struct_metadata(struct, epsg=int(epsg_attr) if epsg_attr not in (None, "") else None)
    if (grid.y_dim, grid.x_dim) != (height, width):
        raise ValueError(f"StructMetadata grid {grid.y_dim}x{grid.x_dim} != cube {height}x{width}")
    strip_id = grid_attrs.get("strip_id") or None
    scene_id = tanager_scene_id(strip_id, path)
    return TanagerMetadata(
        source_path=str(path), scene_id=scene_id, strip_id=strip_id, acquisition_at=_acquisition_time(scene_id),
        bands=bands, height=height, width=width, fill_value=fill,
        wavelengths_nm=wl, fwhm_nm=fw, good_band=good, file_band_index=order, grid=grid,
    )


class TanagerHelper(FileHelper[TanagerMetadata]):
    """Reads a Tanager ortho_sr `.h5`, or an ENVI `.hdr`. `FileSourceConfig.source_path` is the file."""

    def __init__(self, file_source_config: FileSourceConfig):
        super().__init__(file_source_config=file_source_config, template={})
        self.path = file_source_config.source_path
        self._file_metadata = self._construct_metadata_structure()

    def _construct_metadata_structure(self) -> TanagerMetadata:
        return read_tanager_metadata(self.path)

    @property
    def file_metadata(self) -> TanagerMetadata:
        return self._file_metadata

    @property
    def template(self) -> dict:
        return self._template

    @property
    def has_planet_layers(self) -> bool:
        """True for HDF5: nodata, cloud and cirrus layers exist. ENVI carries none."""
        return self.file_metadata.file_format == "hdf5"

    def extract_specific_bands(
        self,
        bands: List[int] | int,
        masking_needed: Optional[bool] = False,
        spectral_family: Optional[SpectralFamily] = None,
        mode: Literal["all", "specific"] = "specific",
    ) -> np.ndarray | np.ma.MaskedArray:
        """(C, H, W) float32 reflectance in ascending-wavelength order.

        `bands` are positions in the SORTED band table. `spectral_family` is
        ignored: Tanager has one cube, and family is a wavelength split.
        Masking hides fill and non-finite values; validity is the builder's job.
        """
        md = self.file_metadata
        positions = list(range(md.bands)) if mode == "all" else ([bands] if isinstance(bands, int) else list(bands))
        file_idx = np.asarray([md.file_band_index[p] for p in positions], dtype=np.int64)
        if file_idx.size == 0:
            return np.empty((0, md.height, md.width), dtype=np.float32)
        if md.file_format == "envi":
            from app.utils.files.tanager_envi import read_envi_bands

            out = read_envi_bands(md, file_idx)
        else:
            out = np.empty((len(positions), md.height, md.width), dtype=np.float32)
            self._read_h5_bands(file_idx, out)
        if masking_needed:
            return np.ma.masked_where((out == md.fill_value) | ~np.isfinite(out), out)
        return out

    def _read_h5_bands(self, file_idx: np.ndarray, out: np.ndarray) -> None:
        with h5py.File(self.path, "r") as f:
            ds = f[f"{FIELDS_PATH}/{CUBE_NAME}"]
            step = (ds.chunks or (16,))[0]
            for b0 in range(int(file_idx.min()), int(file_idx.max()) + 1, step):
                b1 = min(b0 + step, int(file_idx.max()) + 1)
                wanted = np.flatnonzero((file_idx >= b0) & (file_idx < b1))
                if wanted.size == 0:
                    continue
                block = ds[b0:b1]
                out[wanted] = block[file_idx[wanted] - b0]

    def extract_layer(self, name: str) -> np.ndarray:
        """One (H, W) layer from `Data Fields`, e.g. LAYER_NODATA. KeyError for an ENVI scene."""
        if not self.has_planet_layers:
            raise KeyError(f"ENVI Tanager scene has no {name!r} layer")
        with h5py.File(self.path, "r") as f:
            return f[f"{FIELDS_PATH}/{name}"][()]
