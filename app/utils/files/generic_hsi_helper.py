"""
FileHelper for "any hyperspectral file" (`sensor_type="generic_hsi"`): GeoTIFF / COG, ENVI, HDF5 / NetCDF.

The physical-read layer beneath GenericHsiDatasetBuilder. Scene onboarding,
backend/allotrope/sensors/source_path.py and the georef resolver use
`pick_generic_file` / `locate_data` so all of them read the same file.

Why it exists: once evaluation starts the code is frozen, and the data may be
GHOSt GeoTIFFs, OSK's native ENVI (`.hsi` + `.hdr` + `.igm`) or something
nobody has shown us. Native readers stay for PRISMA / EnMAP / AVIRIS-NG /
Tanager (they use provider quality masks); this is the safety net for
everything else (docs/proposals/robust-hsi-ingestion.md, docs/lld/generic-hsi.md).
Every decision it makes about the file - band table, FWHM, scale, no-data,
georef - is written into a pre-flight report the operator sees before
running anything, so a surprise is visible rather than silent.

Gotchas:
  - GDAL (rasterio) reads every cube, every interleave. Never a hand-built
    memmap: the AVIRIS-NG reader's misreads BSQ silently.
  - Band order out of this helper is ascending wavelength; reads go through
    `file_band_index`, so cube and wavelengths move together.
  - The scale decision is applied on read: reflectance = raw * factor +
    offset. Not-reflectance cubes are returned unscaled.
  - Invalid voxel = declared no-data, -9999, non-finite, or a pixel that is
    0 in every band. A lone 0 is kept unless 0 is the declared no-data.
"""

import contextlib
import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from app.abstract_classes.file_helper import FileHelper
from app.models.file_processing.generic_hsi_metadata import GenericHsiMetadata
from app.models.file_processing.sources import FileSourceConfig
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.utils.files import generic_hsi_bands as bt
from app.utils.files import generic_hsi_scale as sc
from app.utils.files.generic_hsi_georef import is_igm

logger = logging.getLogger("GenericHsiHelper")
logger.setLevel(logging.INFO)

TIFF_SUFFIXES = (".tif", ".tiff")
HDF_SUFFIXES = (".h5", ".he5", ".hdf5", ".hdf", ".nc", ".nc4", ".cdf")
# ENVI data-file extensions, most likely first; any other same-stem file is a last resort.
ENVI_DATA_SUFFIXES = (".hsi", ".img", ".bsq", ".bil", ".bip", ".dat", ".raw", ".bin", "")
_NOT_DATA_SUFFIXES = (".hdr", ".json", ".csv", ".txt", ".xml", ".aux", ".ovr", ".md5", ".sha256", ".md", ".pdf",
                      ".png", ".jpg", ".jpeg", ".kml", ".kmz", ".prj", ".log", ".zip", ".igm", ".glt", ".loc", ".obs")
ACCEPTED_SUFFIXES = TIFF_SUFFIXES + HDF_SUFFIXES + (".hdr",)

# Subdataset leaf names that are never the reflectance cube, and names that suggest it is.
_SKIP_VAR_RE = re.compile(r"uncert|error|sigma|std|noise|mask|flag|quality|angle|zenith|azimuth|lat|lon|time|elev|"
                          r"geoloc|igm|aod|water_vapo", re.IGNORECASE)
_CUBE_VAR_RE = re.compile(r"reflect|rfl|radiance|rdn|cube", re.IGNORECASE)
_ID_TIME_RE = re.compile(r"(\d{8})[T_-]?(\d{6})")
_SENSOR_HINT_RE = re.compile(r"ghost|orbital\s*sidekick|\bosk\b", re.IGNORECASE)

# Bands per validity / scaling pass: bounds the temporary bool arrays.
_BLOCK_BANDS = 16
# Always no-data when it appears, declared or not: the common fill of reflectance products.
IMPLICIT_FILL = -9999.0


@contextlib.contextmanager
def open_raster(data_path: str):
    """rasterio.open without the NotGeoreferencedWarning an ungeoreferenced cube raises."""
    import rasterio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(data_path) as src:
            yield src


def envi_cube_data(hdr: Path) -> Optional[Path]:
    """The data file an ENVI header describes: `x.img.hdr` -> `x.img`, else a same-stem file (`.hsi` first)."""
    hdr = Path(hdr)
    direct = hdr.with_suffix("")
    if direct != hdr and direct.is_file() and direct.suffix.lower() not in _NOT_DATA_SUFFIXES:
        return direct
    siblings = [p for p in hdr.parent.iterdir() if p.is_file() and p.stem == hdr.stem and p != hdr]
    for ext in ENVI_DATA_SUFFIXES:
        hit = [p for p in siblings if p.suffix.lower() == ext]
        if hit:
            return hit[0]
    rest = [p for p in siblings if p.suffix.lower() not in _NOT_DATA_SUFFIXES and not is_igm(p)]
    return rest[0] if rest else None


def pick_generic_file(paths: Sequence[Path]) -> Optional[Path]:
    """The cube among an upload's files: the largest GeoTIFF, ENVI header (by its data file) or HDF/NetCDF.

    Shared by source_path, scene onboarding and the georef resolver. `paths`
    must exist on disk (the ENVI data-file check and the sizes).
    """
    cands: List[Tuple[int, str, Path]] = []
    for p in paths:
        p = Path(p)
        suf = p.suffix.lower()
        if suf == ".hdr" and not is_igm(p):
            data = envi_cube_data(p)
            if data is not None:
                cands.append((data.stat().st_size, p.name, p))
        elif suf in TIFF_SUFFIXES + HDF_SUFFIXES:
            cands.append((p.stat().st_size, p.name, p))
    if not cands:
        return None
    return max(cands, key=lambda c: (c[0], c[1]))[2]


def choose_subdataset(container: Path) -> Tuple[str, Optional[str]]:
    """(GDAL name, variable path) of the 3-D cube with the most bands in an HDF5 / NetCDF file."""
    with open_raster(str(container)) as src:
        subs = list(src.subdatasets)
        if not subs:
            if src.count > 1:
                return str(container), None
            raise ValueError(f"{container.name}: GDAL sees no 3-D dataset in it")
    best, best_key = None, None
    for name in subs:
        var = name.rsplit(":", 1)[-1]
        leaf = var.rsplit("/", 1)[-1]
        try:
            with open_raster(name) as s:
                count, pixels = s.count, s.width * s.height
        except Exception:    # noqa: BLE001 - an unreadable subdataset is simply not a candidate
            continue
        if count < 2:
            continue
        key = (not _SKIP_VAR_RE.search(leaf), count, bool(_CUBE_VAR_RE.search(leaf)), pixels)
        if best_key is None or key > best_key:
            best, best_key = (name, var), key
    if best is None:
        raise ValueError(f"{container.name}: no 3-D (bands x rows x cols) dataset among {len(subs)} subdatasets")
    logger.info("HDF/NetCDF %s: picked subdataset %s (%d bands) of %d", container.name, best[1], best_key[1], len(subs))
    return best


def locate_data(paths: Sequence[Path]) -> Tuple[Path, str, Optional[str]]:
    """(picked file, what GDAL opens, HDF variable path) for an upload. ValueError when nothing is readable."""
    source = pick_generic_file(paths)
    if source is None:
        exts = sorted({Path(p).suffix.lower() for p in paths})
        raise ValueError(
            "no hyperspectral cube found: expected a GeoTIFF (.tif/.tiff), an ENVI .hdr with its data file "
            f"(.hsi, .img, .bsq, ...), or HDF5 / NetCDF; got extensions {exts}")
    if source.suffix.lower() == ".hdr":
        return source, str(envi_cube_data(source)), None
    if source.suffix.lower() in HDF_SUFFIXES:
        name, var = choose_subdataset(source)
        return source, name, var
    return source, str(source), None


def read_ingest_options(paths: Sequence[Path]) -> Dict[str, Any]:
    """The operator's `allotrope_ingest.json` (written by the upload form or uploaded by hand), else {}."""
    for p in paths:
        if Path(p).name.lower() == bt.INGEST_OPTIONS_NAME:
            try:
                return dict(json.loads(Path(p).read_text()))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{bt.INGEST_OPTIONS_NAME} is not a JSON object: {exc}") from exc
    return {}


def _acquisition(tags: Dict[str, str], stem: str) -> Optional[datetime]:
    for k, v in tags.items():
        kl = k.lower()
        if any(t in kl for t in ("acquisition", "datetime", "start_time", "time_coverage_start", "date_time")):
            text = str(v).strip()
            m = re.match(r"^(\d{4}):(\d{2}):(\d{2}) (\d{2}:\d{2}:\d{2})", text)    # TIFFTAG_DATETIME
            text = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}" if m else text
            try:
                at = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return at if at.tzinfo else at.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    m = _ID_TIME_RE.search(stem)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def upload_root(source: Path) -> Path:
    """Where the upload's sidecars live: the scene's `raw/` folder when the cube sits under one, else its folder."""
    return next((p for p in Path(source).parents if p.name == "raw"), Path(source).parent)


def read_generic_hsi_metadata(source_path: str | Path, root: Optional[Path] = None) -> GenericHsiMetadata:
    """Band table, scale, no-data, georef and pre-flight report of the upload holding `source_path`.

    Sidecars (wavelength file, allotrope_ingest.json, .igm) are searched for
    under `root`, default `upload_root(source_path)`.
    """
    from app.utils.files.envi_helper import _parse_envi_header
    from app.utils.files.generic_hsi_preflight import assemble

    source = Path(source_path)
    upload = sorted(p for p in Path(root or upload_root(source)).rglob("*") if p.is_file())
    options = read_ingest_options(upload)
    tokens, var = None, None
    if source.suffix.lower() == ".hdr":
        data = envi_cube_data(source)
        if data is None:
            raise ValueError(f"ENVI header {source.name} has no data file beside it")
        data_path = str(data)
        tokens = _parse_envi_header(source.read_text(errors="ignore"))
    elif source.suffix.lower() in HDF_SUFFIXES:
        data_path, var = choose_subdataset(source)
    else:
        data_path = str(source)

    root_tags: Dict[str, str] = {}
    is_hdf = source.suffix.lower() in HDF_SUFFIXES
    if is_hdf:
        with open_raster(str(source)) as container:
            root_tags = dict(container.tags())
    with open_raster(data_path) as src:
        count = src.count
        if tokens is not None and "bands" in tokens and int(tokens["bands"]) != count:
            raise ValueError(f"{Path(data_path).name}: GDAL reads {count} bands, the header says {tokens['bands']}")
        sources = [
            lambda: bt.from_band_tags(src),
            lambda: bt.from_envi_header(tokens, count),
            lambda: bt.from_container_tags(root_tags, var, count) if is_hdf else None,
            lambda: bt.from_h5py(source, var, count) if is_hdf else None,
            lambda: bt.from_descriptions(src.descriptions, count),
            lambda: bt.from_sidecar(upload, count),
        ]
        tables = []
        for get in sources:
            table = get()
            if table is not None:
                tables.append(table)
                if table.fwhm is not None and table.fwhm_exact:
                    break
        if not tables:
            raise ValueError(
                f"no wavelengths found for the {count} bands of {Path(data_path).name}: tried band metadata, the ENVI "
                "header, HDF attributes and band descriptions. Upload a wavelength file (CSV or JSON, one value per "
                "band, nm or um) beside the cube.")
        text = " ".join([json.dumps(root_tags), json.dumps(dict(src.tags())), json.dumps(src.tags(1)),
                         json.dumps(tokens or {})])
        sensor_hint = bool(_SENSOR_HINT_RE.search(text))
        band = sc.sorted_band_table(tables, sensor_hint=sensor_hint)
        fill, fill_source = sc.declared_fill(src, tokens)
        scale = sc.decide_scale(src, band, fill, tokens, root_tags, var, options)
        info = {
            "driver": src.driver, "dtype": src.dtypes[0], "height": src.height, "width": src.width, "count": count,
            "interleave": (src.tags(ns="IMAGE_STRUCTURE").get("INTERLEAVE") or (tokens or {}).get("interleave")),
            "tags": dict(src.tags()),
        }

    md_kwargs, report = assemble(source=source, data_path=data_path, var=var, upload=upload, info=info,
                                 band=band, fill=fill, fill_source=fill_source, scale=scale, options=options)
    platform = "GHOSt" if sensor_hint else ((tokens or {}).get("sensor type") or None)
    stem = Path(data_path).stem if var is None else source.stem
    override = scale.get("override_tag")
    return GenericHsiMetadata(
        source_path=str(source), data_path=data_path, driver=info["driver"],
        scene_id=f"{stem}@{override}" if override else stem,
        acquisition_at=_acquisition({**root_tags, **info["tags"], **(tokens or {})}, stem),
        bands=len(band.wavelengths), file_bands=count, height=info["height"], width=info["width"],
        dtype=info["dtype"], file_band_index=band.file_index, wavelengths_nm=band.wavelengths, fwhm_nm=band.fwhm,
        good_band=band.good, fwhm_source=band.fwhm_source, fill_value=fill,
        scale_factor=scale["factor"], scale_offset=scale["offset"], reflectance=scale["reflectance"],
        scale_decision=scale["decision"], platform=platform, preflight=report, **md_kwargs,
    )


def read_cube(md: GenericHsiMetadata, positions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """((C, H, W) float32, (C, H, W) bool data-ok) for sorted-band `positions`, scaled per the decision.

    One GDAL read for all bands: per-band reads of a pixel-interleaved file
    would scan it once per band. Invalid voxels are set to 0.0.
    """
    file_idx = [int(md.file_band_index[p]) + 1 for p in positions]
    if not file_idx:
        return np.empty((0, md.height, md.width), np.float32), np.empty((0, md.height, md.width), bool)
    with open_raster(md.data_path) as src:
        cube = src.read(indexes=file_idx, out_dtype="float32")
    ok = np.empty(cube.shape, dtype=bool)
    all_zero = np.ones(cube.shape[1:], dtype=bool)
    factor, offset = np.float32(md.scale_factor), np.float32(md.scale_offset)
    for b0 in range(0, cube.shape[0], _BLOCK_BANDS):
        block = cube[b0:b0 + _BLOCK_BANDS]
        good = np.isfinite(block) & (block != np.float32(IMPLICIT_FILL))
        if md.fill_value is not None and np.isfinite(md.fill_value):
            good &= block != np.float32(md.fill_value)
        all_zero &= np.all(block == 0, axis=0)
        if md.reflectance and (factor != 1 or offset != 0):
            block *= factor
            block += offset
        block[~good] = 0.0
        ok[b0:b0 + _BLOCK_BANDS] = good
    if all_zero.any():
        ok[:, all_zero] = False
    return cube, ok


class GenericHsiHelper(FileHelper[GenericHsiMetadata]):
    """`FileSourceConfig.source_path` is the picked file (`pick_generic_file`); sidecars sit anywhere beside it."""

    def __init__(self, file_source_config: FileSourceConfig):
        super().__init__(file_source_config=file_source_config, template={})
        self.path = file_source_config.source_path
        self._file_metadata = self._construct_metadata_structure()

    def _construct_metadata_structure(self) -> GenericHsiMetadata:
        return read_generic_hsi_metadata(self.path)

    @property
    def file_metadata(self) -> GenericHsiMetadata:
        return self._file_metadata

    @property
    def template(self) -> dict:
        return self._template

    def extract_specific_bands(
        self,
        bands: List[int] | int,
        masking_needed: Optional[bool] = False,
        spectral_family: Optional[SpectralFamily] = None,
        mode: Literal["all", "specific"] = "specific",
    ) -> np.ndarray | np.ma.MaskedArray:
        """(C, H, W) float32 in ascending-wavelength order; `bands` are sorted positions. Invalid voxels are 0."""
        md = self.file_metadata
        positions = list(range(md.bands)) if mode == "all" else ([bands] if isinstance(bands, int) else list(bands))
        cube, ok = read_cube(md, positions)
        return np.ma.masked_where(~ok, cube) if masking_needed else cube
