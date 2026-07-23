"""
Dataset builder for AVIRIS-NG ENVI-format hyperspectral scenes.

Two non-negotiables locked in during planning:

  1. The `VendableHyperspectralDataset` interface is unchanged.
     Downstream consumers (api spectrum endpoint, sharder, action_run,
     foundation inferencers, classical detectors) cannot tell whether
     the cube they received is an in-RAM ndarray (PRISMA / EnMAP style)
     or a memory-mapped view onto disk (AVIRIS-NG style).

  2. The 6.4 GB primary cube never has to live in RAM. The builder
     opens the .bin via `numpy.memmap`, streams a single pass to
     materialize a same-shape `validity_cube` as a memmap-backed .npy
     on disk next to it, and returns the vendable with both cubes
     pointing at on-disk arrays.

The "normalization" step PRISMA's builder performs (DN → reflectance)
is a no-op for AVIRIS-NG: the bytes JPL ships in the `_corr_` /
`_rfl_` cube are already in usable units. The builder samples a few
central pixels to sanity-check the product type — reflectance values
sit in [-0.5, 1.5], radiance values in [1, 200] µW · nm⁻¹ · cm⁻² · sr⁻¹.
A mismatch with the header's `description` line is non-fatal but
logged; pure-radiance products will fail downstream when fed to
reflectance-trained foundation models, but that's a model-fit issue,
not an onboarding correctness one.

Memory budget during build:
  - One band tile in RAM at a time (~16 MB for 5592×722 float32).
  - One band tile in RAM at a time for the matching validity output
    (~4 MB uint8).
  - Final returned vendable holds memmap views — ~few hundred bytes.

Disk after build:
  - Primary cube symlinked (or in-place) from the raw folder.
  - validity_cube.npy materialized next to the vendable pickle.
  - vendable.pkl carries paths + shapes + dtypes, not bytes.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pystac import Item

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.abstract_classes.file_helper import FileHelper
from app.models.dataset.vendables import BandFilterConfig, VendableHyperspectralDataset
from app.models.file_processing.sources import FileSourceConfig
from app.models.hyperspectral_concepts.band import (
    HyperSpectralBand,
    HyperpectralBandInformation,
    WavelengthMeasurementUnits,
)
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.images.cube_representation import CubeRepresentation
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter
from app.utils.data_transformations.spectral_interpolator import interpolate_spectral_gaps
from app.utils.data_transformations.spectral_resampler import resample_to_common_grid
from app.utils.files.envi_helper import ENVIHelper


logger = logging.getLogger("AvirisNGDatasetBuilder")
logger.setLevel(logging.INFO)


# Wavelength threshold (nm) separating VNIR from SWIR. Matches the
# convention PRISMA / EnMAP use in this codebase.
_VNIR_SWIR_BOUNDARY_NM = 1000.0


# Disambiguation thresholds for the product-type sanity check. Picked
# loose so legitimate scenes don't trip them — the goal is to catch
# obvious unit mismatches (radiance values 100x bigger than expected),
# not to grade reflectance quality.
_REFLECTANCE_PLAUSIBLE_MAX = 1.5
_REFLECTANCE_PLAUSIBLE_MIN = -0.5


class AvirisNGDatasetBuilder(DatasetBuilder):
    """
    AVIRIS-NG flight-line builder.

    `FileSourceConfig.source_path` must point at the scene folder
    (containing the `_corr_*` and optional `_h2o_*` .bin/.hdr pairs).
    """

    def __init__(self, file_source_configuration: FileSourceConfig):
        super().__init__(file_source_configuration=file_source_configuration)
        logger.info(
            "Initialising ENVI helper for %s", self.file_source_config.source_path
        )
        self._file_helper: ENVIHelper = ENVIHelper(
            file_source_config=self.file_source_config,
            template=ENVIHelper.default_template(),
        )
        self._band_information: Dict[
            SpectralFamily, HyperpectralBandInformation
        ] = self.extract_band_information()
        # Lazily built — we want vend_dataset() to be the only path
        # that touches the .bin.
        self._stac_item_cached: Optional[Item] = None

    # ------------------------------------------------------------------
    # DatasetBuilder contract
    # ------------------------------------------------------------------

    @property
    def file_helper(self) -> FileHelper:
        return self._file_helper

    @property
    def default_cube_representation(self) -> CubeRepresentation:
        """The ENVI .bin is BIL on disk. The vendable, by project
        convention, is BSQ — vend_dataset() materialises a BSQ memmap
        next to the original BIL file. We report BSQ here so anyone
        asking 'what shape is the vendable in?' gets the right answer
        without having to dig into the builder internals."""
        return CubeRepresentation.BSQ

    def initialize_helper(self) -> ENVIHelper:
        """The helper is already initialised in __init__ to give other
        methods (extract_band_information, vend_dataset) something to
        bind against. This accessor exists for parity with PRISMA's
        builder — both return the live helper instance."""
        return self._file_helper

    @property
    def band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation] | None:
        return self._band_information

    @property
    def stac_item(self) -> Item:
        if self._stac_item_cached is None:
            self._stac_item_cached = self._build_stac_item()
        return self._stac_item_cached

    def extract_band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        """
        Classify each of the 425 AVIRIS-NG bands into VNIR or SWIR
        using the same threshold PRISMA / EnMAP use (1000 nm). UV-edge
        bands (376-400 nm) fold into VNIR per the planning decision —
        no new family.

        Returns the canonical {family → HyperpectralBandInformation}
        shape every other hyperspectral builder produces, with the
        crucial detail that **band_index is reset within each family**.
        That's what PRISMA does too; downstream consumers expect it.
        """
        md = self._file_helper.file_metadata
        wavelengths = md.spectral.wavelengths
        fwhms = md.spectral.fwhm or [0.0] * len(wavelengths)
        bbls = md.spectral.bbl or [1] * len(wavelengths)

        units = WavelengthMeasurementUnits.NANO_METERS

        vnir_by_index: Dict[int, HyperSpectralBand] = {}
        vnir_by_wavelength: Dict[float, HyperSpectralBand] = {}
        swir_by_index: Dict[int, HyperSpectralBand] = {}
        swir_by_wavelength: Dict[float, HyperSpectralBand] = {}

        vnir_idx = 0
        swir_idx = 0
        for cw, fwhm, valid in zip(wavelengths, fwhms, bbls):
            band_kwargs = dict(
                wavelength=cw,
                wavelength_measurement_unit=units,
                full_width_at_half_maximum=fwhm,
                is_valid=bool(valid),
            )
            if cw < _VNIR_SWIR_BOUNDARY_NM:
                band = HyperSpectralBand(band_index=vnir_idx, **band_kwargs)
                vnir_by_index[vnir_idx] = band
                vnir_by_wavelength[cw] = band
                vnir_idx += 1
            else:
                band = HyperSpectralBand(band_index=swir_idx, **band_kwargs)
                swir_by_index[swir_idx] = band
                swir_by_wavelength[cw] = band
                swir_idx += 1

        logger.info(
            "AVIRIS-NG band split: VNIR=%d  SWIR=%d  (total=%d)",
            vnir_idx, swir_idx, vnir_idx + swir_idx,
        )

        return {
            SpectralFamily.VNIR: HyperpectralBandInformation(
                bands_by_index=vnir_by_index,
                bands_by_wavelength=vnir_by_wavelength,
            ),
            SpectralFamily.SWIR: HyperpectralBandInformation(
                bands_by_index=swir_by_index,
                bands_by_wavelength=swir_by_wavelength,
            ),
        }

    def vend_dataset(
        self,
        band_filter_config: Optional[BandFilterConfig] = None,
        validity_cube_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> VendableHyperspectralDataset:
        """
        Build the vendable.

        Reads the primary .bin via np.memmap (no RAM cost), validates
        the file size matches the header-declared shape, samples a few
        central pixels for product-type plausibility, streams one
        pass to materialise a same-shape uint8 validity_cube on disk
        next to the primary cube, then returns the vendable.

        Args:
          band_filter_config: When provided, applies the same
            band_filter → spatial-mask → PCHIP gap-fill → optional
            common-grid resample pipeline PRISMA runs. The filtered
            cube is materialised in RAM (the memmap is only used to
            stream the good-band slice in). When None, the vendable
            carries the native 425-band memmap-backed cube unchanged.
          validity_cube_dir: Where to write `validity_cube.npy`. None
            means "next to the primary .bin". Useful for smoke tests
            where you don't want to litter the source folder.
        """
        md = self._file_helper.file_metadata
        core = md.core
        primary_path = md.primary_cube_path

        # ---- 1. Open the memmap. ----
        dtype = _envi_dtype_to_numpy(core.data_type, core.byte_order)
        # ENVI BIL layout: line-major, then band-within-line, then sample.
        memmap_shape: Tuple[int, int, int] = (
            core.lines,
            core.bands,
            core.samples,
        )
        bil_cube = np.memmap(
            primary_path, dtype=dtype, mode="r", shape=memmap_shape
        )

        # ---- 2. Sanity check file size. ----
        bytes_per_value = np.dtype(dtype).itemsize
        expected_bytes = (
            core.lines * core.bands * core.samples * bytes_per_value
        )
        actual_bytes = os.path.getsize(primary_path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Primary cube {primary_path!r} size {actual_bytes:,} bytes "
                f"!= expected {expected_bytes:,} (lines={core.lines} × "
                f"bands={core.bands} × samples={core.samples} × "
                f"{bytes_per_value} bytes/value). File truncated or header "
                f"is wrong."
            )

        # ---- 3. Product-type plausibility check. ----
        # Sample 16 central pixels across all bands, restricted to
        # bbl=1 (good) bands. Including bbl=0 atmospheric-absorption
        # bands would dominate the value range with reconstruction
        # blow-ups even on legitimate reflectance products.
        self._check_product_type(
            bil_cube,
            md.core.data_ignore_value,
            md.core.description,
            bbl=md.spectral.bbl,
        )

        # ---- 4. Build validity_cube as memmap-backed .npy ----
        # Shape is (bands, lines, samples) — i.e. BSQ — to match the
        # canonical vendable layout. The primary cube is BIL on disk
        # but the vendable uses BSQ. We rebuild a BSQ memmap-on-disk
        # for the cube too in step 5.
        if validity_cube_dir is None:
            validity_cube_dir = Path(primary_path).parent
        validity_cube_dir = Path(validity_cube_dir)
        validity_cube_dir.mkdir(parents=True, exist_ok=True)

        validity_path = validity_cube_dir / "validity_cube.npy"
        bsq_cube_path = validity_cube_dir / "primary_cube_bsq.npy"
        bsq_shape: Tuple[int, int, int] = (
            core.bands,
            core.lines,
            core.samples,
        )

        logger.info(
            "Streaming validity_cube + BSQ cube to disk (one pass over %d bands × %d × %d)",
            core.bands, core.lines, core.samples,
        )
        bbl_array = np.asarray(md.spectral.bbl or [1] * core.bands, dtype=np.uint8)
        ignore_value = core.data_ignore_value
        band_level_validity_score = self._stream_build_companion_arrays(
            bil_cube=bil_cube,
            bsq_cube_path=bsq_cube_path,
            bsq_shape=bsq_shape,
            dtype=dtype,
            validity_path=validity_path,
            validity_shape=bsq_shape,
            bbl=bbl_array,
            ignore_value=ignore_value,
        )

        # ---- 5. Open the BSQ cube + validity cube as read-only memmaps ----
        bsq_cube_memmap = np.memmap(
            bsq_cube_path, dtype=dtype, mode="r", shape=bsq_shape
        )
        validity_memmap = np.memmap(
            validity_path, dtype=np.uint8, mode="r", shape=bsq_shape
        )

        # ---- 6. Build per-band parallel lists for the vendable ----
        wavelengths = md.spectral.wavelengths
        fwhms = md.spectral.fwhm or [0.0] * len(wavelengths)
        bbls = md.spectral.bbl or [1] * len(wavelengths)
        spectral_family_by_position: List[SpectralFamily] = [
            SpectralFamily.VNIR if cw < _VNIR_SWIR_BOUNDARY_NM else SpectralFamily.SWIR
            for cw in wavelengths
        ]
        band_validity_by_position = [int(v) for v in bbls]
        band_cw_by_position = [float(cw) for cw in wavelengths]
        band_fwhm_by_position = [float(f) for f in fwhms]

        # Default: hand back the memmap-backed cube + validity as-is.
        out_cube: np.ndarray = bsq_cube_memmap
        out_validity: np.ndarray = validity_memmap

        # ---- 7. Optional filter / interpolate / resample pipeline ----
        # Mirrors prisma_dataset_builder.vend_dataset() lines 302-379.
        # Memory note: we slice the good-band subset out of the BSQ
        # memmap into RAM. For a 165-band float32 result over a typical
        # AVIRIS-NG scene that is ~2.7 GB — workable. We never hold the
        # full 425-band cube in RAM simultaneously.
        if band_filter_config is not None:
            logger.info("Applying spectral band filtering (AVIRIS-NG).")
            band_filter = SpectralBandFilter(
                band_wavelengths=band_cw_by_position,
                band_validity_flags=band_validity_by_position,
                exclusion_ranges=band_filter_config.exclusion_ranges,
                spectral_families=spectral_family_by_position,
                edge_bands_to_trim=band_filter_config.edge_bands_to_trim,
                band_level_validity_scores=band_level_validity_score,
                min_valid_pixel_pct=band_filter_config.min_valid_pixel_pct,
            )
            good_idx = band_filter.get_good_band_indices()
            summary = band_filter.summary()
            logger.info(
                "Band filter: %d → %d bands. Dropped: %d flag, %d wavelength, "
                "%d edge, %d coverage.",
                summary["total_bands"],
                summary["surviving"],
                summary["dropped_by_flags"]["count"],
                summary["dropped_by_wavelength"]["count"],
                summary["dropped_by_edge"]["count"],
                summary["dropped_by_coverage"]["count"],
            )

            idx_arr = np.array(good_idx)
            # Write the good-band slice to disk-backed memmaps in
            # row-chunks. Fancy-indexing the source memmap with a full
            # band index in one shot would allocate ~5 GB for AVIRIS-NG;
            # the row-chunked path holds at most one chunk in RAM
            # (~190 MB at 200 rows × 722 samples × 327 bands × float32).
            n_good = idx_arr.shape[0]
            _, lines_n, samples_n = bsq_cube_memmap.shape
            filt_cube_path = Path(validity_cube_dir) / "filtered_cube.npy"
            filt_validity_path = Path(validity_cube_dir) / "filtered_validity.npy"
            filt_cube = np.memmap(
                filt_cube_path,
                dtype=bsq_cube_memmap.dtype,
                mode="w+",
                shape=(n_good, lines_n, samples_n),
            )
            filt_validity = np.memmap(
                filt_validity_path,
                dtype=np.int8,
                mode="w+",
                shape=(n_good, lines_n, samples_n),
            )
            row_chunk = 200
            for r0 in range(0, lines_n, row_chunk):
                r1 = min(r0 + row_chunk, lines_n)
                # Fancy-index pulls only the chunk's good-band slice
                # into RAM (~190 MB peak); we then write through to the
                # filtered memmap.
                filt_cube[:, r0:r1, :] = np.asarray(
                    bsq_cube_memmap[idx_arr, r0:r1, :],
                    dtype=bsq_cube_memmap.dtype,
                )
                filt_validity[:, r0:r1, :] = np.asarray(
                    validity_memmap[idx_arr, r0:r1, :],
                    dtype=np.int8,
                )
            filt_cube.flush()
            filt_validity.flush()
            spectral_family_by_position = [spectral_family_by_position[i] for i in good_idx]
            band_cw_by_position = [band_cw_by_position[i] for i in good_idx]
            band_fwhm_by_position = [band_fwhm_by_position[i] for i in good_idx]
            band_validity_by_position = [band_validity_by_position[i] for i in good_idx]
            band_level_validity_score = [band_level_validity_score[i] for i in good_idx]

            # ── Spatial masking ──
            num_bands = filt_validity.shape[0]
            valid_count_per_pixel = filt_validity.sum(axis=0)
            invalid_fraction = 1.0 - (valid_count_per_pixel / num_bands)
            bad_pixel_mask = invalid_fraction > band_filter_config.max_invalid_voxel_fraction
            num_masked = int(bad_pixel_mask.sum())
            total_pixels = bad_pixel_mask.shape[0] * bad_pixel_mask.shape[1]
            if num_masked > 0:
                filt_validity[:, bad_pixel_mask] = 0
                logger.info(
                    "Spatial masking: %d / %d pixels (%.1f%%) invalidated "
                    "(>%.0f%% voxels invalid).",
                    num_masked,
                    total_pixels,
                    num_masked / total_pixels * 100.0,
                    band_filter_config.max_invalid_voxel_fraction * 100.0,
                )
            else:
                logger.info("Spatial masking: no pixels exceeded the invalid voxel threshold.")

            # ── PCHIP spectral gap-fill ──
            interpolate_spectral_gaps(
                cube=filt_cube,
                validity_cube=filt_validity,
                band_wavelengths=np.array(band_cw_by_position, dtype=np.float64),
            )

            # ── Resample to common wavelength grid ──
            if band_filter_config.common_wavelength_grid is not None:
                # Spill the resampled cube + validity to disk-backed
                # memmaps in the same directory as the rest of the
                # AVIRIS-NG vendable artefacts. The resampler itself
                # processes the input chunk-by-row so peak RAM stays
                # under ~200 MB regardless of scene size.
                filt_cube, filt_validity, spectral_family_by_position = resample_to_common_grid(
                    cube=filt_cube,
                    validity_cube=filt_validity,
                    source_wavelengths=np.array(band_cw_by_position, dtype=np.float64),
                    target_wavelengths=band_filter_config.common_wavelength_grid,
                    spectral_families=spectral_family_by_position,
                    output_dir=validity_cube_dir,
                )
                band_cw_by_position = band_filter_config.common_wavelength_grid.tolist()
                band_validity_by_position = [1] * len(band_cw_by_position)
                # FWHM mapping isn't defined post-resample; zero it out.
                band_fwhm_by_position = [0.0] * len(band_cw_by_position)

            pixels_per_band = filt_validity.shape[1] * filt_validity.shape[2]
            band_level_validity_score = [
                float(filt_validity[b].sum() / pixels_per_band * 100.0)
                for b in range(filt_validity.shape[0])
            ]

            out_cube = filt_cube
            out_validity = filt_validity

        logger.info(
            "AVIRIS-NG vendable assembled. cube shape=%s  validity shape=%s",
            out_cube.shape, out_validity.shape,
        )

        return VendableHyperspectralDataset(
            normalized_hyperspectral_cube=out_cube,
            validity_cube=out_validity,
            spectral_family_order=spectral_family_by_position,
            band_cw_order=band_cw_by_position,
            band_fwhm_order=band_fwhm_by_position,
            band_validity_by_position=band_validity_by_position,
            band_level_validity_score=band_level_validity_score,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_product_type(
        self,
        bil_cube: np.memmap,
        ignore_value: Optional[float],
        description: Optional[str],
        bbl: Optional[List[int]] = None,
    ) -> None:
        """Sample 16 central pixels across the good (bbl=1) bands and
        characterise the value range. Including bbl=0 bands (typically
        inside atmospheric water-vapor absorption regions) would
        dominate the sample with reconstruction blow-ups even on
        legitimate reflectance products. Non-fatal: a mismatch with
        the header's `description` line is logged at WARNING but
        doesn't fail the build."""
        lines, bands, samples = bil_cube.shape
        # 4x4 grid of central pixels
        ys = np.linspace(lines * 0.4, lines * 0.6, 4, dtype=int)
        xs = np.linspace(samples * 0.4, samples * 0.6, 4, dtype=int)
        good_band_mask: Optional[np.ndarray] = None
        if bbl is not None and len(bbl) == bands:
            good_band_mask = np.asarray(bbl, dtype=bool)
        sample_vals: List[float] = []
        for y in ys:
            for x in xs:
                # 425-band spectrum for (y, x) — ~1.7 KB read
                spec = np.asarray(bil_cube[y, :, x], dtype=np.float32)
                if good_band_mask is not None:
                    spec = spec[good_band_mask]
                if ignore_value is not None:
                    spec = spec[spec != ignore_value]
                if spec.size == 0:
                    continue
                sample_vals.extend(spec.tolist())
        if not sample_vals:
            logger.warning(
                "Product-type check: all sampled pixels were ignore-value. "
                "Skipping plausibility check."
            )
            return

        arr = np.asarray(sample_vals)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return
        vmin, vmax = float(finite.min()), float(finite.max())
        looks_like_reflectance = (
            vmin >= _REFLECTANCE_PLAUSIBLE_MIN
            and vmax <= _REFLECTANCE_PLAUSIBLE_MAX
        )
        looks_like_radiance = vmax > 5.0 and vmax < 500.0

        if description and "radian" in description.lower():
            declared = "radiance"
        elif description and "reflect" in description.lower():
            declared = "reflectance"
        else:
            declared = "unknown"

        if looks_like_reflectance:
            actual = "reflectance"
        elif looks_like_radiance:
            actual = "radiance"
        else:
            actual = "neither"

        logger.info(
            "Product-type check: declared=%s · actual=%s · sample range=[%.4f, %.4f]",
            declared, actual, vmin, vmax,
        )
        if declared != "unknown" and actual != "neither" and declared != actual:
            logger.warning(
                "Header description claims %s but data looks like %s. "
                "Foundation models trained on reflectance will only work "
                "if the data is actually reflectance.",
                declared, actual,
            )

    def _stream_build_companion_arrays(
        self,
        *,
        bil_cube: np.memmap,
        bsq_cube_path: Path,
        bsq_shape: Tuple[int, int, int],
        dtype: np.dtype,
        validity_path: Path,
        validity_shape: Tuple[int, int, int],
        bbl: np.ndarray,
        ignore_value: Optional[float],
    ) -> List[float]:
        """One streaming pass over the BIL primary cube. For each
        band:

          - read the band's (lines, samples) slice from the BIL memmap
          - write it to the BSQ cube .npy at position [band]
          - build the per-pixel validity (cube != ignore_value, ANDed
            with the band's bbl flag)
          - write that to the validity .npy at position [band]
          - record the percentage of valid pixels in this band

        Returns the `band_level_validity_score` parallel list.

        Peak RAM during this method: ~2 × band-tile (one for the cube
        copy, one for the validity computation) ≈ 20 MB for AVIRIS-NG.
        """
        lines, bands, samples = bil_cube.shape
        # Pre-allocate the two output memmaps with the right shape.
        bsq_cube_out = np.memmap(
            bsq_cube_path, dtype=dtype, mode="w+", shape=bsq_shape
        )
        validity_out = np.memmap(
            validity_path, dtype=np.uint8, mode="w+", shape=validity_shape
        )
        scores: List[float] = []
        pixels_per_band = lines * samples
        for b in range(bands):
            # Read the band — BIL slice. NumPy will translate this
            # into the right stride pattern.
            band_tile = np.asarray(bil_cube[:, b, :], dtype=dtype)
            bsq_cube_out[b, :, :] = band_tile
            # Validity: cube != ignore_value AND band-is-good.
            if ignore_value is None:
                band_valid_per_pixel = np.ones_like(band_tile, dtype=np.uint8)
            else:
                band_valid_per_pixel = (
                    (band_tile != ignore_value).astype(np.uint8)
                )
            if bbl[b] == 0:
                band_valid_per_pixel[:] = 0
            validity_out[b, :, :] = band_valid_per_pixel
            valid_count = int(band_valid_per_pixel.sum())
            scores.append(valid_count / pixels_per_band * 100.0)
            if (b + 1) % 50 == 0 or b == bands - 1:
                logger.info(
                    "  streamed %d / %d bands", b + 1, bands
                )
        # Flush to disk and release writeable handles. The vendable
        # gets a fresh read-only memmap from the same paths.
        bsq_cube_out.flush()
        validity_out.flush()
        del bsq_cube_out, validity_out
        return scores

    def _build_stac_item(self) -> Item:
        """Minimal STAC item from filename + map_info. We synthesize a
        WGS-84 bbox by projecting the four scene corners through the
        (possibly rotated) UTM map info into lat/lon. The geometry is
        the axis-aligned envelope of the rotated polygon — slightly
        wider than the true footprint when the scene is rotated, but
        good enough for map placement and library filtering."""
        md = self._file_helper.file_metadata
        scene_id = md.scene_id or os.path.basename(self.file_source_config.source_path)
        properties: Dict[str, object] = {
            "platform": "AVIRIS-NG",
            "instrument": "AVIRIS-NG",
            "datetime": (
                md.acquisition_at.isoformat()
                if md.acquisition_at
                else None
            ),
        }
        bbox: Optional[List[float]] = None
        if md.map_info:
            mi = md.map_info
            properties["proj:rotation_degrees"] = mi.rotation_degrees
            properties["proj:utm_zone"] = mi.utm_zone
            properties["proj:utm_hemisphere"] = mi.utm_hemisphere
            try:
                bbox = self._compute_wgs84_bbox(
                    lines=md.core.lines,
                    samples=md.core.samples,
                    map_info=mi,
                )
            except Exception as exc:  # noqa: BLE001 — bbox is best-effort
                logger.warning(
                    "AVIRIS-NG bbox derivation failed (%s); leaving bbox unset.",
                    exc,
                )
                bbox = None
        return Item(
            id=scene_id,
            geometry=None,
            bbox=bbox,
            datetime=md.acquisition_at,
            properties=properties,
        )

    @staticmethod
    def _compute_wgs84_bbox(
        *,
        lines: int,
        samples: int,
        map_info,
    ) -> Optional[List[float]]:
        """Project the four scene corners (pixel space) through the
        ENVI `map info` affine + rotation into UTM, then UTM → WGS-84.

        Returns ``[min_lon, min_lat, max_lon, max_lat]``. ``None`` when
        UTM zone/hemisphere are missing.

        The math:
          1. Pixel (x, y) → centered offsets (dx, dy) from reference pixel.
          2. Rotate (dx, dy) by `rotation_degrees` (clockwise — ENVI
             convention).
          3. Add reference_easting/northing.
          4. Transform UTM (zone/hemisphere) → EPSG:4326 via rasterio.warp.
          5. Return the axis-aligned envelope.
        """
        from math import cos, radians, sin

        if map_info.utm_zone is None:
            return None

        # ENVI 1-based pixel indexing.
        ref_x = float(map_info.reference_pixel_x)
        ref_y = float(map_info.reference_pixel_y)
        px = float(map_info.pixel_size_x)
        py = float(map_info.pixel_size_y)
        rot = radians(float(map_info.rotation_degrees or 0.0))
        cos_r, sin_r = cos(rot), sin(rot)

        # Pixel corners (1-based). Note ENVI 'lines' grows southward in
        # the un-rotated frame, so y increases means northing decreases.
        corners_pixel = [
            (1.0,            1.0),
            (float(samples), 1.0),
            (float(samples), float(lines)),
            (1.0,            float(lines)),
        ]

        eastings: List[float] = []
        northings: List[float] = []
        for x, y in corners_pixel:
            dx = (x - ref_x) * px
            dy = (y - ref_y) * py
            # Clockwise rotation (ENVI convention) about the reference.
            rdx =  dx * cos_r + dy * sin_r
            rdy = -dx * sin_r + dy * cos_r
            eastings.append(map_info.reference_easting + rdx)
            northings.append(map_info.reference_northing - rdy)

        hem = (map_info.utm_hemisphere or "North").lower()
        epsg = (32600 if hem.startswith("n") else 32700) + int(map_info.utm_zone)
        from rasterio.warp import transform as _warp_transform

        lons, lats = _warp_transform(
            f"EPSG:{epsg}", "EPSG:4326", eastings, northings
        )
        return [
            float(min(lons)),
            float(min(lats)),
            float(max(lons)),
            float(max(lats)),
        ]


def _envi_dtype_to_numpy(data_type: int, byte_order: int) -> np.dtype:
    """ENVI's data-type code → numpy dtype. AVIRIS-NG uses 4 (float32);
    we accept the common alternatives so the helper is future-proof."""
    base_map = {
        1: "uint8",
        2: "int16",
        3: "int32",
        4: "float32",
        5: "float64",
        12: "uint16",
        13: "uint32",
    }
    if data_type not in base_map:
        raise ValueError(
            f"Unsupported ENVI data_type code {data_type}. Supported: "
            f"{sorted(base_map)}."
        )
    base = base_map[data_type]
    if byte_order == 0:
        return np.dtype("<" + np.dtype(base).str[1:])  # little-endian
    return np.dtype(">" + np.dtype(base).str[1:])      # big-endian
