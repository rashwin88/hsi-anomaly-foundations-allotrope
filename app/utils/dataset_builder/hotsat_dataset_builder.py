"""
Dataset builder for SatVu HotSat-1 L2 Visual scenes.

A HotSat L2 Visual scene is a small (~1300 x 1600 pixels) single-band
thermal raster delivered as an orthorectified GeoTIFF alongside a
UDM (Usable Data Mask) GeoTIFF that carries pixel-quality bit flags.
Per SatVu's Imagery User Guide v1.0:

  - The product is **uncalibrated DN** ("Pixel values delivered as
    Digital Numbers (DNs) representing relative radiance differences
    within a scene"). No DN→Kelvin or DN→radiance transform is
    published with the L2 Visual deliverable. The vendable carries DN
    verbatim and stamps ``units="DN_14bit_relative"`` so downstream
    consumers don't accidentally treat DN as temperature.

  - The UDM is a uint8 bit-mask with four flags (1 = bad pixel,
    2 = cloud, 4 = saturated, 8 = no data). A pixel is "valid" iff
    the UDM byte is exactly 0.

Two data assets are shipped — a single-frame ortho and a 25-frame
median-stacked variant. The stacked image has lower noise but lives
on a slightly larger pixel grid that doesn't 1:1 align with the UDM.
For v1 we use the **ortho** as the primary cube source because it
shares the UDM's grid exactly (no reprojection needed). A future
refinement can prefer stacked + nearest-neighbour-reproject the UDM.

Memory profile:
  - HotSat scenes are tiny (~1300 × 1600 × uint16 ≈ 4 MB).
  - Everything stays in RAM; no memmap dance.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import rasterio
from pystac import Item

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.abstract_classes.file_helper import FileHelper
from app.models.dataset.vendables import VendableThermalDataset
from app.models.file_processing.hotsat_metadata import HotSatUDMFlags
from app.models.file_processing.sources import FileSourceConfig
from app.models.hyperspectral_concepts.band import HyperpectralBandInformation
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.images.cube_representation import CubeRepresentation
from app.utils.files.hotsat_helper import HotSatHelper


logger = logging.getLogger("HotSatDatasetBuilder")
logger.setLevel(logging.INFO)


# Units tag stamped on the vendable. Keep this aligned with the
# allowed values documented on `VendableThermalDataset.units`.
_DN_UNITS = "DN_14bit_relative"


class HotSatDatasetBuilder(DatasetBuilder):
    """
    HotSat-1 L2 Visual scene builder.

    ``FileSourceConfig.source_path`` must point at the scene folder
    (containing ``metadata.json``, the ortho and stacked GeoTIFFs,
    and the UDM raster).
    """

    def __init__(self, file_source_configuration: FileSourceConfig):
        super().__init__(file_source_configuration=file_source_configuration)
        logger.info(
            "Initialising HotSat helper for %s", self.file_source_config.source_path
        )
        self._file_helper: HotSatHelper = HotSatHelper(
            file_source_config=self.file_source_config,
            template=HotSatHelper.default_template(),
        )
        self._stac_item_cached: Optional[Item] = None

    # ------------------------------------------------------------------
    # DatasetBuilder contract
    # ------------------------------------------------------------------

    @property
    def file_helper(self) -> FileHelper:
        return self._file_helper

    @property
    def default_cube_representation(self) -> CubeRepresentation:
        """Thermal cubes are always BSQ (one or a few bands stacked
        along the band axis). HotSat is single-band but we still
        carry it in BSQ shape ``(1, H, W)`` so downstream thermal
        consumers (Landsat-trained models, RX, render code) work
        without special-casing."""
        return CubeRepresentation.BSQ

    def initialize_helper(self) -> HotSatHelper:
        """The helper is already initialised in ``__init__`` to give
        other methods (``vend_dataset``) something to bind against.
        Accessor for parity with the other builders."""
        return self._file_helper

    @property
    def band_information(
        self,
    ) -> Optional[Dict[SpectralFamily, HyperpectralBandInformation]]:
        """HotSat is single-band thermal — no per-family information."""
        return None

    def extract_band_information(
        self,
    ) -> Optional[Dict[SpectralFamily, HyperpectralBandInformation]]:
        return None

    @property
    def stac_item(self) -> Item:
        if self._stac_item_cached is None:
            self._stac_item_cached = self._build_stac_item()
        return self._stac_item_cached

    def vend_dataset(self, **kwargs) -> VendableThermalDataset:
        """Build the thermal vendable.

        Reads the ortho GeoTIFF and the UDM raster, derives the
        validity / cloud masks from the UDM bit-flags, and returns a
        ``VendableThermalDataset`` with ``units="DN_14bit_relative"``.

        ``kwargs`` is accepted for parity with other builders; HotSat
        currently has no configurable knobs.
        """
        md = self._file_helper.file_metadata

        # --- Primary cube ---
        primary_path = self._file_helper.primary_visual_path
        logger.info("Reading HotSat ortho raster: %s", primary_path)
        with rasterio.open(primary_path) as src:
            cube_2d = src.read(1)  # (H, W) uint16
            primary_nodata = src.nodata
            primary_shape = src.shape  # (H, W)
        cube = cube_2d[np.newaxis, :, :]  # BSQ shape (1, H, W)
        logger.info(
            "HotSat ortho shape=%s dtype=%s nodata=%s",
            cube.shape, cube.dtype, primary_nodata,
        )

        # --- UDM (optional but expected for L2 Visual) ---
        udm_path = self._file_helper.udm_path
        if udm_path is not None and os.path.isfile(udm_path):
            udm_2d = self._read_udm_aligned(udm_path, primary_shape)
        else:
            logger.warning(
                "HotSat scene has no UDM; falling back to nodata-only validity. "
                "Cloud mask will be all-clear."
            )
            udm_2d = np.zeros(primary_shape, dtype=np.uint8)

        # --- Derived masks from UDM ---
        # All flags ⇒ pixel is invalid for radiometric use.
        any_flag = udm_2d != 0  # bool (H, W)
        # nodata sentinel from the raster also marks invalid pixels —
        # SatVu uses 0 for off-swath in the primary cube. Combine.
        nodata_mask = (
            (cube_2d == primary_nodata) if primary_nodata is not None else np.zeros_like(cube_2d, dtype=bool)
        )
        pure_invalid = any_flag | nodata_mask
        pure_valid = ~pure_invalid

        # Cloud mask: 1 = clear, 0 = cloud. Follows the Landsat-style
        # convention `VendableThermalDataset.cloud_mask` documents.
        cloud_bit = (udm_2d & int(HotSatUDMFlags.CLOUD)).astype(bool)
        cloud_mask = (~cloud_bit).astype(np.int8)

        # Pure validity mask: 1 = pixel is usable in principle (not
        # off-swath, not bad pixel, not saturated). 0 otherwise.
        pure_validity_mask = pure_valid.astype(np.int8)

        # Overall validity used by downstream detectors: usable AND
        # clear of cloud. Shape (1, H, W) to match the cube.
        overall_validity_2d = (pure_validity_mask.astype(bool) & cloud_mask.astype(bool)).astype(np.int8)
        validity_cube = overall_validity_2d[np.newaxis, :, :]

        # Preserve the raw UDM for downstream actions that want to
        # inspect individual flags (saturated, bad pixel). Stored as
        # ``custom_quality_mask`` to reuse the existing schema slot
        # — same role Landsat's QA_PIXEL-derived mask plays.
        custom_quality_mask = udm_2d.astype(np.uint8)

        logger.info(
            "HotSat masks: pure_valid=%d / cloud_clear=%d / overall_valid=%d / total=%d",
            int(pure_validity_mask.sum()),
            int(cloud_mask.sum()),
            int(overall_validity_2d.sum()),
            int(pure_validity_mask.size),
        )

        return VendableThermalDataset(
            normalized_thermal_cube=cube,
            validity_cube=validity_cube,
            pure_validity_mask=pure_validity_mask,
            cloud_mask=cloud_mask,
            custom_quality_mask=custom_quality_mask,
            units=_DN_UNITS,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_udm_aligned(
        self, udm_path: str, primary_shape: tuple,
    ) -> np.ndarray:
        """Read the UDM raster and verify it shares the primary
        cube's grid. The vendor ships the ortho and the UDM on
        identical grids, so we just read + assert shape match. If they
        ever diverge we should grow this to a nearest-neighbour
        reproject (uint8 bit-flags, can't bilinear-interpolate)."""
        with rasterio.open(udm_path) as src:
            udm = src.read(1)  # (H, W) uint8
            if src.shape != primary_shape:
                # Vendor invariant violated. Fail loudly rather than
                # silently misaligning quality flags.
                raise ValueError(
                    f"HotSat UDM grid {src.shape} does not match primary "
                    f"cube grid {primary_shape}. Re-projection of bit-flag "
                    f"masks isn't implemented yet."
                )
        return udm.astype(np.uint8)

    def _build_stac_item(self) -> Item:
        """Build a pystac Item from the metadata.json. Most fields
        translate directly; the helper already parsed bbox, datetime,
        geometry and the projection block."""
        md = self._file_helper.file_metadata
        properties: Dict[str, object] = {
            "platform": md.platform or "hotsat-1",
            "instrument": "hotsat-1",
            "gsd": md.gsd_meters,
            "eo:cloud_cover": md.cloud_cover_pct,
            "proj:epsg": md.projection.epsg,
            "proj:shape": list(md.projection.shape),
            "proj:transform": list(md.projection.transform),
            "proj:bbox": list(md.projection.proj_bbox),
            "view:azimuth": md.view.view_azimuth,
            "view:off_nadir": md.view.view_off_nadir,
            "view:sun_azimuth": md.view.sun_azimuth,
            "view:sun_elevation": md.view.sun_elevation,
            "processing:software": md.processing_software,
            "datetime": md.acquisition_at.isoformat() if md.acquisition_at else None,
            # Provenance for downstream consumers that want to know
            # the units before opening the pickle.
            "satvu:product_units": _DN_UNITS,
            "satvu:product_collection": md.collection,
        }
        # Drop None-valued keys so the STAC item is tidy.
        properties = {k: v for k, v in properties.items() if v is not None}

        item_dt: Optional[datetime] = md.acquisition_at
        return Item(
            id=md.scene_id,
            geometry=md.geometry or None,
            bbox=list(md.bbox_wgs84),
            datetime=item_dt,
            properties=properties,
        )
