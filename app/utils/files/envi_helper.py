"""
File helper for ENVI-format hyperspectral scenes (AVIRIS-NG).

An ENVI scene is a flat folder containing one or two pairs of files:

    <scene_id>_corr_v2m2_img.bin       # primary radiance/reflectance cube
    <scene_id>_corr_v2m2_img.hdr       # primary header (ASCII)
    <scene_id>_h2o_v2m2_img.bin        # optional auxiliary 3-band cube
    <scene_id>_h2o_v2m2_img.hdr        # optional auxiliary header

This helper does ONLY parsing. It:

  - resolves the four file paths from the scene folder via glob
  - reads the primary header text and tokenizes it into typed fields
  - populates an ENVIMetadata Pydantic object
  - does NOT open the .bin (no np.memmap — that's the builder's job)
  - does NOT classify bands by spectral family (downstream science)
  - does NOT decide whether the cube is radiance or reflectance
    (the builder samples actual pixel values to disambiguate)

Mirrors HE5Helper / EnmapHelper structure. After this helper runs we
have a typed metadata blob and four paths; the builder takes it from
there.
"""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from app.abstract_classes.file_helper import FileHelper
from app.models.file_processing.envi_metadata import (
    ENVIAuxiliaryProducts,
    ENVIHeaderCore,
    ENVIInstrumentCal,
    ENVIMapInfo,
    ENVIMetadata,
    ENVISpectralAxis,
)
from app.models.file_processing.sources import FileSourceConfig
from app.models.hyperspectral_concepts.file_components import ENVIFileComponents
from app.models.hyperspectral_concepts.references import ReferenceDefinition


logger = logging.getLogger("ENVIHelper")
logger.setLevel(logging.INFO)


# Filename suffixes for the standard AVIRIS-NG distribution. The
# helper accepts the scene folder; we glob inside it for these.
_PRIMARY_BIN_SUFFIX = "_corr_v2m2_img.bin"
_PRIMARY_HDR_SUFFIX = "_corr_v2m2_img.hdr"
_H2O_BIN_SUFFIX = "_h2o_v2m2_img.bin"
_H2O_HDR_SUFFIX = "_h2o_v2m2_img.hdr"


# Filename parse: ang YYYYMMDD t HHMMSS  e.g. ang20151219t081738
_SCENE_ID_RE = re.compile(r"(ang\d{8}t\d{6})")
_FILENAME_DT_RE = re.compile(r"ang(\d{8})t(\d{6})")


class ENVIHelper(FileHelper[ENVIMetadata]):
    """
    File helper for AVIRIS-NG ENVI folder scenes.

    `FileSourceConfig.source_path` is the path to the scene folder.
    The template is a small dict mapping the four ENVIFileComponents
    members to a ReferenceDefinition carrying the filename suffix as
    its `property_name`. See `ENVIHelper.default_template()` for the
    standard set.
    """

    def __init__(
        self,
        file_source_config: FileSourceConfig,
        template: Dict[ENVIFileComponents, ReferenceDefinition],
    ):
        super().__init__(file_source_config=file_source_config, template=template)
        self.scene_folder: str = file_source_config.source_path
        if not os.path.isdir(self.scene_folder):
            raise FileNotFoundError(
                f"ENVI scene folder not found: {self.scene_folder}"
            )

        # Resolve all four file paths up-front. Primary pair is
        # required; h2o pair is optional. If the primary pair is
        # missing we raise immediately — there's no recovery.
        self._primary_bin = self._resolve_required(ENVIFileComponents.PRIMARY_CUBE)
        self._primary_hdr = self._resolve_required(ENVIFileComponents.PRIMARY_HEADER)
        self._h2o_bin = self._resolve_optional(ENVIFileComponents.H2O_CUBE)
        self._h2o_hdr = self._resolve_optional(ENVIFileComponents.H2O_HEADER)

        # Parse the primary header once at construction time; cache.
        self._file_metadata: ENVIMetadata = self._construct_metadata_structure()

    # ------------------------------------------------------------------
    # FileHelper contract: typed metadata
    # ------------------------------------------------------------------

    def _construct_metadata_structure(self) -> ENVIMetadata:
        """Parses the primary .hdr into a typed ENVIMetadata object."""
        header_text = Path(self._primary_hdr).read_text()
        tokens = _parse_envi_header(header_text)

        core = _build_core(tokens)
        spectral = _build_spectral_axis(tokens, expected_bands=core.bands)
        map_info = _build_map_info(tokens)
        calibration = _build_calibration(tokens)
        auxiliary = _build_auxiliary(
            h2o_bin=self._h2o_bin,
            h2o_hdr=self._h2o_hdr,
        )
        scene_id, acquisition_at = _parse_scene_id_and_time(self._primary_bin)

        return ENVIMetadata(
            core=core,
            spectral=spectral,
            map_info=map_info,
            calibration=calibration,
            auxiliary=auxiliary,
            primary_cube_path=self._primary_bin,
            primary_header_path=self._primary_hdr,
            scene_id=scene_id,
            acquisition_at=acquisition_at,
        )

    @property
    def file_metadata(self) -> ENVIMetadata:
        return self._file_metadata

    @property
    def template(self) -> Dict[ENVIFileComponents, ReferenceDefinition]:
        return self._template

    # ------------------------------------------------------------------
    # Convenience path accessors. Builder uses these.
    # ------------------------------------------------------------------

    @property
    def primary_cube_path(self) -> str:
        return self._primary_bin

    @property
    def primary_header_path(self) -> str:
        return self._primary_hdr

    @property
    def h2o_cube_path(self) -> Optional[str]:
        return self._h2o_bin

    @property
    def h2o_header_path(self) -> Optional[str]:
        return self._h2o_hdr

    # ------------------------------------------------------------------
    # FileHelper contract methods we don't use directly. Provided so
    # the class is a proper concrete subclass.
    # ------------------------------------------------------------------

    def access_dataset(self, path: str) -> Any:
        """Not used by the AVIRIS-NG builder — the builder opens the
        .bin via np.memmap directly. Defined for parity with the base."""
        return None

    def extract_specific_bands(self, *args, **kwargs) -> np.ndarray:
        """No-op for AVIRIS-NG (no nested grouping to dispatch over).
        Band extraction goes through the builder's memmap path."""
        raise NotImplementedError(
            "AVIRIS-NG band extraction is performed by AvirisNGDatasetBuilder "
            "via numpy.memmap; see the builder for the read path."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_required(self, component: ENVIFileComponents) -> str:
        path = self._resolve(component)
        if path is None:
            raise FileNotFoundError(
                f"ENVI scene folder {self.scene_folder!r} is missing required "
                f"component {component.value}. Expected a file ending in "
                f"{self._suffix_for(component)!r}."
            )
        return path

    def _resolve_optional(self, component: ENVIFileComponents) -> Optional[str]:
        return self._resolve(component)

    def _resolve(self, component: ENVIFileComponents) -> Optional[str]:
        """Find the first file in the scene folder whose name ends with
        the template's suffix for this component."""
        suffix = self._suffix_for(component)
        for fn in sorted(os.listdir(self.scene_folder)):
            if fn.endswith(suffix):
                return os.path.join(self.scene_folder, fn)
        return None

    def _suffix_for(self, component: ENVIFileComponents) -> str:
        ref = self._template.get(component)
        if ref is None or ref.property_name is None:
            # Fall back to the built-in defaults — useful when callers
            # construct the helper without a custom template.
            return _DEFAULT_SUFFIX[component]
        return ref.property_name

    @staticmethod
    def default_template() -> Dict[ENVIFileComponents, ReferenceDefinition]:
        """Default suffix mapping for stock AVIRIS-NG distributions.
        Use this if you don't need to override the filename pattern."""
        from app.models.hyperspectral_concepts.references import ReferenceType

        return {
            comp: ReferenceDefinition(
                description=f"AVIRIS-NG {comp.value} filename suffix",
                reference_type=ReferenceType.DIRECT_PROPERTY_DEFINITION,
                property_name=suffix,
            )
            for comp, suffix in _DEFAULT_SUFFIX.items()
        }


_DEFAULT_SUFFIX: Dict[ENVIFileComponents, str] = {
    ENVIFileComponents.PRIMARY_CUBE: _PRIMARY_BIN_SUFFIX,
    ENVIFileComponents.PRIMARY_HEADER: _PRIMARY_HDR_SUFFIX,
    ENVIFileComponents.H2O_CUBE: _H2O_BIN_SUFFIX,
    ENVIFileComponents.H2O_HEADER: _H2O_HDR_SUFFIX,
}


# ----------------------------------------------------------------------
# Header tokenization. The ENVI header format is:
#   ENVI
#   key = value
#   key = { item, item, ... }       # possibly spanning many lines
# Strings can run across newlines until the matching '}'.
# ----------------------------------------------------------------------


def _parse_envi_header(text: str) -> Dict[str, str]:
    """Return a dict mapping lowercase key → raw value string.

    Brace-enclosed values may span multiple lines; we join them and
    strip the surrounding braces. Atomic values are stripped.
    """
    tokens: Dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if "=" not in line:
            i += 1
            continue
        key_raw, _, value_raw = line.partition("=")
        key = key_raw.strip().lower()
        value = value_raw.strip()
        # Multi-line brace block?
        if value.startswith("{") and not value.rstrip().endswith("}"):
            parts: List[str] = [value]
            i += 1
            while i < len(lines):
                part = lines[i]
                parts.append(part)
                if part.rstrip().endswith("}"):
                    i += 1
                    break
                i += 1
            value = " ".join(parts)
        else:
            i += 1
        # Strip surrounding braces if present (single-line or multi-line).
        if value.startswith("{") and value.rstrip().endswith("}"):
            value = value.strip()[1:-1].strip()
        tokens[key] = value
    return tokens


def _csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _csv_ints(s: str) -> List[int]:
    # bbl entries are written as "1.0" / "0.0" in some files; coerce.
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _build_core(tokens: Dict[str, str]) -> ENVIHeaderCore:
    return ENVIHeaderCore(
        samples=int(tokens["samples"]),
        lines=int(tokens["lines"]),
        bands=int(tokens["bands"]),
        header_offset=int(tokens.get("header offset", "0")),
        data_type=int(tokens["data type"]),
        interleave=tokens["interleave"].strip().lower(),
        byte_order=int(tokens.get("byte order", "0")),
        description=tokens.get("description"),
        wavelength_units=(
            tokens["wavelength units"].strip()
            if "wavelength units" in tokens
            else None
        ),
        data_ignore_value=(
            float(tokens["data ignore value"])
            if "data ignore value" in tokens
            else None
        ),
    )


def _build_spectral_axis(
    tokens: Dict[str, str], expected_bands: int
) -> ENVISpectralAxis:
    wavelengths = _csv_floats(tokens["wavelength"])
    if len(wavelengths) != expected_bands:
        raise ValueError(
            f"wavelength count {len(wavelengths)} != bands {expected_bands} "
            f"in primary header — file likely corrupted."
        )
    # JPL occasionally ships AVIRIS-NG headers whose `fwhm` block runs
    # longer than the band count (template / concatenation bug on
    # their side; first N entries still align with the wavelengths).
    # Tolerate it: truncate to expected_bands and log. Raise on the
    # opposite case (fewer entries than bands) — that's unrecoverable.
    fwhm: Optional[List[float]] = None
    if "fwhm" in tokens:
        fwhm_raw = _csv_floats(tokens["fwhm"])
        if len(fwhm_raw) < expected_bands:
            raise ValueError(
                f"fwhm count {len(fwhm_raw)} < bands {expected_bands} "
                f"— header truncated."
            )
        if len(fwhm_raw) > expected_bands:
            logger.warning(
                "fwhm count %d > bands %d — JPL header overrun, "
                "truncating to band count.",
                len(fwhm_raw), expected_bands,
            )
        fwhm = fwhm_raw[:expected_bands]
    # Same tolerance for bbl.
    bbl: Optional[List[int]] = None
    if "bbl" in tokens:
        bbl_raw = _csv_ints(tokens["bbl"])
        if len(bbl_raw) < expected_bands:
            raise ValueError(
                f"bbl count {len(bbl_raw)} < bands {expected_bands} "
                f"— header truncated."
            )
        if len(bbl_raw) > expected_bands:
            logger.warning(
                "bbl count %d > bands %d — JPL header overrun, "
                "truncating to band count.",
                len(bbl_raw), expected_bands,
            )
        bbl = bbl_raw[:expected_bands]
    return ENVISpectralAxis(wavelengths=wavelengths, fwhm=fwhm, bbl=bbl)


def _build_map_info(tokens: Dict[str, str]) -> Optional[ENVIMapInfo]:
    if "map info" not in tokens:
        return None
    raw = tokens["map info"]
    # `map info = { UTM , 1 , 1 , 830592.38909 , 1939122.494 , 4.0 , 4.0 ,
    #               43 , North , WGS-84 , units=Meters , rotation=90.0000000 }`
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 7:
        # Malformed — skip rather than fail; map_info isn't strictly
        # needed for downstream science.
        return None
    projection = parts[0]
    reference_pixel_x = float(parts[1])
    reference_pixel_y = float(parts[2])
    reference_easting = float(parts[3])
    reference_northing = float(parts[4])
    pixel_size_x = float(parts[5])
    pixel_size_y = float(parts[6])
    utm_zone: Optional[int] = None
    utm_hemisphere = None
    datum = None
    units: str = "Meters"
    rotation_degrees = 0.0
    # Parts 7-on are sensor-zone-and-options. Some are positional
    # (UTM zone, hemisphere, datum), some are key=value (units=Meters,
    # rotation=90.0). Walk what's left.
    tail = parts[7:]
    positional: List[str] = []
    kvs: Dict[str, str] = {}
    for p in tail:
        if "=" in p:
            k, _, v = p.partition("=")
            kvs[k.strip().lower()] = v.strip()
        else:
            positional.append(p)
    if positional:
        try:
            utm_zone = int(positional[0])
        except ValueError:
            pass
    if len(positional) >= 2 and positional[1] in ("North", "South"):
        utm_hemisphere = positional[1]
    if len(positional) >= 3:
        datum = positional[2]
    if "units" in kvs:
        units = kvs["units"]
    if "rotation" in kvs:
        try:
            rotation_degrees = float(kvs["rotation"])
        except ValueError:
            pass
    return ENVIMapInfo(
        projection=projection,
        reference_pixel_x=reference_pixel_x,
        reference_pixel_y=reference_pixel_y,
        reference_easting=reference_easting,
        reference_northing=reference_northing,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        utm_zone=utm_zone,
        utm_hemisphere=utm_hemisphere,
        datum=datum,
        units=units if units in ("Meters", "Feet") else "Meters",
        rotation_degrees=rotation_degrees,
    )


def _build_calibration(tokens: Dict[str, str]) -> ENVIInstrumentCal:
    return ENVIInstrumentCal(
        crosstrack_scatter_file=tokens.get("crosstrack scatter file"),
        spectral_scatter_file=tokens.get("spectral scatter file"),
        flat_field_file=tokens.get("flat field file"),
        wavelength_file=tokens.get("wavelength file"),
        rcc_file=tokens.get("rcc file"),
        bad_pixel_map=tokens.get("bad pixel map"),
        correction_factors=(
            _csv_floats(tokens["correction factors"])
            if "correction factors" in tokens
            else None
        ),
        smoothing_factors=(
            _csv_floats(tokens["smoothing factors"])
            if "smoothing factors" in tokens
            else None
        ),
        radiance_version=tokens.get("radiance version"),
    )


def _build_auxiliary(
    h2o_bin: Optional[str], h2o_hdr: Optional[str]
) -> ENVIAuxiliaryProducts:
    h2o_band_names: Optional[List[str]] = None
    if h2o_hdr is not None and os.path.isfile(h2o_hdr):
        try:
            aux_tokens = _parse_envi_header(Path(h2o_hdr).read_text())
            if "band names" in aux_tokens:
                h2o_band_names = [
                    p.strip()
                    for p in aux_tokens["band names"].split(",")
                    if p.strip()
                ]
        except Exception:
            # Best-effort — don't fail the whole onboarding because the
            # auxiliary header was malformed.
            h2o_band_names = None
    return ENVIAuxiliaryProducts(
        h2o_cube_path=h2o_bin,
        h2o_header_path=h2o_hdr,
        h2o_band_names=h2o_band_names,
    )


def _parse_scene_id_and_time(
    primary_path: str,
) -> tuple[Optional[str], Optional[datetime]]:
    basename = os.path.basename(primary_path)
    m = _SCENE_ID_RE.search(basename)
    scene_id = m.group(1) if m else None
    acquisition_at: Optional[datetime] = None
    if scene_id is not None:
        dt_match = _FILENAME_DT_RE.match(scene_id)
        if dt_match:
            try:
                acquisition_at = datetime.strptime(
                    dt_match.group(1) + dt_match.group(2),
                    "%Y%m%d%H%M%S",
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                acquisition_at = None
    return scene_id, acquisition_at
