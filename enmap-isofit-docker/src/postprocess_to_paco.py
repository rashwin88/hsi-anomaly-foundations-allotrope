"""
ISOFIT `apply_oe` output -> PACO-L2A-like product layout.

The downstream anomaly model was trained on DLR PACO L2A products, which have a
specific file layout and int16 reflectance encoding. This module wraps ISOFIT's
float ENVI outputs into that same shape so the model / dataloader needs no
changes.

Output layout, per scene:
  <output>/<scene_id>/
    <scene_id>-SPECTRAL_IMAGE.TIF     int16, 224 bands, BSQ GeoTIFF, scale 10000
    <scene_id>-METADATA.XML           PACO-style tags with ISOFIT provenance
    <scene_id>-QL_*.TIF, HISTORY.XML  every L1C sidecar, renamed to the L2A id
    <scene_id>-RFL_UNC.TIF            int16 uncertainty (scale 10000)
    <scene_id>-ISOFIT_STATE.TIF       per-pixel atmosphere (AOT550, H2OSTR, ...)
    <scene_id>-PROVENANCE.json        ISOFIT config hash, version, timestamps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from rasterio.enums import Compression

log = logging.getLogger("postprocess_to_paco")

REFL_SCALE = 10000
NODATA_INT16 = -32768  # PACO's nodata


def _read_envi(bin_path: Path) -> tuple[np.ndarray, dict]:
    """Read an ENVI file via rasterio (GDAL ENVI driver)."""
    with rasterio.open(bin_path) as src:
        arr = src.read()  # (bands, rows, cols)
        profile = src.profile
    return arr, profile


def find_isofit_outputs(isofit_dir: Path) -> dict[str, Path]:
    """apply_oe writes into <workdir>/output/... — locate the products."""
    outputs = {}
    for name in ["rfl", "uncert", "atm_interp"]:
        # apply_oe canonical names. analytical_line writes no full-image *_state;
        # its per-pixel atmosphere is *_atm_interp, band names in the header.
        for candidate in sorted(isofit_dir.rglob(f"*_{name}")):
            # *_subs_* are the sparse superpixel retrievals apply_oe produces
            # before the empirical line; the full-image cube is the one without.
            if "_subs_" in candidate.name:
                continue
            if candidate.is_file() and not candidate.suffix == ".hdr":
                outputs[name] = candidate
                break
    if "rfl" not in outputs:
        raise FileNotFoundError(f"No ISOFIT reflectance output under {isofit_dir}")
    return outputs


def scale_to_int16(arr: np.ndarray, scale: int = REFL_SCALE) -> np.ndarray:
    """Scale float reflectance (0-1) to int16, preserving nodata."""
    out = np.full(arr.shape, NODATA_INT16, dtype=np.int16)
    valid = np.isfinite(arr) & (arr > -1)
    out[valid] = np.clip(arr[valid] * scale, -32767, 32767).astype(np.int16)
    return out


def write_geotiff(arr: np.ndarray, ref_profile: dict, out_path: Path,
                  dtype: str = "int16", nodata: float | int | None = NODATA_INT16,
                  band_names: list[str] | None = None) -> None:
    """Write a multiband GeoTIFF using rasterio."""
    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        dtype=dtype,
        count=arr.shape[0],
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        interleave="band",
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(arr.shape[0]):
            dst.write(arr[i], i + 1)
            if band_names:
                dst.set_band_description(i + 1, band_names[i])


def copy_l1c_ancillaries(l1c_dir: Path, out_dir: Path, l1c_id: str,
                         scene_id: str) -> list[Path]:
    """Copy every L1C sidecar except the radiance cube and its metadata.

    PACO's L2A ships the same QL masks, quicklooks and HISTORY.XML as the L1C,
    so taking everything by prefix matches its file set by construction.
    """
    copied = []
    for src in sorted(l1c_dir.glob(f"{l1c_id}-*")):
        suffix = src.name[len(l1c_id) + 1:]
        if suffix in ("SPECTRAL_IMAGE.TIF", "METADATA.XML"):
            continue
        dst = out_dir / f"{scene_id}-{suffix}"
        shutil.copy2(src, dst)
        copied.append(dst)
    log.info("Copied %d L1C ancillary files", len(copied))
    return copied


def write_paco_metadata(l1c_xml: Path, out_xml: Path, scene_id: str,
                        season: str, isofit_version: str,
                        provenance: dict) -> None:
    """Take the L1C METADATA.XML as a base, override AC-related fields with
    ISOFIT provenance so downstream code sees a valid PACO-style metadata file.
    """
    tree = ET.parse(l1c_xml)
    root = tree.getroot()

    # PACO writes an <atmosphericCorrection> block; if the L1C parent doesn't
    # have one, append one at the top level.
    ns = ""  # keep the L1C namespaces if any; simplest: create a sibling block.
    ac = ET.SubElement(root, "atmosphericCorrection")

    def _tag(parent: ET.Element, name: str, text: str) -> None:
        el = ET.SubElement(parent, name)
        el.text = str(text)

    _tag(ac, "correction_type", "Land_Mode")
    _tag(ac, "water_reflectance_product", "NA")
    _tag(ac, "water_type", "NA")
    _tag(ac, "terrain_correction", "Yes")
    _tag(ac, "cirrus_haze_removal", "No")
    _tag(ac, "smile_correction_applied", "no")
    _tag(ac, "band_interpolation", "No")
    _tag(ac, "season", season)
    _tag(ac, "dem_database", "Copernicus_GLO30_OCEAN")
    _tag(ac, "caltab_atm_version", f"ISOFIT-{isofit_version}-sRTMnet_v100")
    _tag(ac, "processor_version", f"ENMAP_ISOFIT_PIPE_{provenance['pipeline_version']}")
    _tag(ac, "image_resampling", "Bilinear_Interpolation")
    _tag(ac, "reflectance_unit",
         f"int16, unit: reflectance x {REFL_SCALE}, nodata {NODATA_INT16}")
    _tag(ac, "processed_at_utc",
         provenance["processed_at_utc"])
    _tag(ac, "isofit_config_sha256", provenance["config_sha256"])

    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    log.info("Wrote PACO-style metadata: %s", out_xml)


def get_isofit_version() -> str:
    try:
        import isofit
        return getattr(isofit, "__version__", "unknown")
    except ImportError:
        return "unknown"


def sha256_of_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes()[:1024])  # first 1 KB per file - fingerprint only
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isofit-dir", required=True, help="ISOFIT working directory")
    p.add_argument("--l1c-dir", required=True, help="Source EnMAP L1C dir (for ancillaries)")
    p.add_argument("--output", required=True, help="Output root")
    p.add_argument("--season", required=True, choices=["summer", "winter"])
    p.add_argument("--pipeline-version", default="1.0.0")
    args = p.parse_args(argv)

    isofit_dir = Path(args.isofit_dir)
    l1c_dir = Path(args.l1c_dir)
    output = Path(args.output)

    # -- locate scene id + L1C metadata --
    xml_candidates = list(l1c_dir.glob("*METADATA.XML"))
    if not xml_candidates:
        raise FileNotFoundError(f"No METADATA.XML in {l1c_dir}")
    l1c_xml = xml_candidates[0]
    l1c_id = l1c_xml.stem.replace("-METADATA", "").replace("_METADATA", "")
    scene_id = l1c_id.replace("____L1C-", "____L2A-")  # PACO names by level

    scene_out = output / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    log.info("Scene: %s", scene_id)

    # -- locate ISOFIT outputs --
    outs = find_isofit_outputs(isofit_dir)
    log.info("ISOFIT outputs: %s", {k: str(v) for k, v in outs.items()})

    # -- reflectance -> int16 GeoTIFF --
    rfl, prof = _read_envi(outs["rfl"])
    log.info("Reflectance shape: %s, dtype=%s, range=[%.3f, %.3f]",
             rfl.shape, rfl.dtype, np.nanmin(rfl), np.nanmax(rfl))
    rfl_i16 = scale_to_int16(rfl)
    write_geotiff(rfl_i16, prof,
                  scene_out / f"{scene_id}-SPECTRAL_IMAGE.TIF",
                  dtype="int16", nodata=NODATA_INT16)

    # -- uncertainty (if present) --
    if "uncert" in outs:
        unc, _ = _read_envi(outs["uncert"])
        unc_i16 = scale_to_int16(unc)
        write_geotiff(unc_i16, prof,
                      scene_out / f"{scene_id}-RFL_UNC.TIF",
                      dtype="int16", nodata=NODATA_INT16)

    # -- retrieved atmospheric state (H2O, AOT) --
    if "atm_interp" in outs:
        with rasterio.open(outs["atm_interp"]) as src:
            atm = src.read().astype(np.float32)
            names = [d or f"atm_{i + 1}" for i, d in enumerate(src.descriptions)]
        # Same grid as rfl; take rfl's profile since ISOFIT may omit map info here.
        write_geotiff(atm, prof, scene_out / f"{scene_id}-ISOFIT_STATE.TIF",
                      dtype="float32", nodata=-9999.0, band_names=names)

    # -- copy L1C ancillaries --
    copy_l1c_ancillaries(l1c_dir, scene_out, l1c_id, scene_id)

    # -- provenance sidecar --
    provenance = {
        "pipeline_version": args.pipeline_version,
        "isofit_version": get_isofit_version(),
        "srtmnet_zenodo_doi": "10.5281/zenodo.4096627",
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "config_sha256": sha256_of_dir(isofit_dir / "config") if (isofit_dir / "config").exists() else "n/a",
        "l1c_source": str(l1c_dir),
    }
    (scene_out / f"{scene_id}-PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2)
    )

    # -- PACO-style METADATA.XML --
    write_paco_metadata(l1c_xml,
                        scene_out / f"{scene_id}-METADATA.XML",
                        scene_id, args.season,
                        provenance["isofit_version"], provenance)

    log.info("Wrote L2A product: %s", scene_out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
