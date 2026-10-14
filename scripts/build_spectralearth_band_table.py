"""
Generate app/constants/spectralearth_bands.json from an EnMAP L2A METADATA.XML.

Run once by hand; its output is committed and read by
app/models/dataset/spectralearth.py. Re-run only if the band table is ever in doubt.

SpectralEarth patches are 202-band int16 GeoTIFFs with no wavelength metadata at all, so
nothing in the files says which band is which. They are EnMAP L2A with EnMAP's 224 bands
minus the water-absorption bands 127-141 and 161-167 (1-based), the drop list published
with HySpecNet and SpectralEarth. EnMAP's band centres come from its METADATA.XML, parsed
here by the same EnmapXmlParser the EnMAP builder uses, so a SpectralEarth band means
exactly what the matching EnMAP band means everywhere else in this repo. Centres were
identical (0.0 nm spread) across the six EnMAP scenes checked on 2026-10-14, so any one
scene's sidecar serves.

Gotcha: EnMAP's VNIR and SWIR detectors overlap around 900-1000 nm, so wavelength is not
monotonic in band order; family comes from the detector channel lists, never from
wavelength.

Usage:
    python scripts/build_spectralearth_band_table.py <METADATA.XML | s3://bucket/key> [--profile NAME]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.files.enmap_xml_parser import EnmapXmlParser  # noqa: E402

# 1-based EnMAP band numbers SpectralEarth removed (water absorption, frequent no-data).
DROPPED_ENMAP_BANDS = list(range(127, 142)) + list(range(161, 168))
EXPECTED_KEPT = 202
OUT = Path(__file__).resolve().parents[1] / "app" / "constants" / "spectralearth_bands.json"


def _local_copy(source: str, profile: str | None) -> str:
    if not source.startswith("s3://"):
        return source
    import boto3

    bucket, _, key = source.removeprefix("s3://").partition("/")
    session = boto3.Session(profile_name=profile, region_name="ap-south-1")
    fd, path = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    session.client("s3").download_file(bucket, key, path)
    return path


def build(source: str, profile: str | None = None) -> dict:
    meta = EnmapXmlParser(_local_copy(source, profile)).parse()
    vnir = set(meta.detector_boundary.vnir_expected_channels)
    bands = sorted(meta.band_characterisation, key=lambda b: b.band_number)
    if len(bands) != 224:
        raise ValueError(f"expected 224 EnMAP bands, found {len(bands)}")
    kept = [b for b in bands if b.band_number not in DROPPED_ENMAP_BANDS]
    if len(kept) != EXPECTED_KEPT:
        raise ValueError(f"expected {EXPECTED_KEPT} kept bands, found {len(kept)}")
    return {
        "description": "SpectralEarth's 202 bands: EnMAP L2A minus 1-based bands 127-141 and 161-167.",
        "generated_from": source,
        "reflectance_scale": 0.0001,
        "nodata": -32768,
        "dropped_enmap_bands": DROPPED_ENMAP_BANDS,
        "bands": [
            {
                "index": i,
                "enmap_band": b.band_number,
                "wavelength_nm": b.wavelength_center,
                "fwhm_nm": b.fwhm,
                "family": "VNIR" if b.band_number in vnir else "SWIR",
            }
            for i, b in enumerate(kept)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("metadata_xml", help="local path or s3://bucket/key of an EnMAP L2A METADATA.XML")
    ap.add_argument("--profile", default=None, help="AWS profile for an s3:// source")
    args = ap.parse_args()
    table = build(args.metadata_xml, args.profile)
    OUT.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    fams = [b["family"] for b in table["bands"]]
    print(f"wrote {OUT} — {len(table['bands'])} bands, "
          f"{fams.count('VNIR')} VNIR / {fams.count('SWIR')} SWIR")


if __name__ == "__main__":
    main()
