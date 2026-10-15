"""
Build data/family_library: man-made and natural spectra, tagged by family, for family_match.

The output has exactly the layout of data/splib07_slim (index.json, wavelengths_asd_nm.txt,
fwhm/, spectra/), so library.load_slim_bundle / build_sensor_cache / build_cache_for_vendable
use it unchanged. Each entry's chapter is "<group>.<family>" (e.g. "manmade.roofing"), which
family_library.parse_tag reads back after the per-sensor cache.

Why a second bundle: splib07_slim has 3 usable man-made spectra. Its curation keeps only
ASD-instrument files, but most USGS man-made spectra were measured on the Beckman (BECK,
200-3000 nm, which does cover our sensors), and ECOSTRESS holds concrete, asphalt, brick,
roofing, paint, metal, glass and wood. This bundle carries both.

Sources (both read straight from their zips, nothing extracted):
  - USGS splib07b, convolved to ASD (ASCIIdata_splib07b_cvASD): every instrument, already on
    the 2151-channel ASD axis. Chapters A, C, L, M, S, V; O (lab organics) is excluded.
  - ECOSTRESS spectral library (ecospeclib-all), reflectance only, no thermal-only files,
    interpolated onto the ASD axis inside each spectrum's measured range, NaN outside.

Curation = the rules of the man-made library experiments (research branch
vinoth/hypersl-unmixing, manmade-library-unmixing/step1_build_library.py), unchanged:
pigments, ore, lab chemicals and calibration targets are excluded; an entry must cover >= 95 %
of the 165-band common grid; then each family is capped at FAMILY_CAP spectrally spread
entries (family_library.balanced_subset).

usage: python scripts/curate_family_library.py <usgs cvASD zip> <ecostress zip> [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
from app.spectral_match.family_library import FAMILY_CAP, TaggedLibrary, balanced_subset
from app.spectral_match.resample import gaussian_resample_to_target

VERSION = "family_v1"
SLIM = Path("data/splib07_slim")          # source of the ASD axis and ASD FWHM files
MIN_GRID_COVERAGE = 0.95
SENTINEL = "-1.23e+34"
USGS_WL_FILE = "ASCIIdata_splib07b_cvASD/s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
USGS_CHAPTERS = {"A", "C", "L", "M", "S", "V"}
ECO_UNITS = {"Reflectance (percent)", "Reflectance (percentage)"}

# USGS chapter A, first match wins; excludes first so "Iron_oxide" never reaches "metal".
CHAPTER_A_RULES = [
    (r"spectralon|mylar|vermicul|wtc_dust|sludge|sldg|slime|kerogen|montmorill", "exclude", "lab_or_special"),
    (r"oxide|chloride|nitrate|sulfate|cyanide|powder|phosphoric|sulfide", "exclude", "lab_chemical"),
    (r"ochre|vermill|ultramarine|venetian|viridian|pozzuoli|verdigris|prussian|phthalo|lazurite"
     r"|malachite|madder|umber|sienna|cadmium|cobalt|cerulean|naples|lead|bone_black|alizarin"
     r"|brilliant|manganese_violet", "exclude", "pigment"),
    (r"^oil|oiled", "manmade", "oil"),
    (r"sheet_?metal|aluminum|rusted_tin|steel", "manmade", "metal"),
    (r"asphalt_shingle|asphalt_tar|tar_paper|roofing_felt|roofng|roofing|clear_roof|cedar_shake",
     "manmade", "roofing"),
    (r"asphalt|concrete_gds375|brick.*pav", "manmade", "pavement"),
    (r"brick|cinder|concrete", "manmade", "building"),
    (r"plastic|polystyrene|tyvek|visqueen|fiberglass", "manmade", "plastic"),
    (r"fabric|carpet|webbing|pile|terrycloth|burlap", "manmade", "textile"),
    (r"wood|plywood|board|paper", "manmade", "wood_paper"),
]
# ECOSTRESS man-made, on the 2nd and 3rd file-name tokens. Marble is calcite rock: tagged
# man-made it would make natural carbonate outcrops look man-made.
ECO_MANMADE_RULES = [
    (r"^reflectancetarget", "exclude", "calibration_target"),
    (r"\.marble", "exclude", "natural_lookalike"),
    (r"^concrete\.pavingconcrete|^road", "manmade", "pavement"),
    (r"^concrete", "manmade", "building"),
    (r"^roofingmaterial\.metal", "manmade", "metal"),
    (r"^roofingmaterial", "manmade", "roofing"),
    (r"\.paint", "manmade", "paint"),
    (r"\.glas", "manmade", "glass"),
    (r"\.wood", "manmade", "wood_paper"),
    (r"^generalconstructionmaterial", "manmade", "building"),
]
ECO_NATURAL = {"vegetation": "vegetation", "nonphotosyntheticvegetation": "npv", "soil": "soil",
               "mineral": "mineral", "rock": "rock", "water": "water"}


def _first(text, rules):
    for pattern, group, family in rules:
        if re.search(pattern, text):
            return group, family
    return "exclude", "unsorted"


def classify_usgs(chapter: str, name: str) -> tuple[str, str]:
    low = re.sub(r"\s+", "_", name.lower())
    if chapter == "A":
        return _first(low, CHAPTER_A_RULES)
    if chapter == "L":
        return "natural", ("snow_ice" if re.search(r"snow|ice", low) else "water")
    if chapter in ("C", "M"):
        return "natural", "mineral"
    if chapter == "S":
        return "natural", ("soil" if re.search(r"sand|playa|mud|burn", low) else "mineral")
    if "lawngrass_gds91b" in low:            # shifted / offset copies of one grass spectrum
        return "exclude", "synthetic_variant"
    if "marsh_water" in low or "marshwater" in low:
        return "natural", "water"
    if re.search(r"npv|dry|wrack|dead|tumbleweed|litter", low):
        return "natural", "npv"
    return "natural", "vegetation"


def classify_ecostress(fname: str) -> tuple[str, str]:
    parts = fname.split(".")
    if parts[0] == "manmade":
        return _first(".".join(parts[1:3]), ECO_MANMADE_RULES)
    if parts[0] == "water" and re.search(r"snow|ice", fname):
        return "natural", "snow_ice"
    return ("natural", ECO_NATURAL[parts[0]]) if parts[0] in ECO_NATURAL else ("exclude", parts[0])


def read_usgs(zf: zipfile.ZipFile) -> list[dict]:
    out = []
    for info in zf.infolist():
        m = re.search(r"/Chapter([A-Z])_[^/]+/(s07_ASD_.+_(?:AREF|RREF)\.txt)$", info.filename)
        if not m or m.group(1) not in USGS_CHAPTERS:
            continue
        lines = zf.read(info).decode("latin-1").splitlines()
        refl = np.asarray([float(v) for v in lines[1:] if v.strip()])
        refl[refl < -1e30] = np.nan
        name = lines[0].split(":", 1)[-1].strip()
        out.append(dict(source="usgs", file=m.group(2), name=name, tag=classify_usgs(m.group(1), name),
                        refl=refl))
    return out


def read_ecostress(zf: zipfile.ZipFile, asd_wl: np.ndarray) -> list[dict]:
    out = []
    for info in zf.infolist():
        fname = info.filename.rsplit("/", 1)[-1]
        if not fname.endswith(".spectrum.txt") or ".tir." in fname:
            continue
        header, xs, ys = {}, [], []
        for line in zf.read(info).decode("latin-1").splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line[0] == "-"):
                x, y = line.split()[:2]
                xs.append(float(x) * 1000.0)
                ys.append(float(y) / 100.0)
            elif ":" in line:
                k, _, v = line.partition(":")
                header[k.strip()] = v.strip()
        if header.get("Y Units") not in ECO_UNITS or len(xs) < 2:
            continue
        o = np.argsort(xs)
        wl, y = np.asarray(xs)[o], np.asarray(ys)[o]
        refl = np.interp(asd_wl, wl, y)
        refl[(asd_wl < wl[0]) | (asd_wl > wl[-1])] = np.nan
        out.append(dict(source="ecostress", file=fname, name=header.get("Name", fname),
                        tag=classify_ecostress(fname), refl=refl))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("usgs_zip")
    ap.add_argument("ecostress_zip")
    ap.add_argument("--out", default="data/family_library")
    a = ap.parse_args()
    out = Path(a.out)
    asd_wl = np.loadtxt(SLIM / "wavelengths_asd_nm.txt", skiprows=1)
    with zipfile.ZipFile(a.usgs_zip) as zf:
        usgs_wl = np.asarray([float(v) for v in zf.read(USGS_WL_FILE).decode("latin-1").splitlines()[1:]
                              if v.strip()]) * 1000.0
        assert np.allclose(usgs_wl, asd_wl, atol=0.01), "cvASD axis differs from the slim bundle's"
        items = read_usgs(zf)
    with zipfile.ZipFile(a.ecostress_zip) as zf:
        items += read_ecostress(zf, asd_wl)
    print("read:", dict(Counter(i["source"] for i in items)))

    grid = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, dtype=np.float64)
    kept = []
    for it in items:
        if it["tag"][0] == "exclude":
            continue
        g = gaussian_resample_to_target(asd_wl, it["refl"], grid, np.full(len(grid), 10.0))
        if np.isfinite(g).mean() >= MIN_GRID_COVERAGE:
            it["grid"] = np.nan_to_num(g)
            kept.append(it)
    lib = TaggedLibrary(names=np.array([f"{i['source']}:{i['file']}" for i in kept]),
                        refl=np.stack([i["grid"] for i in kept]).astype(np.float32),
                        valid=np.ones((len(kept), len(grid)), np.uint8),
                        group=np.array([i["tag"][0] for i in kept]),
                        family=np.array([i["tag"][1] for i in kept]))
    cap = balanced_subset(lib, FAMILY_CAP)
    print(f"tagged and covering the grid: {len(kept)}; after the {FAMILY_CAP}-per-family cap: {cap.sum()}")

    if out.exists():
        shutil.rmtree(out)
    (out / "fwhm").mkdir(parents=True)
    shutil.copy(SLIM / "wavelengths_asd_nm.txt", out / "wavelengths_asd_nm.txt")
    shutil.copy(SLIM / "fwhm" / "ASDFR.txt", out / "fwhm" / "ASDFR.txt")
    entries = []
    for it in (k for k, c in zip(kept, cap) if c):
        chapter = f"{it['tag'][0]}.{it['tag'][1]}"
        stem = re.sub(r"[^A-Za-z0-9._+-]", "_", Path(it["file"]).stem)[:120]
        rel = f"spectra/{chapter}/{it['source']}_{stem}.txt"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        vals = "\n".join(SENTINEL if not np.isfinite(v) else f"{v:.5g}" for v in it["refl"])
        (out / rel).write_text(f" {it['source']} {it['file']}: {it['name']}\n{vals}\n")
        # asd_subtype ASDFR: the cache builder requires an ASD FWHM to exist, though it
        # resamples with the sensor's FWHM, not the library's (library.build_sensor_cache).
        entries.append({"name": f"{it['source']}:{it['file']}", "material": it["name"],
                        "chapter": chapter, "asd_subtype": "ASDFR", "path": rel,
                        "source": it["source"], "group": it["tag"][0], "family": it["tag"][1]})
    counts = Counter(e["chapter"] for e in entries)
    index = {"version": VERSION, "wavelengths_file": "wavelengths_asd_nm.txt",
             "fwhm_files": {"ASDFR": "fwhm/ASDFR.txt"}, "wavelength_unit": "nm",
             "n_entries": len(entries), "n_channels": int(asd_wl.size),
             "per_chapter_counts": dict(sorted(counts.items())), "entries": entries}
    (out / "index.json").write_text(json.dumps(index, indent=2))
    for c, n in sorted(counts.items()):
        src = Counter(e["source"] for e in entries if e["chapter"] == c)
        print(f"  {c:28s} {n:3d}  {dict(src)}")


if __name__ == "__main__":
    main()
