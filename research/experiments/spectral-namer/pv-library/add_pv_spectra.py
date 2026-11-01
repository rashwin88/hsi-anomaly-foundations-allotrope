"""
Add in-scene photovoltaic-array spectra to the family library as `manmade.glass` entries (class glass_solar).

    python add_pv_spectra.py <pv_spectra.json> <library_dir> <out_dir> [--holdout 0.2]

Where it sits: data/family_library is read by family matching and by the spectral namer's library bank
(app/spectral_match/namer/bank.py), whose class_of maps family "glass" to glass_solar. The namer's level 1 and its
classes are decided by matching against library embeddings, so new spectra change its calls with no retraining
(docs/proposals/class-driven-flow.md addition B). Provenance and the result: README.md beside this file.

Why a separate script: scripts/curate_family_library.py builds the library from USGS / ECOSTRESS and would drop these
entries on a rebuild; rerun this after it.

Processing, as the curated ECOSTRESS spectra: each spectrum (sensor bands, 0-1 reflectance, NaN at invalid bands) is
linearly interpolated onto the 2,151-channel ASD axis inside its measured range - across internal gaps such as the
water bands too, so the bank's 90 % band-coverage rule keeps it on every sensor - and the USGS sentinel outside it.
The hold-out is spatial, not random: neighbouring array pixels are near-duplicates, so per block (one array area of
one scene) the easternmost share of PV pixels (largest col) is held out and written to holdout.json, never to the
library. Blocks named in --doubtful (identified by eye only, no silicon-bandgap step or no row structure) are not
added at all; they are held out whole and reported apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SENTINEL = -1.23e34
VERSION_SUFFIX = "+pv1"


def to_asd(wl, refl, asd_wl):
    wl, refl = np.asarray(wl, float), np.asarray([np.nan if v is None else v for v in refl], float)
    ok = np.isfinite(refl) & np.isfinite(wl) & (wl > 0)
    order = np.argsort(wl[ok])
    w, r = wl[ok][order], refl[ok][order]
    out = np.interp(asd_wl, w, r)
    out[(asd_wl < w[0]) | (asd_wl > w[-1])] = SENTINEL
    return out


def split(pv):
    """(library, holdout): per block, the easternmost share held out; doubtful blocks held out whole."""
    lib, hold = [], []
    by_block = {}
    doubtful = set(filter(None, ARGS.doubtful.split(",")))
    for s in pv:
        if s.get("block") in doubtful:
            hold.append({**s, "doubtful": True})
            continue
        by_block.setdefault((s["scene_id"], s.get("block")), []).append(s)
    for items in by_block.values():
        items = sorted(items, key=lambda s: (s["col"], s["row"]))
        n_hold = max(1, round(len(items) * ARGS.holdout)) if len(items) > 2 else 0
        lib += items[:len(items) - n_hold]
        hold += items[len(items) - n_hold:]
    return lib, hold


def main():
    data = json.loads(Path(ARGS.pv_json).read_text())["spectra"]
    pv = [s for s in data if s.get("kind", "pv") == "pv"]
    lib_dir, out = Path(ARGS.library_dir), Path(ARGS.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    asd_wl = np.loadtxt(lib_dir / "wavelengths_asd_nm.txt", skiprows=1)      # header line, as library.py reads it
    lib, hold = split(pv)
    index = json.loads((lib_dir / "index.json").read_text())
    index["entries"] = [e for e in index["entries"] if e.get("source") != "scene_pv"]   # idempotent
    folder = lib_dir / "spectra" / "manmade.glass"
    for old in folder.glob("pv_*.txt"):
        old.unlink()
    for s in lib:
        refl = to_asd(s["wavelengths_nm"], s["reflectance"], asd_wl)
        fname = f"pv_{s['id']}.txt"
        title = f"PV array, {s['sensor']} {s['scene_id']} row {s['row']} col {s['col']}"
        (folder / fname).write_text(f" scene_pv {fname}: {title}\n" + "\n".join(f"{v:.6g}" for v in refl) + "\n")
        index["entries"].append({
            "name": f"scene_pv:{s['id']}", "material": title, "chapter": "manmade.glass", "asd_subtype": "ASDFR",
            "path": f"spectra/manmade.glass/{fname}", "source": "scene_pv", "group": "manmade", "family": "glass",
            "provenance": {k: s.get(k) for k in ("sensor", "scene_id", "file", "product", "row", "col", "lat", "lon",
                                                  "note")},
        })
    counts = {}
    for e in index["entries"]:
        counts[e["chapter"]] = counts.get(e["chapter"], 0) + 1
    index["per_chapter_counts"] = dict(sorted(counts.items()))
    index["n_entries"] = len(index["entries"])
    base = index["version"].split("+")[0]
    index["version"] = base + VERSION_SUFFIX           # the per-sensor caches are keyed on it
    (lib_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (out / "holdout.json").write_text(json.dumps({"spectra": hold}, indent=1))
    (out / "library_ids.json").write_text(json.dumps([s["id"] for s in lib], indent=1))
    print(f"library +{len(lib)} PV spectra, held out {len(hold)}; version {index['version']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pv_json")
    ap.add_argument("library_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--doubtful", default="cab,east-mid")
    ARGS = ap.parse_args()
    main()
