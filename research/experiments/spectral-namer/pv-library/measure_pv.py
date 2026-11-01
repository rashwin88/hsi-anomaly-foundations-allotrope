"""
Held-out PV pixels' P(man-made) and class with the served namer, before and after the PV library addition.

    python measure_pv.py <holdout.json> <pv_spectra.json> <crop npz dir> <checkpoint.pt> <library_before> <library_after> <cache_dir> <out.json>

"Before" is the product as served before addition B: the family library without the PV entries AND the class list
without glass_solar (glass was left out of v1, so the one window-glass spectrum was not in the bank). It is
emulated by patching classes.MANMADE_CLASSES / class_of in the loaded modules, not by a second checkout. "After" is
the current code and library. Each pixel is named on its scene crop, so its 3-7 px ring is the real neighbourhood;
the background pixels (kind "background" in pv_spectra.json) are the contrast set and should stay natural.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import app.spectral_match.namer.bank as bank
import app.spectral_match.namer.classes as classes
import app.spectral_match.namer.infer as infer

HOLD, ALL, CROPS, CKPT, LIB_B, LIB_A, CACHE, OUT = sys.argv[1:9]
AFTER_CLASSES = classes.MANMADE_CLASSES
BEFORE_CLASSES = tuple(c for c in AFTER_CLASSES if c != "glass_solar")


def set_classes(before: bool):
    cl = BEFORE_CLASSES if before else AFTER_CLASSES
    orig = classes.class_of

    def class_of(chapter, name):
        c = orig(chapter, name)
        return None if before and c == "glass_solar" else c
    for mod in (bank, infer):
        mod.MANMADE_CLASSES = cl
    bank.class_of = class_of


CROP_FILES = sorted(Path(CROPS).glob("crop_*.npz"))
_META = {}


def crop_for(s):
    """The crop file of the pixel's scene (date in both names) that contains it."""
    for p in CROP_FILES:
        if p not in _META:
            z = np.load(p)
            _META[p] = (int(z["row0"]), int(z["col0"]), z["cube"].shape[1:])
        r0, c0, (h, w) = _META[p]
        date = "".join(ch for ch in p.stem.split("_")[1] if ch.isdigit())
        if date in s["scene_id"].replace("_", "") and 0 <= s["row"] - r0 < h and 0 <= s["col"] - c0 < w:
            return p
    return None


def run(pixels, before: bool):
    set_classes(before)
    model = infer.load_namer(CKPT)
    out = []
    by_crop = {}
    for s in pixels:
        p = crop_for(s)
        if p is None:
            print("no crop for", s["id"])
            continue
        by_crop.setdefault(p, []).append(s)
    for path, items in by_crop.items():
        z = np.load(path)
        cube = z["cube"].astype(np.float32)
        valid = z["valid"].astype(bool) if "valid" in z.files else None
        r0, c0 = int(z["row0"]), int(z["col0"])
        mask = np.zeros(cube.shape[1:], bool)
        for s in items:
            mask[s["row"] - r0, s["col"] - c0] = True
        res = infer.name_pixels(model, cube, z["wavelengths_nm"], mask, LIB_B if before else LIB_A,
                                Path(CACHE) / ("before" if before else "after"), valid=valid)
        rows, cols = np.nonzero(mask)
        at = {(int(r) + r0, int(c) + c0): i for i, (r, c) in enumerate(zip(rows, cols))}
        for s in items:
            i = at[(s["row"], s["col"])]
            out.append({"id": s["id"], "kind": s.get("kind", "pv"), "block": s.get("block"),
                        "doubtful": bool(s.get("doubtful")), "p_manmade": float(res.p_manmade[i]),
                        "cls": res.classes[int(res.cls[i])], "p_cls": float(res.p_cls[i])})
    return out


def summary(rows):
    p = np.array([r["p_manmade"] for r in rows])
    if not len(p):
        return {}
    cls = {}
    for r in rows:
        if r["p_manmade"] > 0.5:
            cls[r["cls"]] = cls.get(r["cls"], 0) + 1
    return {"n": len(p), "mean_p_manmade": round(float(p.mean()), 3), "median": round(float(np.median(p)), 3),
            "share_above_0.5": round(float((p > 0.5).mean()), 3), "share_above_0.35": round(float((p > 0.35).mean()), 3),
            "share_above_0.25": round(float((p > 0.25).mean()), 3), "class_of_manmade_calls": cls}


def main():
    hold = json.loads(Path(HOLD).read_text())["spectra"]
    bg = [s for s in json.loads(Path(ALL).read_text())["spectra"] if s.get("kind") == "background"]
    result = {}
    sets = (("pv_holdout", [s for s in hold if not s.get("doubtful")]),
            ("pv_doubtful", [s for s in hold if s.get("doubtful")]), ("background", bg))
    for name, pix in sets:
        b, a = run(pix, True), run(pix, False)
        result[name] = {"before": summary(b), "after": summary(a), "pixels": {"before": b, "after": a}}
        print(name, "before", result[name]["before"], "\n", name, "after ", result[name]["after"], flush=True)
    Path(OUT).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
