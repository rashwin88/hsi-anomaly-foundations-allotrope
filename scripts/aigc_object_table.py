"""
Per-object material table for the AIGC report: SAM vs the family match vs the spectral namer, side by side.

    python scripts/aigc_object_table.py --match-dir <spectral_library_match output> --mask <prep anomaly_mask.tif>
        --vendable <scene vendable.pkl> --scene-dir <scene raw dir> --sensor prisma --out <dir>
        [--namer-ckpt runs/N001/ep12.pt --library-root data/family_library]

Objects are 8-connected components of the committed 0/1 mask. Per object: size, centroid (lat / lon through the
scene's own georef, app/georef), SAM (matches.parquet: majority top-1 name, its share, median top-1 angle, the three
commonest names), the family match (family_matches.parquet: majority family over answered pixels, mean confidence,
answered share) and, with --namer-ckpt, the namer (mean P(man-made), majority class among its pixels, mean
P(class | man-made), mean fill). Writes objects.csv, objects.geojson and findings.md - agreement counts and the
objects where the methods disagree, the starting point for the report chapter "Findings on Evaluation Dataset".
Runs in the worker container or the C:/av venv; reads files only.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# SAM's splib07 chapters and the family groups -> the namer's level 1, for the agreement count.
NATURAL_SAM_CHAPTERS = ("minerals", "soils", "vegetation")      # splib07 chapters; organics / artificial are not natural-only


def _majority(values):
    if not len(values):
        return "", 0.0
    name, n = Counter(values).most_common(1)[0]
    return name, n / len(values)


def build(args) -> list[dict]:
    import pandas as pd
    import rasterio
    from scipy.ndimage import label

    with rasterio.open(args.mask) as src:
        mask = src.read(1).astype(bool)
    labels, n_obj = label(mask, structure=np.ones((3, 3), int))
    print(f"{int(mask.sum())} anomaly pixels in {n_obj} objects")
    match_dir = Path(args.match_dir)
    sam = pd.read_parquet(match_dir / "matches.parquet") if (match_dir / "matches.parquet").exists() else None
    fam = (pd.read_parquet(match_dir / "family_matches.parquet")
           if (match_dir / "family_matches.parquet").exists() else None)

    transform, crs = None, None
    if args.scene_dir:
        from app.georef.from_scene_file import resolve_scene_georef
        transform, crs = resolve_scene_georef(scene_dir=Path(args.scene_dir), sensor_type=args.sensor,
                                              target_shape=mask.shape)

    namer = None
    if args.namer_ckpt:
        from app.spectral_match.namer.infer import load_namer, name_pixels
        with open(args.vendable, "rb") as f:
            v = pickle.load(f)
        model = load_namer(args.namer_ckpt)
        res = name_pixels(model, v.normalized_hyperspectral_cube, v.band_cw_order, mask, args.library_root,
                          Path(args.out) / "namer_lib_cache", valid=v.validity_cube.astype(bool))
        rr, cc = np.nonzero(mask)
        namer = pd.DataFrame({"row": rr, "col": cc, "p_mm": res.p_manmade, "cls": [res.classes[i] for i in res.cls],
                              "p_cls": res.p_cls, "fill": res.fill})

    lab = lambda df: labels[df["row"].to_numpy(), df["col"].to_numpy()]
    if sam is not None:
        sam = sam[sam["rank"] == 0].assign(obj=lambda d: lab(d))
    if fam is not None:
        fam = fam.assign(obj=lambda d: lab(d))
    if namer is not None:
        namer = namer.assign(obj=lambda d: lab(d))

    objs = []
    for k in range(1, n_obj + 1):
        rr, cc = np.nonzero(labels == k)
        o = {"obj_id": k, "n_px": len(rr), "row": float(rr.mean()), "col": float(cc.mean())}
        if transform is not None:
            from rasterio.warp import transform as warp
            x, y = rasterio.transform.xy(transform, o["row"], o["col"])
            lon, lat = warp(crs, "EPSG:4326", [x], [y])
            o["lon"], o["lat"] = round(lon[0], 6), round(lat[0], 6)
        if sam is not None:
            s = sam[sam["obj"] == k]
            o["sam_top1"], o["sam_share"] = _majority(s["name"].tolist())
            o["sam_chapter"] = _majority(s["chapter"].tolist())[0]
            o["sam_angle_med"] = round(float(s["angle_deg"].median()), 2) if len(s) else None
            o["sam_top3"] = "; ".join(n for n, _ in Counter(s["name"]).most_common(3))
        if fam is not None:
            f = fam[(fam["obj"] == k) & fam["answered"]]
            o["family"], o["family_share"] = _majority(f["family"].tolist())
            o["family_group"] = _majority(f["family_group"].tolist())[0]
            o["family_conf"] = round(float(f["confidence"].mean()), 3) if len(f) else None
            o["family_answered"] = round(len(f) / max(int((fam["obj"] == k).sum()), 1), 3)
        if namer is not None:
            m = namer[namer["obj"] == k]
            o["namer_p_manmade"] = round(float(m["p_mm"].mean()), 3)
            o["namer_class"], o["namer_class_share"] = _majority(m["cls"].tolist())
            o["namer_p_class"] = round(float(m["p_cls"].mean()), 3)
            o["namer_fill"] = round(float(m["fill"].mean()), 3)
        objs.append(o)
    return objs


def findings(objs: list[dict], args) -> str:
    n = len(objs)
    lines = [f"# Findings stub - {Path(args.mask).parent.name}", "",
             f"{n} objects, {sum(o['n_px'] for o in objs)} pixels. SAM = splib07 single-spectrum angle; family = "
             "calibrated Simplex ACE (classical); namer = the spectral foundation model"
             + (f" ({Path(args.namer_ckpt).name}, preliminary)" if args.namer_ckpt else " (not run)") + ".", ""]
    if n and "family_group" in objs[0]:
        answered = [o for o in objs if o.get("family")]
        lines.append(f"- Family match answered {len(answered)} / {n} objects.")
    if n and "namer_p_manmade" in objs[0]:
        mm = [o for o in objs if o["namer_p_manmade"] > 0.5]
        lines.append(f"- Namer calls {len(mm)} / {n} objects man-made (mean P > 0.5).")
        if "family_group" in objs[0]:
            both = [o for o in objs if o.get("family_group")]
            agree = [o for o in both if (o["family_group"] == "manmade") == (o["namer_p_manmade"] > 0.5)]
            lines.append(f"- Level 1 (natural vs man-made): namer and family agree on {len(agree)} / {len(both)} "
                         "objects the family match answered.")
    if n and "sam_chapter" in objs[0] and "namer_p_manmade" in objs[0]:
        sam_nat = [o for o in objs if o["sam_chapter"].startswith(NATURAL_SAM_CHAPTERS)]
        lines.append(f"- SAM's top-1 is a natural-chapter spectrum for {len(sam_nat)} / {n} objects.")
    lines += ["", "## Largest objects", "", "| obj | px | SAM top-1 (angle) | family (conf) | namer class (P, P mm) |",
              "|---|---|---|---|---|"]
    for o in sorted(objs, key=lambda o: -o["n_px"])[:15]:
        lines.append(f"| {o['obj_id']} | {o['n_px']} | {o.get('sam_top1', '')} ({o.get('sam_angle_med', '')}) | "
                     f"{o.get('family', '')} ({o.get('family_conf', '')}) | {o.get('namer_class', '')} "
                     f"({o.get('namer_p_class', '')}, {o.get('namer_p_manmade', '')}) |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--match-dir", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-dir")
    ap.add_argument("--sensor", default="prisma")
    ap.add_argument("--vendable")
    ap.add_argument("--namer-ckpt")
    ap.add_argument("--library-root", default=str(Path(__file__).resolve().parents[1] / "data" / "family_library"))
    args = ap.parse_args()
    if args.namer_ckpt and not args.vendable:
        ap.error("--namer-ckpt needs --vendable")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    objs = build(args)
    import pandas as pd
    pd.DataFrame(objs).to_csv(out / "objects.csv", index=False)
    if objs and "lon" in objs[0]:
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
                  "properties": {k: v for k, v in o.items() if k not in ("lon", "lat")}} for o in objs]
        (out / "objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, default=str))
    (out / "findings.md").write_text(findings(objs, args), encoding="utf-8")
    print(f"wrote {out / 'objects.csv'} ({len(objs)} objects), findings.md")


if __name__ == "__main__":
    main()
