"""
Summaries and the markdown report of an evaluation run (docs/lld/evaluation.md).

Per dataset and (model, scorer): objects caught at each budget as a share of objects, next to the analytic chance
level; pixel measures as the mean over scenes; and the same broken down by every group_by attribute. Datasets are
never pooled - their objects are different populations.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.mean(xs)) if xs else None


def summarise(object_rows: list, scene_rows: list, cfg) -> dict:
    tops = [q for m in cfg.metrics if m.kind == "objects_caught" for q in m.top_pct]
    pixel_keys = sorted({k for r in scene_rows for k in r if k not in ("dataset", "scene", "sensor", "model", "scorer",
                                                                     "objects") and "|" not in k})
    out: dict = {}
    for ds in dict.fromkeys(r["dataset"] for r in scene_rows):
        for (model, scorer) in dict.fromkeys((r["model"], r["scorer"]) for r in scene_rows if r["dataset"] == ds):
            objs = [r for r in object_rows if r["dataset"] == ds and r["model"] == model and r["scorer"] == scorer]
            scenes = [r for r in scene_rows if r["dataset"] == ds and r["model"] == model and r["scorer"] == scorer]
            entry = {"objects": len(objs), "scenes": len(scenes),
                     "caught": {f"{q:g}": _mean([float(o[f"hit_{q:g}"]) for o in objs]) for q in tops},
                     "chance": {f"{q:g}": _mean([o[f"chance_{q:g}"] for o in objs]) for q in tops},
                     "pixel": {k: _mean([s.get(k) for s in scenes]) for k in pixel_keys}, "groups": {}}
            for key in cfg.group_by:
                groups = defaultdict(list)
                for o in objs:
                    if key in o:
                        groups[str(o[key])].append(o)
                for value, members in sorted(groups.items()):
                    entry["groups"][f"{key}={value}"] = {
                        "objects": len(members),
                        "caught": {f"{q:g}": _mean([float(o[f"hit_{q:g}"]) for o in members]) for q in tops},
                        "chance": {f"{q:g}": _mean([o[f"chance_{q:g}"] for o in members]) for q in tops},
                        "pixel": {k: _mean([s.get(f"{k}|{key}={value}") for s in scenes]) for k in pixel_keys
                                  if any(f"{k}|{key}={value}" in s for s in scenes)}}
            out.setdefault(ds, {})[f"{model} | {scorer}"] = entry
    return out


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x):
    return "-" if x is None else f"{x:.3f}"


def write_report(summary: dict, cfg, path: Path) -> None:
    tops = [q for m in cfg.metrics if m.kind == "objects_caught" for q in m.top_pct]
    lines = [f"# Evaluation: {cfg.name}", "",
             "Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the "
             "analytic level for random scores. Pixel measures are means over scenes.", ""]
    for ds, entries in summary.items():
        lines += [f"## {ds}", ""]
        pixel_keys = sorted({k for e in entries.values() for k in e["pixel"]})
        head = ["model", "scorer"] + [f"caught top {q:g}%" for q in tops] + pixel_keys   # labels are "model | scorer"
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        first = next(iter(entries.values()))
        lines.append("| *chance* | | " + " | ".join(_pct(first["chance"][f"{q:g}"]) for q in tops) + " |"
                     + " - |" * len(pixel_keys))
        for label, e in entries.items():
            lines.append(f"| {label} | " + " | ".join(_pct(e["caught"][f"{q:g}"]) for q in tops) + " | "
                         + " | ".join(_num(e["pixel"].get(k)) for k in pixel_keys) + " |")
        group_names = list(dict.fromkeys(g for e in entries.values() for g in e["groups"]))
        for key in cfg.group_by:
            names = [g for g in group_names if g.startswith(f"{key}=")]
            if not names:
                continue
            q = tops[0] if tops else None
            lines += ["", f"### by {key} (caught top {q:g}%; pAUC where measured)", "",
                      "| model | scorer | " + " | ".join(n.split("=", 1)[1] for n in names) + " |",
                      "|" + "---|" * (len(names) + 2)]
            pk = next((k for k in pixel_keys if k.startswith("pauc")), None)
            for label, e in entries.items():
                cells = []
                for n in names:
                    g = e["groups"].get(n)
                    if g is None:
                        cells.append("-")
                        continue
                    c = _pct(g["caught"].get(f"{q:g}"))
                    cells.append(c + (f" ({_num(g['pixel'].get(pk))})" if pk and pk in g["pixel"] else ""))
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))
