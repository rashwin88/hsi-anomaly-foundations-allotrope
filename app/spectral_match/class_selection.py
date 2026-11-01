"""
Which spectral-namer objects a class-driven export picks: the tokens, the man-made cut, the counts.

Used by the class-driven output pack (app/spectral_match/class_pack.py) and by the api's
``GET /actions/{id}/spectral_library_match/selection_counts`` (backend/allotrope/api/action_class_pack.py),
so the count the match screen shows on its export button is the count the pack contains.
Design: docs/proposals/class-driven-flow.md (gap 2, addition A); docs/lld/spectral-namer-in-product.md.

Why it exists: the namer calls each object man-made or natural at P(man-made) > 0.5 and names a man-made one's
closest class. The user picks classes on the match screen ("oil + plastic", "natural") and may lower the man-made
cut (Sensitive 0.35, Very sensitive 0.25) because solar-panel-like objects sit just under 0.5. The export must
re-decide each object at that cut from namer_objects.parquet, with no re-run.

Re-deciding at a cut:
  - at the action's own cut (MANMADE_CALL_P) the stored call, class and confident flag are used as they are;
  - an object still man-made keeps its stored class;
  - a newly man-made object takes its top mean P(class | man-made): the ``prob_<class>`` columns (actions from
    2026-11-01 on), else a lower bound from namer_pixels.parquet (the sum of p_cls over the object's pixels whose
    own top class it is, over n_px - exact for one-pixel objects, low otherwise, so it leans tentative), else
    class "unknown", tentative. It is confident only when that P(class) reaches namer_confident_p.

Selection is the union of the tokens: "natural" = objects called natural, "manmade" = all man-made objects, a
material key = man-made objects of that class. ``confident_only`` then drops tentative man-made objects; natural
objects carry no class confidence and are kept. Material keys are read from namer/classes.py at run time (new
classes such as glass_solar join without a code change here).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The class of a newly man-made object no class probability can be found for (pre-2026-11-01 action, no pixels).
UNKNOWN_CLASS = "unknown"
# Two cuts closer than this are the same cut (floats from a query string).
_CUT_EPS = 1e-9
# Longest selection slug before it is shortened to a count: it goes into every file name of the pack.
_SLUG_MAX = 48


class SelectionError(ValueError):
    """The selection asked for cannot be made (unknown token, no namer data). The api answers 422 with it."""


def material_classes() -> tuple[str, ...]:
    """The namer's man-made class keys, read at call time (namer/classes.py may gain classes)."""
    from app.spectral_match.namer.classes import MANMADE_CLASSES

    return tuple(MANMADE_CLASSES)


def parse_classes(text: str) -> tuple[str, ...]:
    """Comma list -> tokens in canonical order (natural, manmade, then the material classes' own order).

    Raises SelectionError for an empty list or an unknown token, naming the allowed ones."""
    allowed = ("natural", "manmade") + material_classes()
    tokens = {t.strip().lower() for t in (text or "").split(",") if t.strip()}
    if not tokens:
        raise SelectionError(f"classes is empty; give one or more of {', '.join(allowed)}")
    unknown = sorted(tokens - set(allowed))
    if unknown:
        raise SelectionError(f"unknown class {', '.join(unknown)}; choose from {', '.join(allowed)}")
    return tuple(t for t in allowed if t in tokens)


@dataclass
class Selection:
    """Every object re-decided at ``manmade_p``, and which of them the tokens pick."""

    objects: dict           # column -> (n,) array: obj_id, n_px, row, col, p_manmade, call, cls, p_cls, confident
    selected: np.ndarray    # (n,) bool
    tokens: tuple[str, ...]
    confident_only: bool
    manmade_p: float
    default_p: float
    class_source: str       # where newly man-made objects' classes came from: stored | prob_columns | pixels | none

    @property
    def n_objects(self) -> int:
        return int(self.selected.sum())

    @property
    def n_pixels(self) -> int:
        return int(self.objects["n_px"][self.selected].sum())

    def per_class(self) -> dict[str, int]:
        """Selected objects per class ("natural" for natural ones), largest first."""
        o = self.objects
        keys = np.where(o["call"] == "manmade", o["cls"], "natural")[self.selected]
        names, counts = np.unique(keys.astype(str), return_counts=True) if len(keys) else ([], [])
        return dict(sorted(((str(k), int(n)) for k, n in zip(names, counts)), key=lambda kv: -kv[1]))

    def counts(self) -> dict:
        """The selection_counts payload: {objects, pixels, per_class}."""
        return {"objects": self.n_objects, "pixels": self.n_pixels, "per_class": self.per_class()}

    def slug(self) -> str:
        """Short, file-name-safe name of the selection, e.g. "oil_fuel-plastic_conf_p035"."""
        base = "-".join(self.tokens)
        if len(base) > _SLUG_MAX:
            base = f"{len(self.tokens)}classes"
        if self.confident_only:
            base += "_conf"
        if abs(self.manmade_p - self.default_p) > _CUT_EPS:
            base += "_p" + f"{self.manmade_p:.3f}".rstrip("0").rstrip(".").replace(".", "")
        return re.sub(r"[^a-z0-9_-]", "", base.lower())


def read_objects(artifact_dir: Path) -> dict:
    """namer_objects.parquet as column -> numpy array. SelectionError when the action has no namer results."""
    import pyarrow.parquet as pq

    path = artifact_dir / "namer_objects.parquet"
    if not path.is_file():
        raise SelectionError("this action has no spectral-namer results (it ran without the namer, or the "
                             "namer failed), so it cannot be exported by class")
    t = pq.read_table(path).to_pydict()
    return {k: np.asarray(v, dtype=object if k in ("call", "cls") else None) for k, v in t.items()}


def select(artifact_dir: Path, tokens: tuple[str, ...], *, confident_only: bool, manmade_p: float | None,
           confident_p: float, default_p: float | None = None) -> Selection:
    """Re-decide every object at ``manmade_p`` (None = the action's cut) and pick the ``tokens``."""
    from app.spectral_match.namer.objects import MANMADE_CALL_P

    default_p = MANMADE_CALL_P if default_p is None else default_p
    cut = default_p if manmade_p is None else float(manmade_p)
    if not 0.0 <= cut <= 1.0:
        raise SelectionError(f"manmade_p must be between 0 and 1, got {cut}")
    objs, source = recall(read_objects(artifact_dir), artifact_dir, cut, confident_p, default_p)
    mm = objs["call"] == "manmade"
    picked = np.zeros(len(mm), bool)
    if "natural" in tokens:
        picked |= ~mm
    if "manmade" in tokens:
        picked |= mm
    materials = [t for t in tokens if t not in ("natural", "manmade")]
    if materials:
        picked |= mm & np.isin(objs["cls"].astype(str), materials)
    if confident_only:
        picked &= ~mm | objs["confident"]
    return Selection(objs, picked, tokens, confident_only, cut, default_p, source)


def recall(stored: dict, artifact_dir: Path, cut: float, confident_p: float, default_p: float) -> tuple[dict, str]:
    """(objects re-decided at ``cut``, class source). See the module docstring for the rules."""
    out = {k: stored[k] for k in ("obj_id", "n_px", "row", "col", "p_manmade")}
    call = stored["call"].astype(str)
    cls = np.array(["" if c is None else str(c) for c in stored["cls"]], dtype=object)
    p_cls = np.asarray(stored["p_cls"], np.float64)
    confident = np.asarray(stored["confident"], bool)
    source = "stored"
    if abs(cut - default_p) > _CUT_EPS:
        mm = np.asarray(stored["p_manmade"], np.float64) > cut
        new = mm & (call != "manmade")
        if new.any():
            top_cls, top_p, source = _top_classes(stored, artifact_dir)
            cls[new], p_cls[new] = top_cls[new], top_p[new]
            confident[new] = (top_p[new] >= confident_p) & (top_cls[new] != UNKNOWN_CLASS)
        call = np.where(mm, "manmade", "natural")
        cls[~mm], p_cls[~mm], confident[~mm] = "", np.nan, False
    out.update(call=call.astype(object), cls=cls, p_cls=p_cls, confident=confident & (call == "manmade"))
    return out, source


def _top_classes(stored: dict, artifact_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """(n,) top class and its mean P(class | man-made) for every object, and where they came from."""
    n = len(stored["obj_id"])
    prob_cols = [k for k in stored if k.startswith("prob_")]
    if prob_cols:
        probs = np.stack([np.asarray(stored[k], np.float64) for k in prob_cols], axis=1)
        best = probs.argmax(axis=1)
        names = np.array([k[len("prob_"):] for k in prob_cols], dtype=object)
        return names[best], probs[np.arange(n), best], "prob_columns"
    pixels = artifact_dir / "namer_pixels.parquet"
    if pixels.is_file():
        import pyarrow.parquet as pq

        px = pq.read_table(pixels, columns=["obj_id", "cls", "p_cls"]).to_pydict()
        ids = {int(o): i for i, o in enumerate(stored["obj_id"])}
        names = sorted({str(c) for c in px["cls"]})
        if names:
            col = {c: j for j, c in enumerate(names)}
            sums = np.zeros((n, len(names)))
            np.add.at(sums, ([ids[int(o)] for o in px["obj_id"]], [col[str(c)] for c in px["cls"]]),
                      np.asarray(px["p_cls"], np.float64))
            lower = sums / np.maximum(np.asarray(stored["n_px"], np.float64), 1)[:, None]
            best = lower.argmax(axis=1)
            return np.array(names, dtype=object)[best], lower[np.arange(n), best], "pixels"
    return np.full(n, UNKNOWN_CLASS, dtype=object), np.full(n, np.nan), "none"
