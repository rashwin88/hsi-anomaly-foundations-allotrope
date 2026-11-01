"""
Build the story GIF's frames from a spectral_library_match action and the actions upstream of it.

The api (backend/allotrope/api/action_story.py) resolves the chain from the database into a ``StoryChain``
(directories, configs, summaries - no DB here) and calls ``build_story``; app/story/render.py draws the GIF.
Design: docs/lld/story-gif.md.

Why it exists: the committee demo tells one scene's story - scene, usable area, classes left out, every
pixel scored, the threshold, the objects, the answer - from files the actions already wrote. Nothing is
recomputed except the class-driven selection, which reuses app/spectral_match/class_selection.py so the
GIF's last frame names exactly the objects the class-driven export would contain.

Frame 4 is a heatmap. The submission rules disqualify heatmap outputs, so the GIF is a presentation
artefact only and must never be added to a submission bundle or any export zip.

Gotchas:
  - All rasters are resampled to one display grid (render.fit_shape); masks with "any pixel in the box",
    so a one-pixel anomaly survives a 4:1 reduction. Markers are drawn at that grid's scale with a floor
    on their size, so they stay visible on a 4,000 px flight line.
  - A step whose action or file is missing is skipped, never faked. Without namer results the story ends
    at the threshold frame.
  - class_codes.bin: low nibble = class code, 0x10 = thin-cloud display bit, 255 = outside valid
    (docs/lld/segmentation-v2.md).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.story.chain import (
    INVALID, SelectionParams, StoryChain, class_codes, has_namer, namer_labels, read_tif, scene_grid, scene_rgb,
    valid_mask,
)
from app.story.render import (
    DEFAULT_SIZE, MANMADE_RED, NATURAL_GREEN, OCHRE, OUTLINE, PAPER, Frame, blend, fit_shape, resize,
)

SENSOR_LABELS = {"prisma": "PRISMA", "enmap": "EnMAP", "aviris_ng": "AVIRIS-NG", "tanager": "Tanager",
                 "generic_hsi": "Hyperspectral", "landsat9": "Landsat 9", "hotsat1": "HotSat-1"}
V2_CLASSES = ("land", "water", "cloud", "shadow", "snow")
HAZE_BIT = 0x10
# Excluded classes are painted this grey at this weight, then hatched, so "left out" reads without colour.
EXCLUDED_GREY, EXCLUDED_WEIGHT = (0x9A, 0x9A, 0x94), 0.7
# Muted sequential ramp for the score (paper -> straw -> rust -> oxblood) and its opacity at each stop.
SCORE_STOPS = (0.0, 0.3, 0.65, 1.0)
SCORE_RGB = ((0xF3, 0xE9, 0xD2), (0xE0, 0xB8, 0x6A), (0xB8, 0x5C, 0x2E), (0x6E, 0x1A, 0x1A))
SCORE_ALPHA = (0.0, 0.45, 0.8, 0.95)
# How far the RGB is washed toward paper behind markers (frames 4-6) and behind the answer (frame 7).
DIM, FADE = 0.55, 0.78


def build_story(chain: StoryChain, sel: SelectionParams, size: int = DEFAULT_SIZE) -> tuple[list[Frame], str]:
    """(frames, title). Raises StoryError without an rgb.png in the chain, and
    class_selection.SelectionError for a selection that cannot be made."""
    rgb = scene_rgb(chain)
    shape = fit_shape(*rgb.shape[:2], size=size)
    grid = scene_grid(chain, rgb)
    scale = shape[0] / grid[0]                       # scene (row, col) -> display px
    rgb_d = resize(rgb, shape)
    valid = valid_mask(chain, rgb, shape)
    base = np.where(valid[..., None], rgb_d, blend(rgb_d, PAPER, 0.9))
    dim = blend(_desaturate(base, 0.6), PAPER, DIM)

    frames = [Frame(rgb_d, _scene_caption(chain, grid))]
    if chain.band_filter is not None:
        frames.append(Frame(base, _band_caption(chain), overlay=_hatch(~valid, (0x8C, 0x8C, 0x86), 90)))
    seg = _segmentation_frame(chain, base, shape)
    if seg is not None:
        frames.append(seg)
    score = _score_frame(chain, dim, shape)
    if score is not None:
        frames.append(score)
    mask = read_tif(chain.prep.dir / "anomaly_mask.tif") if chain.prep else None
    if mask is not None:
        frames.append(Frame(dim, _threshold_caption(chain, int((mask > 0).sum())),
                            overlay=_points(resize_mask(mask > 0, shape), scale)))
    if has_namer(chain):
        frames.extend(_object_frames(chain, sel, dim, base, scale))
    title = f"Allotrope · {chain.scene_name}"
    return frames, title


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """True where any source pixel inside the display pixel is True."""
    return resize(mask.astype(np.uint8) * 255, shape) > 0


# --- captions ----------------------------------------------------------------


def sensor_label(sensor: str) -> str:
    return SENSOR_LABELS.get(sensor, sensor.replace("_", " ").upper())


def _scene_caption(chain: StoryChain, hw: tuple[int, ...]) -> str:
    parts = [sensor_label(chain.sensor)]
    if chain.acquired is not None:
        parts.append(chain.acquired.strftime("%Y-%m-%d"))
    parts.append(f"{hw[1]}×{hw[0]} px")
    if chain.scene_bands:
        parts.append(f"{chain.scene_bands} bands")
    return " · ".join(parts)


def _band_caption(chain: StoryChain) -> str:
    s = chain.band_filter.summary
    n = s.get("band_count")
    of = f" of {chain.scene_bands}" if chain.scene_bands and n and chain.scene_bands != n else ""
    text = f"{n}{of} bands usable; no-data removed" if n else "Bands filtered; no-data removed"
    pct = s.get("spatial_validity_pct")
    if pct is not None and pct < 99.95:
        text += f" ({100 - float(pct):.1f}% of the frame)"
    return text


def _threshold_caption(chain: StoryChain, n: int) -> str:
    cfg = chain.prep.config
    mode, thr = cfg.get("committed_threshold_mode"), cfg.get("committed_threshold")
    if mode == "percentile" and thr is not None:
        return f"Top {_num(100 - float(thr))}% → {n:,} anomalous pixels"
    if thr is not None:
        return f"Score ≥ {_num(float(thr))} → {n:,} anomalous pixels"
    return f"{n:,} anomalous pixels"


def _num(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:,.0f}"


def _model_names(chain: StoryChain) -> list[str]:
    for src in (chain.prep.summary if chain.prep else {}, chain.scoring.config if chain.scoring else {}):
        names = src.get("active_codenames") or src.get("model_codenames")
        if names:
            return list(names)
    return []


# --- frames 3 and 4 ------------------------------------------------------------


def _segmentation_frame(chain: StoryChain, base: np.ndarray, shape) -> Frame | None:
    seg = chain.segmentation
    if seg is None:
        return None
    scoring_cfg = chain.scoring.config if chain.scoring else {}
    excluded = scoring_cfg.get("exclude_classes")
    if excluded is None:
        excluded = seg.config.get("exclude_classes", seg.summary.get("exclude_classes"))
    codes = class_codes(seg)
    if codes is not None and excluded is not None:
        names = list(seg.summary.get("class_names") or V2_CLASSES)
        inside = codes != INVALID
        out = np.zeros(codes.shape, bool)
        for c in excluded:
            if c == "thin_cloud":
                out |= inside & ((codes & HAZE_BIT) > 0)
            elif c in names:
                out |= inside & ((codes & 0x0F) == names.index(c))
        pct = {e.get("key"): e.get("pct_of_valid") for e in seg.summary.get("legend") or []}
        n_valid = max(int(inside.sum()), 1)
        labels = [(c, pct[c] if pct.get(c) is not None else 100 * (inside & _cls(codes, c, names)).sum() / n_valid)
                  for c in excluded]
    else:
        keep = read_tif(seg.dir / "keep_mask.tif")
        if keep is None:
            return None
        valid = read_tif(seg.dir / "class_map.tif")
        inside = valid != INVALID if valid is not None and valid.shape == keep.shape else np.ones(keep.shape, bool)
        out = inside & (keep == 0)
        classes = seg.config.get("classes_to_mask") or excluded or ["masked classes"]
        labels = [(" / ".join(classes), 100 * out.sum() / max(int(inside.sum()), 1))]
    if not labels:
        return Frame(base, "Scene segmented; no class left out")
    out_d = resize(out.astype(np.uint8) * 255, shape) >= 128
    shown = blend(base, EXCLUDED_GREY, EXCLUDED_WEIGHT)
    img = np.where(out_d[..., None], shown, base)
    if len(labels) == 1:
        caption = f"{_class_label(labels[0][0])} left out ({_pct(labels[0][1])})"
    else:
        words = [f"{_class_label(c).lower()} ({_pct(p)})" for c, p in labels]
        caption = f"{', '.join(words[:-1])} and {words[-1]} left out".capitalize()
    return Frame(img, caption, overlay=_hatch(out_d, (0x6E, 0x6E, 0x6A), 150))


def _pct(p) -> str:
    p = float(p)
    return "none in this scene" if p == 0 else f"{p:.2f}%" if p < 0.1 else f"{p:.1f}%"


def _cls(codes: np.ndarray, c: str, names: list[str]) -> np.ndarray:
    if c == "thin_cloud":
        return (codes & HAZE_BIT) > 0
    return (codes & 0x0F) == names.index(c) if c in names else np.zeros(codes.shape, bool)


def _class_label(c: str) -> str:
    return {"thin_cloud": "Thin cloud"}.get(c, c.replace("_", " ").capitalize())


def _score_frame(chain: StoryChain, dim: np.ndarray, shape) -> Frame | None:
    score = read_tif(chain.prep.dir / "composite_score.tif") if chain.prep else None
    names = _model_names(chain)
    if score is None and chain.scoring is not None and names:
        score = read_tif(chain.scoring.dir / "models" / names[0].lower().replace(" ", "_") / "anomaly_score.tif")
    if score is None:
        return None
    s = np.nan_to_num(score.astype(np.float32), nan=0.0)
    pos = s[s > 0]
    if pos.size == 0:
        return None
    # Scores are heavy-tailed: stretch p25..p99.5, or the map is one colour with a few dots.
    lo, hi = np.percentile(pos, [25, 99.5])
    t = resize(np.clip((s - lo) / max(hi - lo, 1e-12), 0, 1), shape)
    rgba = np.zeros(shape + (4,), np.uint8)
    for c in range(3):
        rgba[..., c] = np.interp(t, SCORE_STOPS, [v[c] for v in SCORE_RGB])
    rgba[..., 3] = 255 * np.interp(t, SCORE_STOPS, SCORE_ALPHA)
    who = " + ".join(names) if names else "The model"
    verb = "score" if len(names) > 1 else "scores"
    return Frame(dim, f"{who} {verb} every pixel", overlay=rgba)


# --- frames 5-7 ------------------------------------------------------------------


def _points(mask_d: np.ndarray, scale: float) -> np.ndarray:
    """Committed pixels as ochre points with a dark outline, at least 3 display px across."""
    core = Image.fromarray(mask_d.astype(np.uint8) * 255)
    if scale < 2:
        core = core.filter(ImageFilter.MaxFilter(3))
    ring = np.asarray(core.filter(ImageFilter.MaxFilter(3))) > 0
    core_a = np.asarray(core) > 0
    rgba = np.zeros(mask_d.shape + (4,), np.uint8)
    rgba[ring] = OUTLINE + (255,)
    rgba[core_a] = OCHRE + (255,)
    return rgba


def _object_frames(chain: StoryChain, sel: SelectionParams, dim: np.ndarray, base: np.ndarray,
                   scale: float) -> list[Frame]:
    from app.spectral_match.class_selection import select

    confident_p = float(chain.match.config.get("namer_confident_p", 0.55))
    s = select(chain.match.dir, sel.tokens, confident_only=sel.confident_only, manmade_p=sel.manmade_p,
               confident_p=confident_p)
    o = s.objects
    mm = o["call"] == "manmade"
    shape = dim.shape[:2]
    everything = np.ones(len(mm), bool)
    objects = Frame(dim, f"{len(mm):,} objects · {int(mm.sum()):,} man-made",
                    overlay=_markers(o, mm, everything, shape, scale, alpha=235))
    faded = blend(_desaturate(base, 0.8), PAPER, FADE)
    ov = _markers(o, mm, ~s.selected, shape, scale, alpha=55)
    # A ring per chosen object; thinner when there are many, or the rings merge into one dark smear.
    top = _markers(o, mm, s.selected, shape, scale, alpha=255, ring=2 if s.n_objects <= 150 else 1, grow=1.5)
    a = top[..., 3:4] > 0
    ov = np.where(a, top, ov)
    k = s.n_objects
    answer = Frame(faded, f"{_selection_label(chain, sel, s)} → {k:,} object{'s' if k != 1 else ''}", overlay=ov)
    return [objects, answer]


def _selection_label(chain: StoryChain, sel: SelectionParams, s) -> str:
    labels = namer_labels(chain.match.dir)
    words = [{"manmade": "Man-made", "natural": "Natural"}.get(t, labels.get(t, t.replace("_", " "))) for t in s.tokens]
    words = [words[0][:1].upper() + words[0][1:]] + words[1:]
    text = " + ".join(words)
    if s.confident_only:
        text += ", confident only"
    if abs(s.manmade_p - s.default_p) > 1e-9:
        text += f" (man-made above P {_num(s.manmade_p)})"
    return text


def _markers(o: dict, mm: np.ndarray, which: np.ndarray, shape, scale: float, *, alpha: int,
             ring: int = 0, grow: float = 1.0) -> np.ndarray:
    """Man-made objects as red squares, natural as green circles, at their centroids. A marker's side
    grows with the object's size and never drops below 6 display px."""
    img = Image.new("RGBA", shape[::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rows, cols = np.asarray(o["row"], float) * scale, np.asarray(o["col"], float) * scale
    side = np.clip(np.sqrt(np.asarray(o["n_px"], float)) * scale * 1.2, 6, 14) * grow
    order = np.argsort(mm, kind="stable")                          # natural first, man-made on top
    for i in order:
        if not which[i]:
            continue
        r, c, h = rows[i], cols[i], side[i] / 2
        box = (c - h, r - h, c + h, r + h)
        colour = (MANMADE_RED if mm[i] else NATURAL_GREEN) + (alpha,)
        edge = OUTLINE + (alpha,)
        if mm[i]:
            d.rectangle(box, fill=colour, outline=edge)
        else:
            d.ellipse(box, fill=colour, outline=edge)
        if ring:
            g = h + 4
            d.ellipse((c - g, r - g, c + g, r + g), outline=OUTLINE + (230,), width=ring)
    return np.asarray(img)


def _desaturate(img: np.ndarray, amount: float) -> np.ndarray:
    grey = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return blend(img, np.repeat(grey[..., None], 3, axis=2), amount)


def _hatch(mask: np.ndarray, colour: tuple[int, int, int], alpha: int, pitch: int = 7) -> np.ndarray:
    """Diagonal hatching inside ``mask``: (H, W, 4) RGBA."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    lines = ((xx + yy) % pitch) == 0
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask & lines] = colour + (alpha,)
    return rgba
