"""
Render a list of captioned map frames as one looping, cross-faded GIF ("story GIF").

Called by app/story/frames.py (which turns a spectral_library_match chain into frames) and through it by
the api's GET /actions/{id}/story.gif (backend/allotrope/api/action_story.py). numpy + PIL only.

Why it exists: the committee demo needs one file that shows how a scene was narrowed down to the
selected anomalies, step by step, playable anywhere (slides, chat, a browser) with no viewer. A GIF is
the one format every one of those plays inline.

Gotchas:
  - Every frame is one map extent. The renderer resizes each frame's image (and overlay) to one display
    shape, ``fit_shape(h, w, size)``; builders that draw markers at a minimum on-screen size must draw at
    that shape, so they ask ``fit_shape`` too.
  - One global palette for all frames, no dithering. Dithering turns flat paper and dimmed areas into
    noise that LZW cannot compress; a shared palette lets Pillow write only each frame's changed box.
    The brand colours and a paper-to-ink ramp are reserved in the palette so text and markers stay exact.
  - Size is bounded: if the GIF is over ``max_bytes`` it is re-encoded with fewer cross-fade frames,
    then fewer colours.
"""

from __future__ import annotations

import importlib.util
import io
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Tufte palette: cream paper, near-black ink (tufte-css).
PAPER = (0xFF, 0xFF, 0xF8)
INK = (0x11, 0x11, 0x11)
# Markers (agreed for the demo): committed anomaly pixels, man-made and natural objects, outlines.
OCHRE = (0xD4, 0xA0, 0x17)
MANMADE_RED = (0xC0, 0x39, 0x2B)
NATURAL_GREEN = (0x2E, 0x7D, 0x6B)
OUTLINE = (0x1A, 0x1A, 0x1A)
MUTED_INK = (0x6B, 0x6B, 0x66)

DEFAULT_SIZE = 960
# Below this the caption would wrap into a column: a tall, narrow scene (AVIRIS-NG flight lines are
# 691 x 4118) gets paper either side of the map instead.
MIN_TEXT_WIDTH = 560
MARGIN = 20
TITLE_PT, CAPTION_PT = 15, 26
DEFAULT_HOLD_S = 2.0
LAST_HOLD_S = 4.5
FADE_FRAME_S = 0.08
DEFAULT_FADES = 3
# 8 MB is the agreed ceiling at 960 px: chat and slide tools refuse or re-encode larger GIFs.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024

# Serif faces, best first. ET Book is Tufte's; STIX (bundled with matplotlib in the api image) is the
# nearest Times-like face we ship; DejaVu Serif is in every Debian image.
_FONT_FILES = ("et-book-roman-line-figures.ttf", "ETBembo-RomanLF.ttf", "etbookot-roman-webfont.ttf",
               "STIXGeneral.ttf", "DejaVuSerif.ttf", "Georgia.ttf", "georgia.ttf", "times.ttf")
_FONT_DIRS = (Path(__file__).parent / "fonts", Path("/usr/share/fonts/truetype/dejavu"),
              Path("/usr/share/fonts/truetype"), Path("/usr/share/fonts"), Path("C:/Windows/Fonts"))


@dataclass
class Frame:
    """One step of the story.

    image: (H, W, 3) uint8 RGB, the map; every frame of a story must have the same aspect ratio.
    overlay: optional (H, W, 4) uint8 RGBA composited over the image (straight alpha).
    hold_s: seconds on screen; None = the default (the last frame always gets the long hold)."""

    image: np.ndarray
    caption: str
    overlay: np.ndarray | None = None
    hold_s: float | None = None


def fit_shape(h: int, w: int, size: int = DEFAULT_SIZE) -> tuple[int, int]:
    """(height, width) of the map on screen: the longer side is ``size``."""
    s = size / max(h, w)
    return max(1, round(h * s)), max(1, round(w * s))


def render_gif(frames: list[Frame], *, title: str, size: int = DEFAULT_SIZE, fades: int = DEFAULT_FADES,
               max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
    """GIF bytes: each frame held, cross-faded into the next with ``fades`` blends, looping forever.

    Raises ValueError for no frames or frames of different aspect ratio."""
    if not frames:
        raise ValueError("a story needs at least one frame")
    keys = _compose_all(frames, title, size)
    colours = 240
    while True:
        data = _encode(keys, frames, fades, colours)
        if len(data) <= max_bytes or (fades == 0 and colours <= 64):
            return data
        if fades > 0:
            fades -= 1
        else:
            colours //= 2


def blend(a: np.ndarray, b, t: float) -> np.ndarray:
    """(1 - t) a + t b as uint8; ``b`` an image or one RGB colour."""
    return np.clip(a.astype(np.float32) * (1 - t) + np.asarray(b, np.float32) * t, 0, 255).astype(np.uint8)


def resize(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize an (H, W[, C]) uint8 or float array to ``shape`` with area averaging (BOX)."""
    if arr.shape[:2] == tuple(shape):
        return arr
    if arr.dtype != np.uint8:
        return np.asarray(Image.fromarray(arr.astype(np.float32), "F").resize(shape[::-1], Image.BOX))
    return np.asarray(Image.fromarray(arr).resize(shape[::-1], Image.BOX))


def load_font(pt: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The first serif face found (see _FONT_FILES), else Pillow's built-in face."""
    for path in _font_candidates():
        try:
            return ImageFont.truetype(str(path), pt)
        except OSError:
            continue
    return ImageFont.load_default(pt)


def _font_candidates() -> list[Path]:
    dirs = list(_FONT_DIRS)
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.submodule_search_locations:
        dirs.insert(1, Path(list(spec.submodule_search_locations)[0]) / "mpl-data" / "fonts" / "ttf")
    return [d / f for f in _FONT_FILES for d in dirs if (d / f).is_file()]


def _compose_all(frames: list[Frame], title: str, size: int) -> list[np.ndarray]:
    """Every frame laid out on paper: title and step counter, map, caption."""
    h0, w0 = frames[0].image.shape[:2]
    shape = fit_shape(h0, w0, size)
    for f in frames:
        h, w = f.image.shape[:2]
        if abs(h / w - h0 / w0) > 0.02:
            raise ValueError(f"frame '{f.caption}' is {w}x{h}; the story's extent is {w0}x{h0}")
    title_font, cap_font = load_font(TITLE_PT), load_font(CAPTION_PT)
    width = max(shape[1], MIN_TEXT_WIDTH) + 2 * MARGIN
    wrap = max(20, int((width - 2 * MARGIN) / (CAPTION_PT * 0.48)))
    captions = [textwrap.wrap(f.caption, wrap) or [""] for f in frames]
    line_h = int(CAPTION_PT * 1.35)
    top = MARGIN + int(TITLE_PT * 1.6)
    height = top + shape[0] + MARGIN // 2 + line_h * max(len(c) for c in captions) + MARGIN
    x0 = (width - shape[1]) // 2
    out = []
    for i, (f, lines) in enumerate(zip(frames, captions)):
        canvas = Image.new("RGB", (width, height), PAPER)
        canvas.paste(Image.fromarray(_flatten(f, shape)), (x0, top))
        d = ImageDraw.Draw(canvas)
        d.text((MARGIN, MARGIN), title, font=title_font, fill=MUTED_INK)
        counter = f"{i + 1} / {len(frames)}"
        d.text((width - MARGIN - d.textlength(counter, font=title_font), MARGIN), counter,
               font=title_font, fill=MUTED_INK)
        y = top + shape[0] + MARGIN // 2
        for line in lines:
            d.text((x0 if shape[1] >= MIN_TEXT_WIDTH else MARGIN, y), line, font=cap_font, fill=INK)
            y += line_h
        out.append(np.asarray(canvas))
    return out


def _flatten(f: Frame, shape: tuple[int, int]) -> np.ndarray:
    img = resize(np.ascontiguousarray(f.image[..., :3]), shape).astype(np.float32)
    if f.overlay is not None:
        ov = resize(np.ascontiguousarray(f.overlay), shape).astype(np.float32)
        a = ov[..., 3:4] / 255.0
        img = img * (1 - a) + ov[..., :3] * a
    return np.clip(img, 0, 255).astype(np.uint8)


def _palette(keys: list[np.ndarray], colours: int) -> Image.Image:
    """A P-mode image carrying one shared palette: ``colours`` from the frames plus reserved colours."""
    step = max(1, int(np.sqrt(sum(k.shape[0] * k.shape[1] for k in keys) / 250_000)))
    sample = np.concatenate([k[::step, ::step].reshape(-1, 3) for k in keys])
    side = int(np.ceil(np.sqrt(len(sample))))
    mosaic = np.zeros((side * side, 3), np.uint8)
    mosaic[:len(sample)] = sample
    mosaic[len(sample):] = PAPER
    q = Image.fromarray(mosaic.reshape(side, side, 3)).quantize(colors=colours, method=Image.Quantize.MEDIANCUT,
                                                                 dither=Image.Dither.NONE)
    ramp = [tuple(int(round(INK[c] + (PAPER[c] - INK[c]) * t)) for c in range(3)) for t in np.linspace(0, 1, 10)]
    reserved = np.array([PAPER, INK, OCHRE, MANMADE_RED, NATURAL_GREEN, OUTLINE, MUTED_INK, *ramp], np.int32)
    learned = np.array(q.getpalette()[:colours * 3], np.int32).reshape(-1, 3)
    # A learned colour a hair off a reserved one would steal its pixels (paper becoming 254,254,247).
    near = (np.abs(learned[:, None, :] - reserved[None, :, :]).max(axis=2) <= 4).any(axis=1)
    flat = [int(v) for v in np.concatenate([reserved, learned[~near]]).ravel()]
    flat += [0] * (768 - len(flat))
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat[:768])
    return pal


def _encode(keys: list[np.ndarray], frames: list[Frame], fades: int, colours: int) -> bytes:
    pal = _palette(keys, colours)
    images, durations = [], []
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        hold = LAST_HOLD_S if last else (frames[i].hold_s or DEFAULT_HOLD_S)
        images.append(k)
        durations.append(hold)
        if not last:
            for j in range(1, fades + 1):
                images.append(blend(k, keys[i + 1], j / (fades + 1)))
                durations.append(FADE_FRAME_S)
    ps = [Image.fromarray(a).quantize(palette=pal, dither=Image.Dither.NONE) for a in images]
    buf = io.BytesIO()
    ps[0].save(buf, format="GIF", save_all=True, append_images=ps[1:], loop=0, optimize=False,
               duration=[int(round(d * 1000)) for d in durations], disposal=1)
    return buf.getvalue()
