"""
Build the training courses into single-file web pages.

Each course folder of Markdown parts (`NN-*.md`, plus an optional index file as part 00)
becomes one self-contained HTML page: the parts are embedded as JSON and rendered in the
browser by marked, with a section nav, a spectrum progress strip and the interactive
figures defined in `template.html`.

Usage:
    python scripts/course_site/build.py                 # every course
    python scripts/course_site/build.py indradhanu      # one course
    python scripts/course_site/build.py --out DIR       # default build/course_site/

The pages are published as claude.ai Artifacts (URLs below); `build/` is gitignored, so
the HTML itself is never committed. Republish to the same URL after a rebuild.

  indradhanu  https://claude.ai/artifact/5NcTMpErXsgJ6AdKVyiQHn
  anomaly     https://claude.ai/artifact/YJd3G3YiPr6EK2J8z5LCsZ

Two kinds of figure:
  - Static: an SVG in the course folder's `figures/`, referenced from the Markdown as
    `![Figure 16.1 - title](figures/16-encoder.svg)` followed by an italic `*Figure 16.1 - ...*`
    caption paragraph, so it also renders on GitHub. The build inlines each one as a data: URI
    (the page is a single file and cannot load relative paths); the template wraps it in a
    white card with the caption underneath.
  - Interactive: write a `fig<Name>(host)` function in `template.html` and list it under the
    course's key in `FIGSETS` against the part number. It is inserted after that part's
    "one thing this part teaches" blockquote.
`se_spectrum.json` is real data - the median and 10th/90th percentile reflectance of 128,000
SpectralEarth pixels (500 patches x 256), from `scripts/spectralearth_sanity.py`.
"""

import argparse
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

COURSES = {
    "indradhanu": {
        "folder": "indradhanu workings", "index": None, "out": "indradhanu_course.html",
        "title": "Indradhanu, from Zero", "brand": "Indradhanu", "brand_small": "from zero",
        "storageKey": "indradhanu-course-read", "figs": "indradhanu",
        "sections": [["Start", 0, 0], ["Background - what and why", 1, 5], ["Foundations", 6, 8],
                     ["The machine, piece by piece", 9, 22], ["Training and running it", 23, 30],
                     ["Sequel - the next Indradhanu", 31, 99]],
        "notes": {"31": "Designed, then built and measured (parts 40-41)."},
    },
    "anomaly": {
        "folder": "docs/training/anomaly-detection", "index": "README.md", "out": "anomaly_course.html",
        "title": "Finding Anomalies", "brand": "Finding anomalies", "brand_small": "a course",
        "storageKey": "anomaly-course-read", "figs": "anomaly",
        "sections": [["Start", 0, 0], ["The ground - data and product", 1, 5],
                     ["Detecting", 6, 12], ["Putting it together", 13, 99]],
        "notes": {},
    },
}


FIGURE_LINK = re.compile(r"\]\((figures/[^)\s]+\.svg)\)")


def inline_figures(md: str, folder: Path) -> str:
    """Rewrite `](figures/x.svg)` image links to base64 data: URIs. Raises FileNotFoundError
    naming the part's missing figure, rather than publishing a page with a broken image."""
    def to_data_uri(m: re.Match) -> str:
        path = folder / m.group(1)
        if not path.is_file():
            raise FileNotFoundError(f"figure referenced but missing: {path}")
        return "](data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode("ascii") + ")"
    return FIGURE_LINK.sub(to_data_uri, md)


def load_parts(course: dict) -> list[dict]:
    src = REPO / course["folder"]
    files = [(0, src / course["index"])] if course["index"] else []
    files += [(int(f.name[:2]), f) for f in sorted(src.glob("[0-9][0-9]-*.md"))]
    parts = []
    for num, f in files:
        md = inline_figures(f.read_text(encoding="utf-8"), src)
        # "# 38 - Title", "# Part 1 - Title" -> "Title"
        title = re.sub(r"^#\s*(Part\s+)?(\d+\s*[·\-]\s*)?", "", md.splitlines()[0]).strip()
        parts.append({"num": num, "title": "Start here" if num == 0 else title, "md": md})
    return parts


def build(name: str, out_dir: Path) -> Path:
    course = COURSES[name]
    parts = load_parts(course)
    cfg = {k: course[k] for k in ("folder", "title", "storageKey", "figs", "sections", "notes")}
    # json inside <script> must not contain "</" or the browser ends the script early
    safe = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    html = (HERE / "template.html").read_text(encoding="utf-8")
    spectrum = json.loads((HERE / "se_spectrum.json").read_text(encoding="utf-8"))
    for key, value in {"__TITLE__": course["title"], "__BRAND__": course["brand"],
                       "__BRAND_SMALL__": course["brand_small"], "__CONFIG_JSON__": safe(cfg),
                       "__COURSE_JSON__": safe(parts), "__SPECTRUM_JSON__": safe(spectrum)}.items():
        html = html.replace(key, value)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / course["out"]
    out.write_text(html, encoding="utf-8", newline="\n")
    print(f"{name}: {len(parts)} parts, {out.stat().st_size / 1024:.0f} KB -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("courses", nargs="*", help=f"any of {', '.join(COURSES)}; default all")
    ap.add_argument("--out", type=Path, default=REPO / "build" / "course_site")
    args = ap.parse_args()
    if unknown := set(args.courses) - set(COURSES):
        ap.error(f"unknown course(s): {', '.join(sorted(unknown))}")
    for name in args.courses or COURSES:
        build(name, args.out)


if __name__ == "__main__":
    main()
