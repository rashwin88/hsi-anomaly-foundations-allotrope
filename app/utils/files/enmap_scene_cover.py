"""
Scene-level cover percentages from an EnMAP METADATA.XML.

Used to stratify the segmentation train/test split. Cloud is present in a
small minority of scenes, so a random split can strand nearly all of it on
one side - which is exactly what happened to an earlier cloud-mask
experiment, where nine of twelve validation scenes contained no cloud at all
and the headline score rested on the other three.

Only the head of the file is read. All six tags sit within the first ~7 KB
of a ~4.3 MB document, so a 64 KB read is ~1/65th of the file. Screening 212
scenes this way costs single-digit megabytes rather than a gigabyte, which
matters when the scenes are on a mounted Google Drive.

Caution: these percentages are of the **imaged swath**, not of the raster.
Roughly a quarter of every EnMAP raster is off-swath background (value 3 in
QL_QUALITY_CLASSES), and it is excluded from the denominator. A reported
`waterCover` of 76 corresponds to ~55% of the pixels in the file.
"""

import glob
import os
import re

COVER_TAGS: tuple[str, ...] = (
    "cloudCover",
    "cloudShadow",
    "hazeCover",
    "cirrusCover",
    "snowCover",
    "waterCover",
)


def read_scene_cover(scene_folder: str, head_bytes: int = 65536) -> dict[str, float]:
    """Cover percentages for one scene folder, keyed by tag name.

    Tags that are absent or unparseable are omitted rather than defaulted,
    so a caller can tell "reported as zero" from "not reported".

    Raises FileNotFoundError if the folder holds no METADATA.XML.
    """
    matches = glob.glob(os.path.join(scene_folder, "*METADATA.XML"))
    if not matches:
        raise FileNotFoundError(f"no METADATA.XML in {scene_folder}")

    with open(matches[0], "rb") as handle:
        head = handle.read(head_bytes).decode("utf-8", errors="ignore")

    found: dict[str, float] = {}
    for tag in COVER_TAGS:
        match = re.search(rf"<{tag}>([^<]*)</{tag}>", head)
        if match is None:
            continue
        try:
            found[tag] = float(match.group(1))
        except ValueError:
            continue
    return found
