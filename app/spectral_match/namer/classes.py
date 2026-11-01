"""
The namer's v1 material classes and how each family-library entry maps to one (user, 2026-10-31).

Classes follow materials, not uses: research/experiments/spectral-namer/README.md found "roofing" scattering (it holds
asphalt shingle, wood shake, metal and fibreglass) and building / pavement confusing each other, so roofing is
dissolved by material and building + pavement + asphalt become one built-mineral class. Glass has one spectrum and is
left out of v1 (it cannot be tested); camouflage and military coatings join when field spectra exist. A glass / solar
class built from in-scene PV spectra, without retraining, was tried on 2026-11-01 and did not work (in-scene pixels
never match in-scene spectra used as library entries): research/experiments/spectral-namer/pv-library/README.md.
"""

from __future__ import annotations

import re

MANMADE_CLASSES = ("oil_fuel", "plastic", "textile_fabric", "wood_paper", "paint_coatings", "metal", "concrete_asphalt")
NATURAL = "natural"
EXCLUDED = None

_DIRECT = {"oil": "oil_fuel", "plastic": "plastic", "textile": "textile_fabric", "wood_paper": "wood_paper",
           "paint": "paint_coatings", "metal": "metal", "building": "concrete_asphalt", "pavement": "concrete_asphalt"}
_ROOFING = [(r"fiberglass|rubber|clear_roof", "plastic"),
            (r"cedar|shake|wood", "wood_paper"),
            (r"metal|tin|steel|alumin", "metal"),
            (r"tar|felt|asphalt|shingle|roofingpaper|tile|asp", "concrete_asphalt")]


def class_of(chapter: str, name: str) -> str | None:
    """'natural', one of MANMADE_CLASSES, or None (left out) for a family-library entry's chapter and name."""
    group, _, family = chapter.partition(".")
    if group == "natural":
        return NATURAL
    if group != "manmade":
        return EXCLUDED
    if family in _DIRECT:
        return _DIRECT[family]
    if family == "roofing":
        low = name.lower()
        for pattern, cls in _ROOFING:
            if re.search(pattern, low):
                return cls
        return "concrete_asphalt"
    return EXCLUDED                               # glass (v1)
