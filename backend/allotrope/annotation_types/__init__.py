"""Annotation type registry (abstractions-spec § 5.3, CC-13a).

Each annotation type is a module under this package that conforms to
the `AnnotationTypeSpec` Protocol. The registry maps `KIND` strings to
spec instances; the api uses it for upload validation, the worker uses
it for dispatch.

Adding a new type:
    1. Create `<kind>.py` with KIND / LABEL / ACCEPTED_EXTENSIONS plus
       the four lifecycle functions (validate_upload / materialise /
       render_overlay / extract_metadata).
    2. Import it below and add a registry entry.

Both the api and worker import this module. Don't put rasterio /
scipy / Pillow imports at module top-level — only inside the functions
that need them, so the api process (which doesn't render) doesn't pay
the import cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class AnnotationTypeSpec(Protocol):
    """Per-type contract.

    Every type module MUST expose these as module-level attributes /
    functions. We use a Protocol (not an abstract base class) so each
    module is just a flat collection of constants + functions — no
    instances, no boilerplate.
    """

    KIND: str
    LABEL: str
    ACCEPTED_EXTENSIONS: tuple[str, ...]

    @staticmethod
    def validate_upload(filename: str) -> None:
        """API-side cheap reject. Raise ValueError for known-bad input.

        Should be fast — extension / filename checks only. Don't open
        the file; that happens in the worker.
        """
        ...

    @staticmethod
    def materialise(staging_file: Path, final_dir: Path) -> Path:
        """Worker-side: move the staged file into its final location.

        Some types may transform here (e.g., normalise CRS, simplify
        geometry, compress). Returns the absolute path of the file
        actually written (caller stores its relative form on the row).
        """
        ...

    @staticmethod
    def render_overlay(
        final_file: Path,
        dest_png: Path,
        *,
        radius: int | None = None,
    ) -> bool:
        """Render an RGBA overlay PNG for the panzoom viewport.

        `radius` is an optional type-specific tuning knob (output-pixel
        scale for dot-style overlays; types that don't use a radius
        should ignore it). The api's overlay endpoint forwards a
        `?radius=` query param into this kwarg so the frontend can
        adjust dot size per annotation without re-onboarding.

        Returns True if a PNG was written, False if this type has no
        spatial overlay (the api's `has_overlay` flag flips accordingly
        and the frontend hides the overlay toggle for the row).
        """
        ...

    @staticmethod
    def extract_metadata(final_file: Path) -> dict[str, Any]:
        """Type-specific extras — written into Annotation.metadata JSONB.

        Used by the UI for display (e.g., feature count, class palette,
        wavelength range) and by future analyses for filtering.
        """
        ...


# --- Registry ---------------------------------------------------------

# Imported lazily to keep this top-level light. Each module's heavy deps
# (rasterio, scipy, etc.) only load when something actually calls into it.
from . import raster_mask  # noqa: E402

REGISTRY: dict[str, AnnotationTypeSpec] = {
    raster_mask.KIND: raster_mask,
}


def get_spec(kind: str) -> AnnotationTypeSpec:
    """Look up a type spec by KIND. Raises KeyError on unknown kind."""
    spec = REGISTRY.get(kind)
    if spec is None:
        raise KeyError(f"unknown annotation type: {kind!r}")
    return spec


def infer_kind_from_filename(filename: str) -> str | None:
    """Best-guess type kind from a filename's extension.

    Matches against `ACCEPTED_EXTENSIONS` of every registered type. If
    *exactly one* type accepts the extension, returns its KIND. If
    multiple types accept it (e.g., several types both take `.tif`),
    returns None — caller must ask the user. If no type accepts it,
    also returns None.

    Used by the `POST /scenes/{id}/annotations` endpoint when the
    `annotation_type` form field is omitted: the api sniffs the
    filename and only falls back to "ask the user" when the extension
    is ambiguous across types.
    """
    if not filename:
        return None
    lower = filename.lower()
    matches: list[str] = []
    for kind, spec in REGISTRY.items():
        if any(lower.endswith(ext) for ext in spec.ACCEPTED_EXTENSIONS):
            matches.append(kind)
    if len(matches) == 1:
        return matches[0]
    return None


def supported_kinds() -> list[str]:
    """Sorted list of registered KIND strings."""
    return sorted(REGISTRY.keys())


def public_catalog() -> list[dict[str, Any]]:
    """Lightweight catalogue for the api `/annotation-types` endpoint
    and the frontend's form."""
    return [
        {
            "kind": s.KIND,
            "label": s.LABEL,
            "accepted_extensions": list(s.ACCEPTED_EXTENSIONS),
        }
        for s in REGISTRY.values()
    ]
