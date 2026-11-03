"""
The one place the product's name and version are set.

Read by the export endpoints (every MANIFEST.json's ``software_version`` is
``f"allotrope/{__version__}"``), by ``GET /version`` for the UI, and mirrored by hand in
``frontend/src/version.ts`` and ``frontend/package.json`` (the frontend has no build-time
access to Python). Bump all three together at release.
"""

APP_NAME = "Allotrope"
__version__ = "1.0.2"
