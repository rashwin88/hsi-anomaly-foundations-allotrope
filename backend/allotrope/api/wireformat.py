"""Wire-format helpers for prefixed UUIDs (CC-1).

The api boundary uses `<entity>_<uuid>` strings (e.g. `job_3f29c4a8-…`).
Inside the DB / SQLAlchemy, ids are bare UUIDs. These helpers do the
back-and-forth in one place so individual routes stay tidy.

We deliberately do NOT use FastAPI Path validators / regex types — a clear
HTTPException with detail="bad_id_format" beats a 422 with a regex error
the frontend then has to interpret.
"""

import uuid

from fastapi import HTTPException, status


def parse_prefixed_id(prefix: str, value: str) -> uuid.UUID:
    """Strip `<prefix>_` and parse the rest as a UUID.

    Example:
        parse_prefixed_id("job", "job_3f29c4a8-…") -> UUID("3f29c4a8-…")

    Raises 422 with `detail="bad_id_format"` for unparseable input — same
    shape FastAPI uses for its own body-validation errors.
    """
    expected = f"{prefix}_"
    if not value.startswith(expected):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_id_format",
        )
    raw = value[len(expected):]
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_id_format",
        )


def to_prefixed(prefix: str, value: uuid.UUID | None) -> str | None:
    """Inverse of parse_prefixed_id. None passes through unchanged."""
    return f"{prefix}_{value}" if value is not None else None
