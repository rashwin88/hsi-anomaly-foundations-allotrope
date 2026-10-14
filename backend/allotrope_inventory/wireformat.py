"""Prefixed wire ids (`ds_<uuid>`, `sensor_<uuid>`) — same convention as
allotrope/api/wireformat.py, duplicated so this service imports nothing from
the main app."""

import uuid

from fastapi import HTTPException, status


def parse_prefixed_id(prefix: str, value: str) -> uuid.UUID:
    """`parse_prefixed_id("ds", "ds_3f29…")` -> UUID; 422 bad_id_format otherwise."""
    head, _, raw = value.partition("_")
    try:
        if head != prefix:
            raise ValueError(value)
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="bad_id_format")


def to_prefixed(prefix: str, value: uuid.UUID | None) -> str | None:
    """Inverse of parse_prefixed_id. None passes through unchanged."""
    return f"{prefix}_{value}" if value is not None else None
