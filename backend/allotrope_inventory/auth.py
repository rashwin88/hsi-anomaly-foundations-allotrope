"""API-key auth for the inventory service.

Every route depends on `require_key`, which returns the key's name — the
actor recorded in created_by and the audit log. Keys come from
INVENTORY_API_KEYS as comma-separated `name:key` pairs. The browser UI will
reach this service through nginx later; until then, scripts and the CLI are
the only clients.
"""

import hmac
from functools import cache

from fastapi import Header, HTTPException, status

from .config import settings


@cache
def _keys() -> dict[str, str]:
    """`{key: name}` parsed once from settings."""
    pairs = (p.strip().split(":", 1) for p in settings.api_keys.split(",") if p.strip())
    return {key: name for name, key in pairs}


def require_key(x_inventory_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: 401 unless X-Inventory-Key matches a configured key."""
    if x_inventory_key is not None:
        for key, name in _keys().items():
            # Constant-time compare so response timing doesn't leak the key.
            if hmac.compare_digest(key, x_inventory_key):
                return name
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthenticated")
