"""Shared FastAPI dependencies.

`current_user_claims` is the auth gate for every authenticated route. It
reads the JWT cookie, verifies the signature + expiry, and returns the
decoded claims. Stateless — does NOT touch the DB.

`require_admin` composes on top: same auth gate + asserts is_admin=True.

For the (rare) endpoints that need the full User row (e.g. /auth/me), do
a DB lookup using `claims["sub"]` after this dependency resolves.
"""

import jwt as pyjwt
from fastapi import Cookie, Depends, HTTPException, status

from ..auth.jwt import Claims, decode_token
from ..config import settings


def current_user_claims(
    token: str | None = Cookie(default=None, alias=settings.jwt_cookie_name),
) -> Claims:
    """Decode the auth cookie's JWT and return its claims.

    Raises 401 with a specific detail code so the frontend can route cleanly:
    - "unauthenticated"   — no cookie, or malformed/invalid signature
    - "session_expired"   — token exp is in the past (frontend → /login)
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthenticated",
        )

    try:
        return decode_token(token)
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="session_expired",
        )
    except pyjwt.InvalidTokenError:
        # Malformed token, bad signature, wrong algorithm, etc.
        # Lump them together — never leak which one failed.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthenticated",
        )


def require_admin(
    claims: Claims = Depends(current_user_claims),
) -> Claims:
    """Require the caller to be authenticated AND have is_admin=True.

    Composes on top of `current_user_claims` — FastAPI resolves the chain
    automatically. Returns the same claims so endpoints don't have to
    re-declare the dependency.

    Distinction:
    - 401 unauthenticated → not logged in (handled by current_user_claims)
    - 403 forbidden       → logged in but lacks admin (handled here)
    """
    if not claims["is_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="forbidden",
        )
    return claims
