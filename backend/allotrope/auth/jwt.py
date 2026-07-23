"""JWT issuance + verification (HS256, stateless).

See `final design.md` "Auth flow end-to-end" for the full sequence diagram
and "Auth Flow (JWT)" sheet in er-diagram.drawio for the visual.

Design choices locked elsewhere:
- Algorithm: HS256 (single secret on the api). Simpler than RS256/ES256;
  pays off only when third parties verify tokens.
- Lifetime: 24h default. No refresh-token flow in v1.
- Storage: HttpOnly + SameSite=Strict cookie set by the api (this module
  doesn't touch cookies — that's the endpoint's job).
- Claims: `sub` (User uuid), `iat`, `exp`, `username`, `is_admin`.
  We include `is_admin` in the claims so admin checks don't need a DB
  lookup; the trade-off is that an admin demoted while a token is in flight
  retains admin rights until the token expires (≤24h). Acceptable for v1.
"""

from datetime import datetime, timedelta, timezone
from typing import TypedDict

import jwt

from ..config import settings
from ..models import User

ALGORITHM = "HS256"


class Claims(TypedDict):
    sub: str          # User.id as a string
    iat: int          # issued-at epoch seconds
    exp: int          # expires-at epoch seconds
    username: str
    is_admin: bool


def issue_token(user: User) -> str:
    """Build and sign a JWT for the given user.

    The output is a single string of the form `header.payload.signature`,
    suitable for placing in a Set-Cookie response header.
    """
    now = datetime.now(timezone.utc)
    payload: Claims = {
        "sub": str(user.id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.jwt_lifetime_seconds)).timestamp()),
        "username": user.username,
        "is_admin": user.is_admin,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> Claims:
    """Verify the signature, check `exp`, and return the claims.

    Raises:
        jwt.ExpiredSignatureError    — token's `exp` is in the past.
        jwt.InvalidSignatureError    — signature doesn't match secret.
        jwt.InvalidTokenError        — malformed token (parent class).
    Callers should catch `InvalidTokenError` to handle all three.
    """
    return jwt.decode(  # type: ignore[no-any-return]
        token,
        settings.jwt_secret,
        algorithms=[ALGORITHM],
    )
