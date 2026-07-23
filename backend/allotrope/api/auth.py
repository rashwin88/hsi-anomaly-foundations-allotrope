"""Authentication endpoints: POST /auth/login, POST /auth/logout, GET /auth/me.

Sequence diagrams: final design/diagrams/auth-login.drawio,
                   final design/diagrams/auth-logout.drawio,
                   final design/diagrams/auth-me.drawio
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims, issue_token
from ..auth.password import hash_password, verify_password
from ..config import settings
from ..db import get_db
from ..models import User
from .deps import current_user_claims

router = APIRouter(prefix="/auth", tags=["auth"])


# Pre-computed at module load — used to keep login response time uniform
# regardless of whether the username exists. Without this, attackers can
# distinguish "no such user" (fast) from "wrong password" (slow argon2
# verify) via timing.
_DUMMY_PASSWORD_HASH = hash_password("dummy-for-timing-safety")


# --- Request / response models -----------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class UserPublic(BaseModel):
    """The User shape returned to clients. Wire id format: user_<uuid>."""

    id: str
    username: str
    email: str
    display_name: str | None
    is_admin: bool

    @classmethod
    def from_orm_user(cls, user: User) -> "UserPublic":
        return cls(
            id=f"user_{user.id}",
            username=user.username,
            email=user.email,
            display_name=user.display_name,
            is_admin=user.is_admin,
        )


class LoginResponse(BaseModel):
    user: UserPublic


# --- Routes -------------------------------------------------------------


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Sign in with username + password",
)
def login(
    creds: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> LoginResponse:
    """Verify credentials, issue a JWT, set it as an HttpOnly cookie.

    Failure → 401 with a generic error (no enumeration of valid usernames).
    Success → 200 with the user object in the body and the JWT in a cookie.
    """
    # Case-insensitive lookup uses our LOWER(username) functional unique index.
    user = db.scalar(
        select(User).where(func.lower(User.username) == creds.username.lower())
    )

    # Always run verify_password against either the real hash or a dummy
    # one. Constant-time-ish: avoids leaking "user exists" via response time.
    hash_to_check = user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
    password_ok = verify_password(hash_to_check, creds.password)

    if user is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_credentials",
        )

    # Update last_login_at — single DB write, in the same session.
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    # Issue + set cookie.
    token = issue_token(user)
    response.set_cookie(
        key=settings.jwt_cookie_name,
        value=token,
        max_age=settings.jwt_lifetime_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )

    return LoginResponse(user=UserPublic.from_orm_user(user))


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear the auth cookie",
)
def logout(response: Response) -> None:
    """Stateless logout: just clear the cookie.

    The token itself remains technically valid until its `exp`; the browser
    no longer holds it. No DB write. If the same token exists in another
    browser tab/window, that tab is unaffected — accepted cost of stateless
    JWT auth.
    """
    response.delete_cookie(
        key=settings.jwt_cookie_name,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
    )


@router.get(
    "/me",
    response_model=UserPublic,
    summary="Current user (claims + DB lookup)",
)
def me(
    claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> UserPublic:
    """Return the full User row for the currently-authenticated principal.

    Most authenticated endpoints don't need a DB read for auth (claims are
    sufficient for ownership filters). /me is the exception — the frontend
    calls it on app load to populate the user's profile, and the response
    includes fields not in the JWT claims (email, display_name).

    Returns 401 if the User was deleted while their token is still valid.
    """
    user_id = uuid.UUID(claims["sub"])
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="user_not_found",
        )
    return UserPublic.from_orm_user(user)
