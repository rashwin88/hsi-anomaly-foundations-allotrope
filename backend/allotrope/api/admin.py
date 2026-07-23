"""Admin-only endpoints.

Sequence diagram: final design/diagrams/admin-create-user.drawio

Routes here all require `is_admin=True` via the `require_admin` dependency.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..auth.password import hash_password
from ..db import get_db
from ..models import User
from .auth import UserPublic
from .deps import require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    email: str = Field(..., min_length=3, max_length=256)
    password: str = Field(..., min_length=8)
    display_name: str | None = Field(default=None, max_length=128)
    is_admin: bool = False


@router.post(
    "/users",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user (admin only)",
)
def create_user(
    payload: CreateUserRequest,
    _claims: Claims = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserPublic:
    """Create a new user. Admin-only.

    The unique constraints on `LOWER(username)` and `LOWER(email)` are the
    real safety net — we let the constraint catch races and only translate
    the IntegrityError to 409.
    """
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
        is_admin=payload.is_admin,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="username_or_email_taken",
        )
    db.refresh(user)
    return UserPublic.from_orm_user(user)
