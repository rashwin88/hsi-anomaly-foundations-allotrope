"""ActionTemplate CRUD (Step 18 — Models destination).

Routes:
    GET    /action-templates                  list (optional ?type=...)
    POST   /action-templates                  create a user template (is_system=False)
    GET    /action-templates/{id}             detail
    PATCH  /action-templates/{id}             rename / re-describe / update configuration
    DELETE /action-templates/{id}             delete (SET NULL on actions.action_template_id)

System templates (`is_system=True`) are seeded by the
`seed-action-templates` CLI and are read-only on PATCH/DELETE — the api
rejects mutations with 409 to preserve the audit story (the recipe a
demo shipped with should not be edited).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import action_types as action_types_registry
from ..auth.jwt import Claims
from ..db import get_db
from ..models import ActionTemplate
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.action_templates")

router = APIRouter(prefix="/action-templates", tags=["action-templates"])


# --- Wire schemas ----------------------------------------------------


class ActionTemplatePublic(BaseModel):
    id: str                              # action_template_<uuid>
    type: str
    name: str
    description: str | None
    configuration: dict[str, Any]
    is_system: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm(cls, t: ActionTemplate) -> "ActionTemplatePublic":
        return cls(
            id=f"action_template_{t.id}",
            type=t.type,
            name=t.name,
            description=t.description,
            configuration=t.configuration or {},
            is_system=t.is_system,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )


class ActionTemplateList(BaseModel):
    items: list[ActionTemplatePublic]


class ActionTemplateCreate(BaseModel):
    type: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    configuration: dict[str, Any] = Field(default_factory=dict)


class ActionTemplatePatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    configuration: dict[str, Any] | None = None


# --- Helpers ---------------------------------------------------------


def _template_or_404(template_id_wire: str, db: Session) -> ActionTemplate:
    raw = parse_prefixed_id("action_template", template_id_wire)
    t = db.get(ActionTemplate, raw)
    if t is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="template_not_found"
        )
    return t


def _validate_against_registry(
    type_: str, configuration: dict[str, Any]
) -> None:
    """Reject types unknown to the action_types registry; templates that
    don't conform to a real recipe are dead weight.

    Note: we deliberately don't validate the configuration body itself
    here — config validation depends on the target Scene's sensor (per
    action_types contract) and a template isn't bound to a scene. The
    Action submit flow re-validates at use time."""
    if type_ not in action_types_registry.REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unknown_action_type",
        )


# --- Routes ----------------------------------------------------------


@router.get(
    "",
    response_model=ActionTemplateList,
    status_code=status.HTTP_200_OK,
    summary="List ActionTemplates (optionally filtered by type)",
)
def list_templates(
    type_: str | None = Query(
        None,
        alias="type",
        description="Filter by action type slug.",
    ),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionTemplateList:
    filters = []
    if type_ is not None:
        filters.append(ActionTemplate.type == type_)
    # System templates first (stable headline of the picker), then
    # user templates by recency.
    rows = list(
        db.scalars(
            select(ActionTemplate)
            .where(*filters)
            .order_by(
                ActionTemplate.is_system.desc(),
                ActionTemplate.updated_at.desc(),
            )
        ).all()
    )
    return ActionTemplateList(
        items=[ActionTemplatePublic.from_orm(t) for t in rows]
    )


@router.post(
    "",
    response_model=ActionTemplatePublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user ActionTemplate",
)
def create_template(
    body: ActionTemplateCreate,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionTemplatePublic:
    _validate_against_registry(body.type, body.configuration)
    t = ActionTemplate(
        id=uuid.uuid4(),
        type=body.type,
        name=body.name,
        description=body.description,
        configuration=body.configuration,
        is_system=False,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    logger.info(
        "action_template created id=%s type=%s name=%r", t.id, t.type, t.name
    )
    return ActionTemplatePublic.from_orm(t)


@router.get(
    "/{template_id}",
    response_model=ActionTemplatePublic,
    status_code=status.HTTP_200_OK,
    summary="Get one ActionTemplate by id",
)
def get_template(
    template_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionTemplatePublic:
    return ActionTemplatePublic.from_orm(_template_or_404(template_id, db))


@router.patch(
    "/{template_id}",
    response_model=ActionTemplatePublic,
    status_code=status.HTTP_200_OK,
    summary="Edit a user ActionTemplate",
)
def patch_template(
    template_id: str,
    body: ActionTemplatePatch,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> ActionTemplatePublic:
    t = _template_or_404(template_id, db)
    if t.is_system:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="system_template_immutable",
        )
    if body.name is not None:
        t.name = body.name
    if body.description is not None:
        t.description = body.description
    if body.configuration is not None:
        t.configuration = body.configuration
    db.commit()
    db.refresh(t)
    return ActionTemplatePublic.from_orm(t)


@router.delete(
    "/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a user ActionTemplate (system templates can't be deleted)",
    response_class=Response,
)
def delete_template(
    template_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    t = _template_or_404(template_id, db)
    if t.is_system:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="system_template_immutable",
        )
    db.delete(t)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
