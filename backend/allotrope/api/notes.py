"""Notes + NoteReferences endpoints (Step 16).

Routes:
    POST   /projects/{project_id}/notes  — create (synchronous)
    GET    /projects/{project_id}/notes  — list (newest first)
    GET    /notes/{note_id}              — detail (incl. resolved references)
    PATCH  /notes/{note_id}              — edit content + replace references
    DELETE /notes/{note_id}              — synchronous delete (cascades references)

Reference model
---------------
The api keeps reference parsing out of the wire — the frontend sends an
explicit `references` list of wire ids (e.g. `action_<uuid>`,
`scene_<uuid>`) alongside the markdown content. Easier to validate, no
fragile @-mention regex, and roundtrips cleanly through PATCH.

On create / PATCH we diff the supplied refs against what's in the DB,
delete what's gone, insert what's new. All references must point at
entities inside the same Project (Scene is matched against
project.scene_id; Actions / Outputs are checked by walking back to
their project; Visualizations carry project_id directly).

Sequence diagram: final design/diagrams/notes-create.drawio
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.jwt import Claims
from ..db import get_db
from ..models import (
    Action,
    ActionOutput,
    Note,
    NoteReference,
    Project,
    Scene,
    Visualization,
)
from .deps import current_user_claims
from .wireformat import parse_prefixed_id

logger = logging.getLogger("allotrope.api.notes")

project_notes_router = APIRouter(
    prefix="/projects/{project_id}/notes", tags=["notes"]
)
notes_router = APIRouter(prefix="/notes", tags=["notes"])


RefKind = Literal["project", "action", "output", "viz", "scene"]


# --- Wire schemas ----------------------------------------------------


class NoteReferencePublic(BaseModel):
    id: str  # note_ref_<uuid>
    kind: RefKind
    target_id: str  # wire id of the referenced entity
    created_at: datetime


class NoteCreate(BaseModel):
    content: str = Field(default="", max_length=200_000)
    references: list[str] = Field(
        default_factory=list,
        description=(
            "Wire ids of referenced entities — project_<uuid>, action_<uuid>, "
            "output_<uuid>, viz_<uuid>, or scene_<uuid>."
        ),
    )


class NotePatch(BaseModel):
    content: str | None = Field(default=None, max_length=200_000)
    references: list[str] | None = None


class NotePublic(BaseModel):
    id: str  # note_<uuid>
    project_id: str
    content: str
    references: list[NoteReferencePublic]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm(
        cls, note: Note, refs: list[NoteReference]
    ) -> "NotePublic":
        return cls(
            id=f"note_{note.id}",
            project_id=f"project_{note.project_id}",
            content=note.content,
            references=[_ref_to_public(r) for r in refs],
            created_at=note.created_at,
            updated_at=note.updated_at,
        )


class NoteList(BaseModel):
    items: list[NotePublic]
    total: int
    limit: int
    offset: int


# --- Helpers ---------------------------------------------------------


def _project_or_404(project_id_wire: str, db: Session) -> Project:
    raw = parse_prefixed_id("project", project_id_wire)
    project = db.get(Project, raw)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="project_not_found"
        )
    return project


def _note_or_404(note_id_wire: str, db: Session) -> Note:
    raw = parse_prefixed_id("note", note_id_wire)
    note = db.get(Note, raw)
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="note_not_found"
        )
    return note


def _ref_to_public(r: NoteReference) -> NoteReferencePublic:
    if r.ref_project_id is not None:
        return NoteReferencePublic(
            id=f"note_ref_{r.id}",
            kind="project",
            target_id=f"project_{r.ref_project_id}",
            created_at=r.created_at,
        )
    if r.ref_action_id is not None:
        return NoteReferencePublic(
            id=f"note_ref_{r.id}",
            kind="action",
            target_id=f"action_{r.ref_action_id}",
            created_at=r.created_at,
        )
    if r.ref_output_id is not None:
        return NoteReferencePublic(
            id=f"note_ref_{r.id}",
            kind="output",
            target_id=f"output_{r.ref_output_id}",
            created_at=r.created_at,
        )
    if r.ref_viz_id is not None:
        return NoteReferencePublic(
            id=f"note_ref_{r.id}",
            kind="viz",
            target_id=f"viz_{r.ref_viz_id}",
            created_at=r.created_at,
        )
    if r.ref_scene_id is not None:
        return NoteReferencePublic(
            id=f"note_ref_{r.id}",
            kind="scene",
            target_id=f"scene_{r.ref_scene_id}",
            created_at=r.created_at,
        )
    # CHECK constraint should make this unreachable.
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="note_reference_empty",
    )


def _kind_and_uuid(wire_id: str) -> tuple[RefKind, uuid.UUID]:
    """Map a wire id to (kind, uuid). 422 on unknown prefix."""
    if wire_id.startswith("project_"):
        return "project", parse_prefixed_id("project", wire_id)
    if wire_id.startswith("action_"):
        return "action", parse_prefixed_id("action", wire_id)
    if wire_id.startswith("output_"):
        return "output", parse_prefixed_id("output", wire_id)
    if wire_id.startswith("viz_"):
        return "viz", parse_prefixed_id("viz", wire_id)
    if wire_id.startswith("scene_"):
        return "scene", parse_prefixed_id("scene", wire_id)
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="bad_reference_kind",
    )


def _validate_ref_in_project(
    kind: RefKind, raw: uuid.UUID, project: Project, db: Session
) -> None:
    """Raise 422 unless the referenced entity belongs to this Project."""
    if kind == "project":
        if raw != project.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="reference_not_in_project",
            )
        return
    if kind == "scene":
        if raw != project.scene_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="reference_not_in_project",
            )
        return
    if kind == "action":
        action = db.get(Action, raw)
        if action is None or action.project_id != project.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="reference_not_in_project",
            )
        return
    if kind == "output":
        output = db.get(ActionOutput, raw)
        if output is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="reference_not_in_project",
            )
        action = db.get(Action, output.action_id)
        if action is None or action.project_id != project.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="reference_not_in_project",
            )
        return
    if kind == "viz":
        viz = db.get(Visualization, raw)
        if viz is None or viz.project_id != project.id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="reference_not_in_project",
            )
        return


def _ref_column_for_kind(kind: RefKind) -> str:
    return {
        "project": "ref_project_id",
        "action": "ref_action_id",
        "output": "ref_output_id",
        "viz": "ref_viz_id",
        "scene": "ref_scene_id",
    }[kind]


def _build_reference_rows(
    note_id: uuid.UUID,
    references: list[str],
    project: Project,
    db: Session,
) -> list[NoteReference]:
    """Validate each wire id, dedupe, and produce NoteReference rows."""
    seen: set[tuple[RefKind, uuid.UUID]] = set()
    rows: list[NoteReference] = []
    for wire in references:
        kind, raw = _kind_and_uuid(wire)
        if (kind, raw) in seen:
            continue
        seen.add((kind, raw))
        _validate_ref_in_project(kind, raw, project, db)
        kwargs = {_ref_column_for_kind(kind): raw}
        rows.append(NoteReference(note_id=note_id, **kwargs))
    return rows


def _load_refs(note_id: uuid.UUID, db: Session) -> list[NoteReference]:
    return list(
        db.scalars(
            select(NoteReference)
            .where(NoteReference.note_id == note_id)
            .order_by(NoteReference.created_at.asc())
        ).all()
    )


# --- POST /projects/{id}/notes ---------------------------------------


@project_notes_router.post(
    "",
    response_model=NotePublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Note inside a Project",
)
def create_note(
    project_id: str,
    body: NoteCreate,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> NotePublic:
    project = _project_or_404(project_id, db)
    note = Note(project_id=project.id, content=body.content)
    db.add(note)
    db.flush()  # populate note.id without committing
    rows = _build_reference_rows(note.id, body.references, project, db)
    for r in rows:
        db.add(r)
    db.commit()
    db.refresh(note)
    refs = _load_refs(note.id, db)
    logger.info(
        "note created id=%s project=%s refs=%d", note.id, project.id, len(refs)
    )
    return NotePublic.from_orm(note, refs)


# --- GET /projects/{id}/notes ----------------------------------------


@project_notes_router.get(
    "",
    response_model=NoteList,
    status_code=status.HTTP_200_OK,
    summary="List Notes for a Project (newest first)",
)
def list_notes(
    project_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> NoteList:
    project = _project_or_404(project_id, db)
    total = (
        db.scalar(
            select(func.count(Note.id)).where(Note.project_id == project.id)
        )
        or 0
    )
    notes = list(
        db.scalars(
            select(Note)
            .where(Note.project_id == project.id)
            .order_by(Note.updated_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    )
    items = [NotePublic.from_orm(n, _load_refs(n.id, db)) for n in notes]
    return NoteList(items=items, total=total, limit=limit, offset=offset)


# --- GET /notes/{note_id} --------------------------------------------


@notes_router.get(
    "/{note_id}",
    response_model=NotePublic,
    status_code=status.HTTP_200_OK,
    summary="Get one Note by id",
)
def get_note(
    note_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> NotePublic:
    note = _note_or_404(note_id, db)
    return NotePublic.from_orm(note, _load_refs(note.id, db))


# --- PATCH /notes/{note_id} ------------------------------------------


@notes_router.patch(
    "/{note_id}",
    response_model=NotePublic,
    status_code=status.HTTP_200_OK,
    summary="Edit a Note's content and/or replace its references",
)
def patch_note(
    note_id: str,
    body: NotePatch,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> NotePublic:
    note = _note_or_404(note_id, db)
    if body.content is not None:
        note.content = body.content
    if body.references is not None:
        project = db.get(Project, note.project_id)
        if project is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="project_missing_for_note",
            )
        # Wipe-and-recreate. References don't carry surrogate meaning
        # outside the (note, target) edge, so we don't bother diffing.
        db.execute(
            NoteReference.__table__.delete().where(
                NoteReference.note_id == note.id
            )
        )
        for row in _build_reference_rows(note.id, body.references, project, db):
            db.add(row)
    db.commit()
    db.refresh(note)
    refs = _load_refs(note.id, db)
    return NotePublic.from_orm(note, refs)


# --- DELETE /notes/{note_id} -----------------------------------------


@notes_router.delete(
    "/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a Note (cascades references)",
    response_class=Response,
)
def delete_note(
    note_id: str,
    _claims: Claims = Depends(current_user_claims),
    db: Session = Depends(get_db),
) -> Response:
    note = _note_or_404(note_id, db)
    db.delete(note)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
