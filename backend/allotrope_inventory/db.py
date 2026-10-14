"""Inventory service — SQLAlchemy engine + session factory.

Same shape as allotrope/db.py, pointed at the inventory database.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one Session per request, closed on request end."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
