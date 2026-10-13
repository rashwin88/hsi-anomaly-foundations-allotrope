"""SQLAlchemy 2.x declarative base for the inventory database."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all inventory ORM models."""

    pass
