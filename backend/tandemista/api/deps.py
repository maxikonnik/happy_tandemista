# backend/tandemista/api/deps.py
from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ..config import Settings
from ..db.base import SessionLocal
from ..storage.base import StorageBackend


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_storage(request: Request) -> StorageBackend:
    return request.app.state.storage


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
