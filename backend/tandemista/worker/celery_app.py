# backend/tandemista/worker/celery_app.py
from __future__ import annotations

from celery import Celery

from ..config import Settings, get_settings
from ..db.base import configure_session, make_engine


def make_celery(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    celery = Celery(
        "tandemista",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    celery.conf.task_always_eager = settings.celery_task_always_eager
    celery.conf.task_serializer = "json"
    celery.conf.result_serializer = "json"
    celery.conf.accept_content = ["json"]
    # Bind the worker's DB session factory to the configured database.
    configure_session(make_engine(settings.database_url))
    return celery


celery = make_celery()
