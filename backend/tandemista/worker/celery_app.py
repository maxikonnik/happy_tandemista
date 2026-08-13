# backend/tandemista/worker/celery_app.py
from __future__ import annotations

from celery import Celery
from celery.signals import worker_process_init

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
    return celery


celery = make_celery()


# Bind the worker's DB session on actual worker process boot, not at module
# import time. Other processes (e.g. the API, to enqueue tasks) only need the
# task signatures and must be free to import this module without clobbering
# their own SessionLocal binding; only a real Celery worker process should
# ever rebind it.
@worker_process_init.connect
def _bind_worker_session(**_kwargs) -> None:
    settings = get_settings()
    configure_session(make_engine(settings.database_url))
