# backend/tandemista/api/app.py
from __future__ import annotations

from fastapi import FastAPI

from ..config import Settings, get_settings
from ..db.base import configure_session, make_engine
from ..storage.factory import build_storage
from ..worker.celery_app import celery


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    configure_session(engine)
    # The Celery app is a process-wide singleton bound at import time to
    # env-based settings. Rebind its eager flag to this app's settings so
    # tests (and any alternate Settings) actually control whether
    # `.delay()` dispatches to a real broker or runs inline.
    celery.conf.task_always_eager = settings.celery_task_always_eager

    app = FastAPI(title="happy_tandemista API")
    app.state.settings = settings
    app.state.engine = engine
    app.state.storage = build_storage(settings)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    from .routes import dropzones, jumps, media

    app.include_router(dropzones.router)
    app.include_router(jumps.router)
    app.include_router(media.router)

    return app
