# backend/tandemista/api/app.py
from __future__ import annotations

from fastapi import FastAPI

from ..config import Settings, get_settings
from ..db.base import configure_session, make_engine
from ..storage.factory import build_storage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    configure_session(engine)

    app = FastAPI(title="happy_tandemista API")
    app.state.settings = settings
    app.state.engine = engine
    app.state.storage = build_storage(settings)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
