import pytest
from fastapi.testclient import TestClient

from tandemista.api.app import create_app
from tandemista.config import Settings
from tandemista.db.base import Base, SessionLocal


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_backend="local",
        storage_local_root=str(tmp_path / "storage"),
        celery_task_always_eager=True,
    )


@pytest.fixture()
def app(settings):
    application = create_app(settings)
    # create_app configured SessionLocal against the app engine; build the schema.
    Base.metadata.create_all(SessionLocal().get_bind())
    return application


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def db(app):
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
