import uuid

import pytest

from tandemista.config import Settings
from tandemista.db.base import Base, configure_session, make_engine, SessionLocal
from tandemista.db import models as m


@pytest.fixture()
def eager_db():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    configure_session(engine)
    return engine


def _make_media(status=m.MediaStatus.REGISTERED) -> uuid.UUID:
    with SessionLocal() as s:
        dz = m.Dropzone(name="DZ")
        s.add(dz)
        s.flush()
        media = m.MediaFile(dropzone_id=dz.id, filename="a.mp4", status=status)
        s.add(media)
        s.commit()
        return media.id


def test_analyze_media_transitions_to_analyzed(eager_db):
    from tandemista.worker.tasks import analyze_media

    media_id = _make_media()
    result = analyze_media.run(str(media_id))
    assert result == "analyzed"
    with SessionLocal() as s:
        assert s.get(m.MediaFile, media_id).status == m.MediaStatus.ANALYZED


def test_analyze_media_missing_row(eager_db):
    from tandemista.worker.tasks import analyze_media

    assert analyze_media.run(str(uuid.uuid4())) == "missing"


def test_celery_configured_eager_from_settings():
    from tandemista.worker.celery_app import make_celery

    celery = make_celery(Settings(celery_task_always_eager=True))
    assert celery.conf.task_always_eager is True
