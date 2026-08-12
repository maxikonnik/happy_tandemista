import uuid

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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


def test_analyze_media_recovers_on_broken_transaction(eager_db):
    """I3: force a genuine IntegrityError while flushing the ANALYZED
    status (by nulling out the NOT NULL `filename` column on the same
    object right before flush). This leaves the SQLAlchemy session in the
    real "transaction rolled back due to a previous exception during
    flush" state -- the exact situation the except block must recover
    from by rolling back before it re-fetches and commits FAILED.
    """
    from tandemista.worker.tasks import analyze_media

    media_id = _make_media()

    def _break_analyzed_write(session, flush_context, instances):
        for obj in session.dirty:
            if isinstance(obj, m.MediaFile) and obj.status == m.MediaStatus.ANALYZED:
                obj.filename = None  # violates NOT NULL -> real IntegrityError

    event.listen(Session, "before_flush", _break_analyzed_write)
    try:
        with pytest.raises(IntegrityError):
            analyze_media.run(str(media_id))
    finally:
        event.remove(Session, "before_flush", _break_analyzed_write)

    with SessionLocal() as s:
        row = s.get(m.MediaFile, media_id)
        assert row.status == m.MediaStatus.FAILED
        # filename must be untouched: the FAILED write only refetched the
        # row and set status; it never persisted the poisoned None value.
        assert row.filename == "a.mp4"


def test_celery_configured_eager_from_settings():
    from tandemista.worker.celery_app import make_celery

    celery = make_celery(Settings(celery_task_always_eager=True))
    assert celery.conf.task_always_eager is True
