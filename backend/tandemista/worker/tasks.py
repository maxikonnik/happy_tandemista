# backend/tandemista/worker/tasks.py
from __future__ import annotations

from ..db.base import SessionLocal
from ..db import models as m
from .celery_app import celery


@celery.task(name="tandemista.analyze_media")
def analyze_media(media_id: str) -> str:
    """Stub pipeline: mark media ANALYZING then ANALYZED.

    The real engine call (telemetry -> timeline -> EDL -> render) lands in the
    pipeline plan. Failures set status FAILED and re-raise: material is never
    dropped silently.
    """
    with SessionLocal() as session:
        media = session.get(m.MediaFile, media_id)
        if media is None:
            return "missing"
        try:
            media.status = m.MediaStatus.ANALYZING
            session.commit()
            # placeholder for analysis work
            media.status = m.MediaStatus.ANALYZED
            session.commit()
            return media.status.value
        except Exception:
            # The transaction may already be broken by whatever raised above,
            # so committing FAILED on it can itself raise and leave the row
            # stuck in ANALYZING. Roll back first, then re-fetch on a clean
            # transaction before recording the failure.
            session.rollback()
            media = session.get(m.MediaFile, media_id)
            if media is not None:
                media.status = m.MediaStatus.FAILED
                session.commit()
            raise
