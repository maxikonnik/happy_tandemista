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
            media.status = m.MediaStatus.FAILED
            session.commit()
            raise
