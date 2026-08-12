# backend/tandemista/api/routes/media.py
from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ...db import models as m
from ...storage.base import StorageBackend
from ..deps import get_db, get_storage
from ..schemas import MediaRead

router = APIRouter(prefix="/dropzones/{dropzone_id}", tags=["media"])


@router.post("/media", response_model=MediaRead, status_code=status.HTTP_201_CREATED)
def upload_media(
    dropzone_id: uuid.UUID,
    file: UploadFile = File(...),
    jump_id: uuid.UUID | None = Form(default=None),
    device_id: uuid.UUID | None = Form(default=None),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
) -> m.MediaFile:
    if db.get(m.Dropzone, dropzone_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dropzone not found")

    filename = Path(file.filename).name if file.filename else "upload.bin"
    if filename in ("", ".", ".."):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid filename")

    media_id = uuid.uuid4()
    content = file.file.read()
    key = f"dropzones/{dropzone_id}/media/{media_id}/{filename}"
    stored = storage.put(key, io.BytesIO(content))

    media = m.MediaFile(
        id=media_id,
        dropzone_id=dropzone_id,
        jump_id=jump_id,
        device_id=device_id,
        filename=filename,
        locations=[stored.location],
        sha256=hashlib.sha256(content).hexdigest(),
        status=m.MediaStatus.REGISTERED,
    )
    db.add(media)
    db.flush()
    # Commit BEFORE enqueueing: a real worker can dequeue and look the row up
    # before this request's transaction would otherwise commit (in get_db,
    # after the response). Without this, the row may not exist yet and the
    # task silently no-ops (see analyze_media's `if media is None: return`).
    # The trailing get_db() commit becomes a harmless no-op after this.
    db.commit()

    from ...worker.tasks import analyze_media

    analyze_media.delay(str(media_id))
    return media


@router.get("/media/{media_id}", response_model=MediaRead)
def get_media(
    dropzone_id: uuid.UUID, media_id: uuid.UUID, db: Session = Depends(get_db)
) -> m.MediaFile:
    media = db.get(m.MediaFile, media_id)
    if media is None or media.dropzone_id != dropzone_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "media not found")
    return media
