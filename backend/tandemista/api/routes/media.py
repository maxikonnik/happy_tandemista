# backend/tandemista/api/routes/media.py
from __future__ import annotations

import hashlib
import io
import uuid

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

    media_id = uuid.uuid4()
    content = file.file.read()
    key = f"dropzones/{dropzone_id}/media/{media_id}/{file.filename}"
    stored = storage.put(key, io.BytesIO(content))

    media = m.MediaFile(
        id=media_id,
        dropzone_id=dropzone_id,
        jump_id=jump_id,
        device_id=device_id,
        filename=file.filename or "upload.bin",
        locations=[stored.location],
        sha256=hashlib.sha256(content).hexdigest(),
        status=m.MediaStatus.REGISTERED,
    )
    db.add(media)
    db.flush()

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
