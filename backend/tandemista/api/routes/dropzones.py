from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import models as m
from ..deps import get_db
from ..schemas import DeviceCreate, DeviceRead, DropzoneCreate, DropzoneRead

router = APIRouter(prefix="/dropzones", tags=["dropzones"])


def _get_dropzone(db: Session, dropzone_id: uuid.UUID) -> m.Dropzone:
    dz = db.get(m.Dropzone, dropzone_id)
    if dz is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dropzone not found")
    return dz


@router.post("/", response_model=DropzoneRead, status_code=status.HTTP_201_CREATED)
def create_dropzone(payload: DropzoneCreate, db: Session = Depends(get_db)) -> m.Dropzone:
    dz = m.Dropzone(name=payload.name, currency=payload.currency)
    db.add(dz)
    db.flush()
    return dz


@router.get("/", response_model=list[DropzoneRead])
def list_dropzones(db: Session = Depends(get_db)) -> list[m.Dropzone]:
    return list(db.execute(select(m.Dropzone)).scalars())


@router.post(
    "/{dropzone_id}/devices", response_model=DeviceRead, status_code=status.HTTP_201_CREATED
)
def create_device(
    dropzone_id: uuid.UUID, payload: DeviceCreate, db: Session = Depends(get_db)
) -> m.Device:
    _get_dropzone(db, dropzone_id)
    dev = m.Device(
        dropzone_id=dropzone_id, name=payload.name, kind=payload.kind,
        role=payload.role, clock_offset_seconds=payload.clock_offset_seconds,
    )
    db.add(dev)
    db.flush()
    return dev


@router.get("/{dropzone_id}/devices", response_model=list[DeviceRead])
def list_devices(dropzone_id: uuid.UUID, db: Session = Depends(get_db)) -> list[m.Device]:
    _get_dropzone(db, dropzone_id)
    return list(
        db.execute(select(m.Device).where(m.Device.dropzone_id == dropzone_id)).scalars()
    )
