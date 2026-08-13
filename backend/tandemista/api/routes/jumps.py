from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import models as m
from ..deps import get_db
from ..schemas import (
    CustomerCreate, CustomerRead, JumpCreate, JumpRead, LoadCreate, LoadRead,
)

router = APIRouter(prefix="/dropzones/{dropzone_id}", tags=["manifest"])


def _require_dropzone(db: Session, dropzone_id: uuid.UUID) -> None:
    if db.get(m.Dropzone, dropzone_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dropzone not found")


@router.post("/loads", response_model=LoadRead, status_code=status.HTTP_201_CREATED)
def create_load(
    dropzone_id: uuid.UUID, payload: LoadCreate, db: Session = Depends(get_db)
) -> m.Load:
    _require_dropzone(db, dropzone_id)
    load = m.Load(dropzone_id=dropzone_id, name=payload.name, takeoff_at=payload.takeoff_at)
    db.add(load)
    db.flush()
    return load


@router.get("/loads", response_model=list[LoadRead])
def list_loads(dropzone_id: uuid.UUID, db: Session = Depends(get_db)) -> list[m.Load]:
    _require_dropzone(db, dropzone_id)
    return list(db.execute(select(m.Load).where(m.Load.dropzone_id == dropzone_id)).scalars())


@router.post("/customers", response_model=CustomerRead, status_code=status.HTTP_201_CREATED)
def create_customer(
    dropzone_id: uuid.UUID, payload: CustomerCreate, db: Session = Depends(get_db)
) -> m.Customer:
    _require_dropzone(db, dropzone_id)
    cust = m.Customer(dropzone_id=dropzone_id, name=payload.name, contact=payload.contact)
    db.add(cust)
    db.flush()
    return cust


@router.post("/jumps", response_model=JumpRead, status_code=status.HTTP_201_CREATED)
def create_jump(
    dropzone_id: uuid.UUID, payload: JumpCreate, db: Session = Depends(get_db)
) -> m.TandemJump:
    _require_dropzone(db, dropzone_id)
    jump = m.TandemJump(
        dropzone_id=dropzone_id, customer_id=payload.customer_id, load_id=payload.load_id,
        instructor_user_id=payload.instructor_user_id, outside_user_id=payload.outside_user_id,
        video_package=payload.video_package,
    )
    db.add(jump)
    db.flush()
    return jump


@router.get("/jumps", response_model=list[JumpRead])
def list_jumps(dropzone_id: uuid.UUID, db: Session = Depends(get_db)) -> list[m.TandemJump]:
    _require_dropzone(db, dropzone_id)
    return list(
        db.execute(select(m.TandemJump).where(m.TandemJump.dropzone_id == dropzone_id)).scalars()
    )
