from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ..db.models import DeviceKind, DeviceRole


class _Read(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DropzoneCreate(BaseModel):
    name: str
    currency: str = "USD"


class DropzoneRead(_Read):
    id: uuid.UUID
    name: str
    currency: str
    enabled_variants: list[str]


class DeviceCreate(BaseModel):
    name: str
    kind: DeviceKind
    role: DeviceRole
    clock_offset_seconds: float = 0.0


class DeviceRead(_Read):
    id: uuid.UUID
    dropzone_id: uuid.UUID
    name: str
    kind: DeviceKind
    role: DeviceRole
    active: bool


class LoadCreate(BaseModel):
    name: str
    takeoff_at: datetime | None = None


class LoadRead(_Read):
    id: uuid.UUID
    name: str
    takeoff_at: datetime | None


class CustomerCreate(BaseModel):
    name: str
    contact: str | None = None


class CustomerRead(_Read):
    id: uuid.UUID
    name: str
    contact: str | None


class JumpCreate(BaseModel):
    customer_id: uuid.UUID | None = None
    load_id: uuid.UUID | None = None
    instructor_user_id: uuid.UUID | None = None
    outside_user_id: uuid.UUID | None = None
    video_package: bool = False


class JumpRead(_Read):
    id: uuid.UUID
    dropzone_id: uuid.UUID
    customer_id: uuid.UUID | None
    load_id: uuid.UUID | None
    video_package: bool


class MediaRead(_Read):
    id: uuid.UUID
    dropzone_id: uuid.UUID
    jump_id: uuid.UUID | None
    filename: str
    status: str
    locations: list
