# backend/tandemista/db/models.py
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .types import GUID, JSONColumn


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeviceKind(str, enum.Enum):
    GOPRO = "gopro"
    INSTA360 = "insta360"
    DSLR = "dslr"
    PHONE = "phone"
    OTHER = "other"


class DeviceRole(str, enum.Enum):
    HANDCAM = "handcam"
    OUTSIDE = "outside"
    GROUND_INTERVIEW = "ground_interview"
    GROUND_LANDING = "ground_landing"
    MIXED = "mixed"


class UploadSourceKind(str, enum.Enum):
    UPLOADER_AGENT = "uploader_agent"
    CLOUD_FOLDER = "cloud_folder"
    CAMERA_WIFI = "camera_wifi"
    MANUAL_WEB = "manual_web"


class MediaStatus(str, enum.Enum):
    REGISTERED = "registered"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    NEEDS_ATTENTION = "needs_attention"
    FAILED = "failed"


class CutVariant(str, enum.Enum):
    FULL_16X9 = "full_16x9"
    EMOTIONS_16X9 = "emotions_16x9"
    EMOTIONS_9X16 = "emotions_9x16"
    HIGHLIGHTS_9X16 = "highlights_9x16"


class CutStatus(str, enum.Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    FAILED = "failed"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(GUID, primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Dropzone(TimestampMixin, Base):
    __tablename__ = "dropzones"
    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String, nullable=False)
    currency: Mapped[str] = mapped_column(String, default="USD", nullable=False)
    payment_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    storage_config: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    enabled_variants: Mapped[list] = mapped_column(
        JSONColumn, default=lambda: [v.value for v in CutVariant], nullable=False
    )
    devices: Mapped[list["Device"]] = relationship(back_populates="dropzone")


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str | None] = mapped_column(String, nullable=True)


class Customer(TimestampMixin, Base):
    __tablename__ = "customers"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    contact: Mapped[str | None] = mapped_column(String, nullable=True)


class Load(TimestampMixin, Base):
    __tablename__ = "loads"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    takeoff_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UploadSource(TimestampMixin, Base):
    __tablename__ = "upload_sources"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    kind: Mapped[UploadSourceKind] = mapped_column(String, nullable=False)
    config: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)


class Device(TimestampMixin, Base):
    __tablename__ = "devices"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[DeviceKind] = mapped_column(String, nullable=False)
    role: Mapped[DeviceRole] = mapped_column(String, nullable=False)
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    clock_offset_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fingerprints: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    upload_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("upload_sources.id"), nullable=True
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    dropzone: Mapped["Dropzone"] = relationship(back_populates="devices")


class TandemJump(TimestampMixin, Base):
    __tablename__ = "tandem_jumps"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    customer_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("customers.id"), nullable=True)
    load_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("loads.id"), nullable=True)
    instructor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    outside_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    video_package: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cuts: Mapped[list["JumpCut"]] = relationship(back_populates="jump")


class MediaFile(TimestampMixin, Base):
    __tablename__ = "media_files"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    jump_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tandem_jumps.id"), nullable=True)
    device_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("devices.id"), nullable=True)
    upload_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("upload_sources.id"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    locations: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[MediaStatus] = mapped_column(
        String, default=MediaStatus.REGISTERED, nullable=False
    )
    telemetry: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)


class JumpCut(TimestampMixin, Base):
    __tablename__ = "jump_cuts"
    __table_args__ = (UniqueConstraint("jump_id", "variant", name="uq_jumpcut_jump_variant"),)
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    jump_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tandem_jumps.id"), nullable=False)
    variant: Mapped[CutVariant] = mapped_column(String, nullable=False)
    status: Mapped[CutStatus] = mapped_column(String, default=CutStatus.DRAFT, nullable=False)
    edl: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    proxy_location: Mapped[str | None] = mapped_column(String, nullable=True)
    render_location: Mapped[str | None] = mapped_column(String, nullable=True)
    jump: Mapped["TandemJump"] = relationship(back_populates="cuts")


class Order(TimestampMixin, Base):
    __tablename__ = "orders"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    jump_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tandem_jumps.id"), nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[OrderStatus] = mapped_column(String, default=OrderStatus.PENDING, nullable=False)
    external_id: Mapped[str | None] = mapped_column(String, nullable=True)
