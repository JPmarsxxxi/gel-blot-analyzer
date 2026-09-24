import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Float, Boolean, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_token() -> str:
    # Unguessable project access token (URL-embedded, replaces auth for V1).
    return uuid.uuid4().hex + uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_token)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    images: Mapped[list["GelImage"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="GelImage.order_index"
    )


class GelImage(Base):
    __tablename__ = "gel_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    orig_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255))  # processed/current image on disk
    source_filename: Mapped[str] = mapped_column(String(255))  # untouched original on disk

    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    # Image adjustment state (applied to produce stored_filename from source_filename)
    crop_x: Mapped[float] = mapped_column(Float, default=0.0)
    crop_y: Mapped[float] = mapped_column(Float, default=0.0)
    crop_w: Mapped[float | None] = mapped_column(Float, nullable=True)
    crop_h: Mapped[float | None] = mapped_column(Float, nullable=True)
    rotate_deg: Mapped[float] = mapped_column(Float, default=0.0)
    brightness: Mapped[float] = mapped_column(Float, default=0.0)  # additive, -100..100
    contrast: Mapped[float] = mapped_column(Float, default=1.0)  # multiplicative, 0.1..3.0

    sensitivity: Mapped[float] = mapped_column(Float, default=0.5)  # 0..1 detection threshold slider

    is_gel_like: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    project: Mapped["Project"] = relationship(back_populates="images")
    lanes: Mapped[list["Lane"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", order_by="Lane.index"
    )


class Lane(Base):
    __tablename__ = "lanes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gel_image_id: Mapped[int] = mapped_column(ForeignKey("gel_images.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)  # left-to-right order

    x_start: Mapped[float] = mapped_column(Float)  # pixel x in image space
    x_end: Mapped[float] = mapped_column(Float)

    label: Mapped[str] = mapped_column(String(255), default="")
    is_ladder: Mapped[bool] = mapped_column(Boolean, default=False)

    image: Mapped["GelImage"] = relationship(back_populates="lanes")
    bands: Mapped[list["Band"]] = relationship(
        back_populates="lane", cascade="all, delete-orphan", order_by="Band.y"
    )


class Band(Base):
    __tablename__ = "bands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lane_id: Mapped[int] = mapped_column(ForeignKey("lanes.id"), index=True)

    # Box in image pixel space
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)

    intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    percent_of_lane: Mapped[float | None] = mapped_column(Float, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)

    # Ladder calibration: known size entered by user (only meaningful when lane.is_ladder)
    known_kda: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Estimated size from calibration curve (only meaningful for non-ladder lanes)
    estimated_kda: Mapped[float | None] = mapped_column(Float, nullable=True)

    manually_edited: Mapped[bool] = mapped_column(Boolean, default=False)

    lane: Mapped["Lane"] = relationship(back_populates="bands")
