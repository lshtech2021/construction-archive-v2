import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    sheets: Mapped[list["Sheet"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Sheet(Base):
    __tablename__ = "sheets"
    __table_args__ = (UniqueConstraint("project_id", "sheet_number"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    sheet_number: Mapped[str] = mapped_column(String(50), nullable=False)
    sheet_title: Mapped[str] = mapped_column(String(255), nullable=False)
    discipline: Mapped[str] = mapped_column(String(50), nullable=False)
    revision_date: Mapped[str | None] = mapped_column(String(20))
    scale: Mapped[str | None] = mapped_column(String(50))
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    dzi_path: Mapped[str] = mapped_column(Text, nullable=False)
    ocr_result_path: Mapped[str | None] = mapped_column(Text)
    qdrant_point_id: Mapped[int | None] = mapped_column()
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    project: Mapped["Project"] = relationship(back_populates="sheets")
    ocr_words: Mapped[list["OcrWord"]] = relationship(back_populates="sheet", cascade="all, delete-orphan")
