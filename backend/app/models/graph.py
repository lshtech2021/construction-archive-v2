import uuid

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class SheetHyperlink(Base):
    __tablename__ = "sheet_hyperlinks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_sheet_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_sheet_number: Mapped[str] = mapped_column(String(50), nullable=False)
    context_text: Mapped[str | None] = mapped_column(String(255))
    bbox_x: Mapped[float | None] = mapped_column(Float)
    bbox_y: Mapped[float | None] = mapped_column(Float)
    bbox_w: Mapped[float | None] = mapped_column(Float)
    bbox_h: Mapped[float | None] = mapped_column(Float)


class CalloutLink(Base):
    __tablename__ = "callout_links"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_sheet_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_sheet_number: Mapped[str] = mapped_column(String(50), nullable=False)
    target_detail_number: Mapped[str | None] = mapped_column(String(10))
    bbox_x: Mapped[float | None] = mapped_column(Float)
    bbox_y: Mapped[float | None] = mapped_column(Float)
    bbox_w: Mapped[float | None] = mapped_column(Float)
    bbox_h: Mapped[float | None] = mapped_column(Float)
