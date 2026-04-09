import uuid

from sqlalchemy import BigInteger, Float, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class OcrWord(Base):
    __tablename__ = "ocr_words"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    sheet_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Polygon: 4 normalized [x,y] pairs
    poly_x1: Mapped[float | None] = mapped_column(Float)
    poly_y1: Mapped[float | None] = mapped_column(Float)
    poly_x2: Mapped[float | None] = mapped_column(Float)
    poly_y2: Mapped[float | None] = mapped_column(Float)
    poly_x3: Mapped[float | None] = mapped_column(Float)
    poly_y3: Mapped[float | None] = mapped_column(Float)
    poly_x4: Mapped[float | None] = mapped_column(Float)
    poly_y4: Mapped[float | None] = mapped_column(Float)

    sheet: Mapped["Sheet"] = relationship(back_populates="ocr_words")  # type: ignore[name-defined]
