import uuid

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ocr import OcrWord


class SpatialResolver:
    async def find_polygon(
        self, sheet_id: str, target_text: str, db: AsyncSession
    ) -> list[list[float]] | None:
        """Find the OCR polygon for target_text on a given sheet.

        First tries exact ILIKE match, falls back to trigram similarity.
        """
        # Exact / partial match
        result = await db.execute(
            select(OcrWord)
            .where(
                OcrWord.sheet_id == uuid.UUID(sheet_id),
                OcrWord.content.ilike(f"%{target_text}%"),
            )
            .limit(1)
        )
        word = result.scalar_one_or_none()

        if word is None:
            # pg_trgm fuzzy fallback - find most similar word
            result = await db.execute(
                select(OcrWord)
                .where(OcrWord.sheet_id == uuid.UUID(sheet_id))
                .order_by(
                    func.similarity(OcrWord.content, target_text).desc()
                )
                .limit(1)
            )
            word = result.scalar_one_or_none()

        if word is None:
            return None

        return [
            [word.poly_x1 or 0.0, word.poly_y1 or 0.0],
            [word.poly_x2 or 0.0, word.poly_y2 or 0.0],
            [word.poly_x3 or 0.0, word.poly_y3 or 0.0],
            [word.poly_x4 or 0.0, word.poly_y4 or 0.0],
        ]
