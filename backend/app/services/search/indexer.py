import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.qdrant import get_qdrant_client
from app.models.ocr import OcrWord
from app.models.project import Sheet
from app.services.search.engine import SemanticSearchEngine


class SheetIndexer:
    def __init__(self) -> None:
        self._engine = SemanticSearchEngine(get_qdrant_client())

    async def index_all_sheets(self, project_id: str, db: AsyncSession) -> int:
        """Index all sheets for a project into Qdrant. Returns count indexed."""
        await self._engine.setup_collection()

        result = await db.execute(
            select(Sheet).where(Sheet.project_id == uuid.UUID(project_id))
        )
        sheets = result.scalars().all()

        for sheet in sheets:
            # Gather OCR words for searchable text
            ocr_result = await db.execute(
                select(OcrWord).where(OcrWord.sheet_id == sheet.id)
            )
            words = ocr_result.scalars().all()
            ocr_text = " ".join(
                f"{sheet.sheet_number} {sheet.sheet_title} {sheet.discipline} "
                + " ".join(w.content for w in words)
            )

            await self._engine.index_sheet(
                sheet_id=str(sheet.id),
                image_path=sheet.image_path,
                ocr_text=ocr_text,
                discipline=sheet.discipline,
                project_id=project_id,
            )

            # Update qdrant_point_id on the sheet record
            point_id = uuid.UUID(str(sheet.id)).int % (2**63)
            sheet.qdrant_point_id = point_id

        await db.commit()
        return len(sheets)
