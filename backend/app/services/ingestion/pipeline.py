import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.ocr import OcrWord as OcrWordModel
from app.models.project import Sheet
from app.schemas.ingestion import (
    IngestionManifest,
    OcrResult,
    PageImage,
    ProcessedSheet,
    SheetMetadata,
)
from app.services.ingestion.discipline_router import DisciplineRouter
from app.services.ingestion.dzi_generator import DZIGenerator
from app.services.ingestion.metadata_extractor import MetadataExtractor
from app.services.ingestion.ocr_client import AzureOCRClient
from app.services.ingestion.pdf_splitter import PDFSplitter


class ConstructionIngestionPipeline:
    def __init__(self) -> None:
        self._splitter = PDFSplitter()
        self._extractor = MetadataExtractor()
        self._router = DisciplineRouter()
        self._dzi = DZIGenerator()
        self._ocr = AzureOCRClient()

    def _process_page(
        self, page: PageImage, project_dir: str, project_id: str
    ) -> ProcessedSheet:
        """Process a single blueprint page: extract metadata, route discipline, generate DZI, run OCR."""
        dzi_dir = os.path.join(project_dir, "dzi")
        ocr_dir = os.path.join(project_dir, "ocr")

        # Extract title block metadata
        metadata: SheetMetadata = self._extractor.extract(page.image_path)

        # Route to discipline
        discipline = self._router.route(
            metadata.sheet_number, metadata.sheet_title, page.image_path
        )

        # Generate DZI pyramid
        output_prefix = f"page_{page.page_index:03d}"
        dzi_path = self._dzi.generate(page.image_path, output_prefix, dzi_dir)

        # Run Azure OCR
        ocr_json_path = os.path.join(ocr_dir, f"page_{page.page_index:03d}.json")
        ocr_result: OcrResult = self._ocr.analyze(page.image_path, ocr_json_path)

        return ProcessedSheet(
            page_index=page.page_index,
            image_path=page.image_path,
            dzi_path=dzi_path,
            metadata=metadata,
            discipline=discipline,
            ocr_result=ocr_result,
        )

    async def process_pdf(
        self, pdf_path: str, project_id: str, db: AsyncSession
    ) -> IngestionManifest:
        """Full pipeline: split PDF → process pages concurrently → persist to PostgreSQL."""
        pdf_filename = Path(pdf_path).name
        project_dir = os.path.join(settings.local_storage_path, project_id)
        images_dir = os.path.join(project_dir, "images")

        # Split PDF into page images
        pages: list[PageImage] = self._splitter.split_to_images(
            pdf_path, images_dir, dpi=300
        )

        # Process all pages concurrently (network-bound: GPT-4o + Azure DI)
        processed_sheets: list[ProcessedSheet] = [None] * len(pages)  # type: ignore
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._process_page, page, project_dir, project_id): page
                for page in pages
            }
            for future in as_completed(futures):
                sheet = future.result()
                processed_sheets[sheet.page_index] = sheet

        # Persist to PostgreSQL
        for sheet in processed_sheets:
            db_sheet = Sheet(
                project_id=uuid.UUID(project_id),
                sheet_number=sheet.metadata.sheet_number,
                sheet_title=sheet.metadata.sheet_title,
                discipline=sheet.discipline,
                revision_date=sheet.metadata.revision_date or None,
                scale=sheet.metadata.scale or None,
                image_path=sheet.image_path,
                dzi_path=sheet.dzi_path,
                ocr_result_path=(
                    sheet.ocr_result.raw_json_path if sheet.ocr_result else None
                ),
            )
            db.add(db_sheet)
            await db.flush()  # get db_sheet.id

            sheet.sheet_db_id = db_sheet.id

            # Persist OCR words
            if sheet.ocr_result:
                for word in sheet.ocr_result.words:
                    p = word.polygon
                    db.add(
                        OcrWordModel(
                            sheet_id=db_sheet.id,
                            content=word.content,
                            poly_x1=p[0][0] if len(p) > 0 else None,
                            poly_y1=p[0][1] if len(p) > 0 else None,
                            poly_x2=p[1][0] if len(p) > 1 else None,
                            poly_y2=p[1][1] if len(p) > 1 else None,
                            poly_x3=p[2][0] if len(p) > 2 else None,
                            poly_y3=p[2][1] if len(p) > 2 else None,
                            poly_x4=p[3][0] if len(p) > 3 else None,
                            poly_y4=p[3][1] if len(p) > 3 else None,
                        )
                    )

        await db.commit()

        return IngestionManifest(
            project_id=project_id,
            pdf_filename=pdf_filename,
            sheets=processed_sheets,
            status="complete",
        )
