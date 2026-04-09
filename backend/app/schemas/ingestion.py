import uuid
from pydantic import BaseModel


class SheetMetadata(BaseModel):
    """Structured output target for GPT-4o vision extraction."""
    sheet_number: str
    sheet_title: str
    revision_date: str = ""
    scale: str = ""


class PageImage(BaseModel):
    page_index: int
    image_path: str


class OcrWord(BaseModel):
    content: str
    polygon: list[list[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] normalized 0-1


class OcrResult(BaseModel):
    words: list[OcrWord]
    raw_json_path: str = ""


class ProcessedSheet(BaseModel):
    page_index: int
    image_path: str
    dzi_path: str
    metadata: SheetMetadata
    discipline: str
    ocr_result: OcrResult | None = None
    sheet_db_id: uuid.UUID | None = None


class IngestionManifest(BaseModel):
    project_id: str
    pdf_filename: str
    sheets: list[ProcessedSheet]
    status: str = "complete"
