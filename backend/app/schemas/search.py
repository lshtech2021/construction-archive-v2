from pydantic import BaseModel


class SearchResult(BaseModel):
    score: float
    sheet_id: str
    sheet_number: str
    sheet_title: str
    discipline: str
    dzi_path: str
    image_path: str
