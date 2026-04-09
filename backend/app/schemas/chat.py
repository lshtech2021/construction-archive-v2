from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    project_id: str
    discipline_filter: str | None = None
    chat_history: list[dict] = []


class TargetHighlight(BaseModel):
    """Sub-schema returned by GPT-4o structured output — identifies what to highlight."""
    exact_text: str          # Exact label visible on blueprint, e.g. "W12x50", "AHU-04"
    sheet_id: str            # Which sheet this text is on
    reasoning: str           # Why this element is relevant


class LLMVisionResponse(BaseModel):
    """GPT-4o structured output schema."""
    answer: str
    objects_to_highlight: list[TargetHighlight] = []


class ResolvedBoundingBox(BaseModel):
    """After resolving TargetHighlight against OCR database."""
    text: str
    sheet_id: str
    normalized_polygon: list[list[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


class Citation(BaseModel):
    sheet_number: str
    sheet_title: str
    dzi_path: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    highlights: list[ResolvedBoundingBox] = []
