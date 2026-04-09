"""Phase 1 unit tests — PDF splitter, discipline router, DZI generator."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.ingestion import SheetMetadata
from app.services.ingestion.discipline_router import DisciplineRouter
from app.services.ingestion.pdf_splitter import PDFSplitter


# --- PDFSplitter ---

def test_pdf_splitter_page_count(tmp_path):
    """Each PDF page should produce one JPEG."""
    import fitz

    # Create a 3-page PDF in memory
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page(width=612, height=792)
    pdf_path = str(tmp_path / "test.pdf")
    doc.save(pdf_path)
    doc.close()

    splitter = PDFSplitter()
    images = splitter.split_to_images(pdf_path, str(tmp_path / "images"), dpi=72)

    assert len(images) == 3
    for img in images:
        assert os.path.exists(img.image_path)
        assert os.path.getsize(img.image_path) > 0


# --- DisciplineRouter ---

@pytest.mark.parametrize("sheet_number,expected", [
    ("A-101", "Architectural"),
    ("S2.0", "Structural"),
    ("M-301", "Mechanical"),
    ("E-201", "Electrical"),
    ("P-101", "Plumbing"),
    ("FP-01", "Fire Protection"),
    ("C-100", "Civil"),
    ("L-100", "Landscape"),
    ("001", None),
    ("GENERAL", None),
])
def test_discipline_router_regex(sheet_number, expected):
    router = DisciplineRouter.__new__(DisciplineRouter)  # skip __init__ (no API key needed)
    result = router.route_by_regex(sheet_number)
    assert result == expected


@patch("app.services.ingestion.discipline_router.OpenAI")
def test_discipline_router_semantic_fallback(mock_openai_cls):
    """When regex returns None, semantic fallback is called and respects confidence threshold."""
    from pydantic import BaseModel

    class FakeClassification(BaseModel):
        discipline: str = "Structural"
        confidence: float = 0.85
        reasoning: str = "rebar visible"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = FakeClassification()
    mock_client.beta.chat.completions.parse.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    router = DisciplineRouter()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # Write minimal JPEG bytes
        f.write(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xd9"
        )
        img_path = f.name

    result = router.route_semantically("Unknown Sheet", img_path)
    os.unlink(img_path)
    assert result == "Structural"


@patch("app.services.ingestion.discipline_router.OpenAI")
def test_discipline_router_low_confidence_defaults_general(mock_openai_cls):
    from pydantic import BaseModel

    class FakeClassification(BaseModel):
        discipline: str = "Mechanical"
        confidence: float = 0.4
        reasoning: str = "unclear"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = FakeClassification()
    mock_client.beta.chat.completions.parse.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    router = DisciplineRouter()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
        img_path = f.name

    result = router.route_semantically("Mystery Sheet", img_path)
    os.unlink(img_path)
    assert result == "General"
