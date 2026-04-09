"""Phase 2.1 unit tests — callout extractor and graph normalization."""
import pytest

from app.schemas.ingestion import OcrWord
from app.services.graph.callout_extractor import CalloutExtractor, _normalize


def make_word(content: str) -> OcrWord:
    return OcrWord(
        content=content,
        polygon=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
    )


def test_callout_extractor_basic():
    extractor = CalloutExtractor()
    words = [make_word("SEE"), make_word("DETAIL"), make_word("4/S-201"), make_word("FOR"), make_word("ELEV."), make_word("A-102")]
    edges = extractor.extract_from_sheet(
        sheet_id="sheet-1",
        sheet_number="A-101",
        ocr_words=words,
        all_sheet_numbers={"S-201", "A-102"},
    )
    targets = {e.target_sheet_number for e in edges}
    assert "S-201" in targets
    assert "A-102" in targets


def test_callout_extractor_no_self_reference():
    extractor = CalloutExtractor()
    words = [make_word("REFER"), make_word("TO"), make_word("A-101")]
    edges = extractor.extract_from_sheet(
        sheet_id="sheet-1",
        sheet_number="A-101",
        ocr_words=words,
        all_sheet_numbers={"A-101", "S-201"},
    )
    targets = {e.target_sheet_number for e in edges}
    assert "A-101" not in targets


def test_callout_normalization():
    assert _normalize("A - 101") == "A-101"
    assert _normalize("a-101") == "A-101"
    assert _normalize("S 2.0") == "S2.0"


def test_callout_extractor_filters_unknown_sheets():
    extractor = CalloutExtractor()
    words = [make_word("SEE"), make_word("X-999")]
    edges = extractor.extract_from_sheet(
        sheet_id="sheet-1",
        sheet_number="A-101",
        ocr_words=words,
        all_sheet_numbers={"A-101"},
    )
    assert len(edges) == 0


def test_detail_callout_extracts_detail_number():
    extractor = CalloutExtractor()
    words = [make_word("4/A-401"), make_word("SEE")]
    edges = extractor.extract_from_sheet(
        sheet_id="sheet-1",
        sheet_number="A-101",
        ocr_words=words,
        all_sheet_numbers={"A-401"},
    )
    assert len(edges) >= 1
    detail_edges = [e for e in edges if e.detail_number == "4"]
    assert len(detail_edges) == 1
