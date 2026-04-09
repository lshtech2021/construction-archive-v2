import re
from dataclasses import dataclass

from app.schemas.ingestion import OcrWord

# Matches NCS sheet number patterns: A-101, S2.0, FP-01, A 101, etc.
_CALLOUT_PATTERN = re.compile(r"\b([A-Z]{1,2}[-\s]?\d{1,3}[A-Z]?)\b")
# Matches "4/A-401" style detail callouts
_DETAIL_CALLOUT_PATTERN = re.compile(r"\b(\d{1,2})/([A-Z]{1,2}[-\s]?\d{1,3}[A-Z]?)\b")


@dataclass
class CalloutEdge:
    source_sheet_id: str
    source_sheet_number: str
    target_sheet_number: str
    context_text: str
    detail_number: str | None
    # Bounding box from OCR polygon (min-rect)
    bbox_x: float | None
    bbox_y: float | None
    bbox_w: float | None
    bbox_h: float | None


def _normalize(sheet_ref: str) -> str:
    return sheet_ref.replace(" ", "").upper()


def _polygon_to_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float] | None:
    if not polygon:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return x, y, w, h


class CalloutExtractor:
    def extract_from_sheet(
        self,
        sheet_id: str,
        sheet_number: str,
        ocr_words: list[OcrWord],
        all_sheet_numbers: set[str],
    ) -> list[CalloutEdge]:
        """Scan OCR words for cross-references to other sheets."""
        edges: list[CalloutEdge] = []
        source_norm = _normalize(sheet_number)

        full_text = " ".join(w.content for w in ocr_words)

        # Find "4/A-401" style detail callouts first
        for match in _DETAIL_CALLOUT_PATTERN.finditer(full_text):
            detail_num = match.group(1)
            target_raw = match.group(2)
            target_norm = _normalize(target_raw)
            if target_norm == source_norm or target_norm not in all_sheet_numbers:
                continue
            # Try to find polygon of the matching word
            bbox = None
            for word in ocr_words:
                if target_raw.replace(" ", "") in word.content.replace(" ", ""):
                    bbox = _polygon_to_bbox(word.polygon)
                    break
            x, y, w, h = bbox if bbox else (None, None, None, None)
            edges.append(
                CalloutEdge(
                    source_sheet_id=sheet_id,
                    source_sheet_number=sheet_number,
                    target_sheet_number=target_norm,
                    context_text=match.group(0),
                    detail_number=detail_num,
                    bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                )
            )

        # General sheet references
        seen: set[str] = {e.target_sheet_number for e in edges}
        for match in _CALLOUT_PATTERN.finditer(full_text):
            target_norm = _normalize(match.group(1))
            if target_norm == source_norm or target_norm in seen:
                continue
            if target_norm not in all_sheet_numbers:
                continue
            seen.add(target_norm)
            bbox = None
            for word in ocr_words:
                if _normalize(word.content) == target_norm:
                    bbox = _polygon_to_bbox(word.polygon)
                    break
            x, y, w, h = bbox if bbox else (None, None, None, None)
            edges.append(
                CalloutEdge(
                    source_sheet_id=sheet_id,
                    source_sheet_number=sheet_number,
                    target_sheet_number=target_norm,
                    context_text=match.group(0),
                    detail_number=None,
                    bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                )
            )

        return edges
