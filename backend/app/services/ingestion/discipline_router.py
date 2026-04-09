import base64
import re
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel

from app.config import settings

# AIA / NCS standard prefixes
_NCS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^A[-\s]?\d", re.IGNORECASE), "Architectural"),
    (re.compile(r"^S[-\s]?\d", re.IGNORECASE), "Structural"),
    (re.compile(r"^M[-\s]?\d", re.IGNORECASE), "Mechanical"),
    (re.compile(r"^E[-\s]?\d", re.IGNORECASE), "Electrical"),
    (re.compile(r"^P[-\s]?\d", re.IGNORECASE), "Plumbing"),
    (re.compile(r"^C[-\s]?\d", re.IGNORECASE), "Civil"),
    (re.compile(r"^L[-\s]?\d", re.IGNORECASE), "Landscape"),
    (re.compile(r"^FP[-\s]?\d", re.IGNORECASE), "Fire Protection"),
    (re.compile(r"^I[-\s]?\d", re.IGNORECASE), "Interior"),
]


class _SemanticClassification(BaseModel):
    discipline: str
    confidence: float
    reasoning: str


class DisciplineRouter:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)

    def route_by_regex(self, sheet_number: str) -> str | None:
        for pattern, discipline in _NCS_PATTERNS:
            if pattern.match(sheet_number.strip()):
                return discipline
        return None

    def route_semantically(self, sheet_title: str, image_path: str) -> str:
        with open(image_path, "rb") as f:
            raw = f.read()

        # Downscale for cost — encode as-is at detail=low
        image_b64 = base64.standard_b64encode(raw).decode()

        response = self._client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Classify this construction blueprint sheet titled '{sheet_title}' "
                                "into one discipline: Architectural, Structural, Mechanical, Electrical, "
                                "Plumbing, Civil, Landscape, Fire Protection, Interior, or General. "
                                "Look for visual cues: rebar/concrete→Structural, ducts/HVAC→Mechanical, "
                                "toilets/pipes→Plumbing, walls/rooms→Architectural, conduit/panels→Electrical. "
                                "Return discipline, confidence (0-1), and brief reasoning."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            response_format=_SemanticClassification,
            temperature=0.0,
        )
        result = response.choices[0].message.parsed
        if result and result.confidence >= 0.7:
            return result.discipline
        return "General"

    def route(self, sheet_number: str, sheet_title: str, image_path: str) -> str:
        discipline = self.route_by_regex(sheet_number)
        if discipline:
            return discipline
        return self.route_semantically(sheet_title, image_path)
