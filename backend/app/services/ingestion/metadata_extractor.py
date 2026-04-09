import base64
import time

from openai import OpenAI, RateLimitError

from app.config import settings
from app.schemas.ingestion import SheetMetadata


class MetadataExtractor:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)

    def extract(self, image_path: str) -> SheetMetadata:
        """Extract title block metadata from a blueprint image using GPT-4o vision."""
        with open(image_path, "rb") as f:
            image_b64 = base64.standard_b64encode(f.read()).decode()

        for attempt in range(3):
            try:
                response = self._client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "This is a construction blueprint sheet. "
                                        "Extract the title block information: "
                                        "sheet_number (e.g. A-101, S2.0, MEP-01), "
                                        "sheet_title (e.g. 'First Floor Plan'), "
                                        "revision_date (YYYY-MM-DD if found, else empty), "
                                        "scale (e.g. '1/8\"=1\\'-0\"', else empty). "
                                        "Return only what is explicitly visible."
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
                    response_format=SheetMetadata,
                    temperature=0.0,
                )
                return response.choices[0].message.parsed  # type: ignore[return-value]
            except RateLimitError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError("Failed to extract metadata after retries")
