import json
import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from app.config import settings
from app.schemas.ingestion import OcrResult, OcrWord


class AzureOCRClient:
    def __init__(self) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=settings.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(settings.azure_document_intelligence_key),
        )

    def analyze(self, image_path: str, output_json_path: str = "") -> OcrResult:
        """Run Azure Document Intelligence prebuilt-layout on an image.

        Returns structured OCR result with normalized polygon coordinates.
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        poller = self._client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=image_bytes,
            content_type="image/jpeg",
        )
        result = poller.result()

        words: list[OcrWord] = []
        if result.pages:
            for page in result.pages:
                page_width = page.width or 1.0
                page_height = page.height or 1.0
                if page.words:
                    for word in page.words:
                        if word.polygon and len(word.polygon) >= 8:
                            # Azure returns flat list [x1,y1,x2,y2,...] in points
                            pts = word.polygon
                            polygon = [
                                [pts[i] / page_width, pts[i + 1] / page_height]
                                for i in range(0, len(pts), 2)
                            ]
                            words.append(OcrWord(content=word.content, polygon=polygon))

        raw_json_path = ""
        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            raw = result.as_dict() if hasattr(result, "as_dict") else {}
            with open(output_json_path, "w") as f:
                json.dump(raw, f)
            raw_json_path = output_json_path

        return OcrResult(words=words, raw_json_path=raw_json_path)
