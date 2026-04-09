import base64
import uuid

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.qdrant import get_qdrant_client
from app.models.project import Sheet
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Citation,
    LLMVisionResponse,
    ResolvedBoundingBox,
)
from app.services.chat.spatial_resolver import SpatialResolver
from app.services.search.engine import SemanticSearchEngine

_SYSTEM_PROMPT = """You are an expert architect and structural engineer reviewing construction blueprints.

Rules:
1. Answer ONLY from the blueprint images provided. Do not use outside knowledge.
2. Explicitly cite the sheet number (e.g., "Sheet A-101") for every claim.
3. Never guess or hallucinate dimensions, specifications, or locations not clearly visible.
4. For each key element you identify, include its EXACT text label as it appears on the blueprint
   in the objects_to_highlight list — this drives spatial highlighting in the UI.
5. Be precise and professional."""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


class RAGOrchestrator:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._search = SemanticSearchEngine(get_qdrant_client())
        self._resolver = SpatialResolver()

    async def answer(self, request: ChatRequest, db: AsyncSession) -> ChatResponse:
        # Step 1: Retrieve top matching sheets
        results = await self._search.search(
            query=request.query,
            project_id=request.project_id,
            discipline_filter=request.discipline_filter,
            top_k=2,
        )

        if not results:
            return ChatResponse(
                answer="No relevant blueprint sheets found for this query.",
                citations=[],
                highlights=[],
            )

        # Step 2: Load sheet DB records for citation metadata
        sheet_map: dict[str, Sheet] = {}
        for r in results:
            res = await db.execute(
                select(Sheet).where(Sheet.id == uuid.UUID(r.sheet_id))
            )
            sheet = res.scalar_one_or_none()
            if sheet:
                sheet_map[r.sheet_id] = sheet

        # Step 3: Build multimodal GPT-4o message
        content: list[dict] = [{"type": "text", "text": request.query}]
        for r in results:
            sheet = sheet_map.get(r.sheet_id)
            label = f"Sheet {sheet.sheet_number}: {sheet.sheet_title}" if sheet else r.sheet_id
            image_b64 = _encode_image(r.image_path)
            content.append({"type": "text", "text": f"\n[{label}]"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                },
            })

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *request.chat_history,
            {"role": "user", "content": content},
        ]

        # Step 4: Call GPT-4o with structured output
        response = self._client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=LLMVisionResponse,
            temperature=0.1,
        )
        llm_response: LLMVisionResponse = response.choices[0].message.parsed  # type: ignore

        # Step 5: Resolve text labels to OCR polygons
        highlights: list[ResolvedBoundingBox] = []
        for target in llm_response.objects_to_highlight:
            polygon = await self._resolver.find_polygon(
                sheet_id=target.sheet_id,
                target_text=target.exact_text,
                db=db,
            )
            if polygon:
                highlights.append(
                    ResolvedBoundingBox(
                        text=target.exact_text,
                        sheet_id=target.sheet_id,
                        normalized_polygon=polygon,
                    )
                )

        citations: list[Citation] = [
            Citation(
                sheet_number=sheet_map[r.sheet_id].sheet_number,
                sheet_title=sheet_map[r.sheet_id].sheet_title,
                dzi_path=sheet_map[r.sheet_id].dzi_path,
            )
            for r in results
            if r.sheet_id in sheet_map
        ]

        return ChatResponse(
            answer=llm_response.answer,
            citations=citations,
            highlights=highlights,
        )
