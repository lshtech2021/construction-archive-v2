import uuid

import networkx as nx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.graph import CalloutLink, SheetHyperlink
from app.models.ocr import OcrWord
from app.models.project import Sheet
from app.schemas.ingestion import OcrWord as OcrWordSchema
from app.services.graph.callout_extractor import CalloutEdge, CalloutExtractor


class GraphManager:
    def __init__(self) -> None:
        self._extractor = CalloutExtractor()

    async def build_project_graph(
        self, project_id: str, db: AsyncSession
    ) -> nx.DiGraph:
        """Build an in-memory callout graph for all sheets in a project."""
        # Load all sheets
        result = await db.execute(
            select(Sheet).where(Sheet.project_id == uuid.UUID(project_id))
        )
        sheets = result.scalars().all()
        all_sheet_numbers = {s.sheet_number.replace(" ", "").upper() for s in sheets}

        graph = nx.DiGraph()
        for s in sheets:
            graph.add_node(s.sheet_number)

        for sheet in sheets:
            # Load OCR words for this sheet
            ocr_result = await db.execute(
                select(OcrWord).where(OcrWord.sheet_id == sheet.id)
            )
            db_words = ocr_result.scalars().all()
            ocr_words = [
                OcrWordSchema(
                    content=w.content,
                    polygon=[
                        [w.poly_x1 or 0.0, w.poly_y1 or 0.0],
                        [w.poly_x2 or 0.0, w.poly_y2 or 0.0],
                        [w.poly_x3 or 0.0, w.poly_y3 or 0.0],
                        [w.poly_x4 or 0.0, w.poly_y4 or 0.0],
                    ],
                )
                for w in db_words
            ]

            edges: list[CalloutEdge] = self._extractor.extract_from_sheet(
                sheet_id=str(sheet.id),
                sheet_number=sheet.sheet_number,
                ocr_words=ocr_words,
                all_sheet_numbers=all_sheet_numbers,
            )

            for edge in edges:
                graph.add_edge(
                    edge.source_sheet_number,
                    edge.target_sheet_number,
                    context=edge.context_text,
                    bbox=(edge.bbox_x, edge.bbox_y, edge.bbox_w, edge.bbox_h),
                    detail_number=edge.detail_number,
                    source_sheet_id=edge.source_sheet_id,
                )

        return graph

    async def persist_graph(
        self, graph: nx.DiGraph, project_id: str, db: AsyncSession
    ) -> None:
        """Bulk insert graph edges into PostgreSQL."""
        # Delete existing links for this project first
        from sqlalchemy import delete
        await db.execute(
            delete(SheetHyperlink).where(
                SheetHyperlink.project_id == uuid.UUID(project_id)
            )
        )

        for source, target, data in graph.edges(data=True):
            bbox_x, bbox_y, bbox_w, bbox_h = data.get("bbox", (None, None, None, None))
            source_id = data.get("source_sheet_id")
            if not source_id:
                continue

            db.add(
                SheetHyperlink(
                    project_id=uuid.UUID(project_id),
                    source_sheet_id=uuid.UUID(source_id),
                    target_sheet_number=target,
                    context_text=data.get("context", "")[:255],
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                )
            )

            # Also persist as CalloutLink (with detail number)
            db.add(
                CalloutLink(
                    source_sheet_id=uuid.UUID(source_id),
                    target_sheet_number=target,
                    target_detail_number=data.get("detail_number"),
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                )
            )

        await db.commit()

    async def get_linked_sheets(
        self, sheet_id: str, db: AsyncSession
    ) -> list[dict]:
        """Return callout links originating from the given sheet."""
        result = await db.execute(
            select(CalloutLink).where(
                CalloutLink.source_sheet_id == uuid.UUID(sheet_id)
            )
        )
        links = result.scalars().all()
        return [
            {
                "target_sheet": link.target_sheet_number,
                "target_detail_number": link.target_detail_number,
                "bounding_box": [link.bbox_x, link.bbox_y, link.bbox_w, link.bbox_h],
            }
            for link in links
        ]
