import asyncio

from celery import Celery

from app.config import settings

celery_app = Celery("construction_archive", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(bind=True, name="ingest_pdf")
def ingest_pdf_task(self, pdf_path: str, project_id: str) -> dict:
    """Background task: run the full ingestion pipeline for a PDF."""
    from app.db.session import AsyncSessionLocal
    from app.services.ingestion.pipeline import ConstructionIngestionPipeline
    from app.services.graph.graph_manager import GraphManager
    from app.services.search.indexer import SheetIndexer

    async def _run() -> dict:
        pipeline = ConstructionIngestionPipeline()
        async with AsyncSessionLocal() as db:
            manifest = await pipeline.process_pdf(pdf_path, project_id, db)

        # Build callout graph
        async with AsyncSessionLocal() as db:
            graph_manager = GraphManager()
            graph = await graph_manager.build_project_graph(project_id, db)
            await graph_manager.persist_graph(graph, project_id, db)

        # Index all sheets into Qdrant
        indexer = SheetIndexer()
        async with AsyncSessionLocal() as db:
            await indexer.index_all_sheets(project_id, db)

        return {"status": "complete", "sheet_count": len(manifest.sheets)}

    return asyncio.run(_run())
