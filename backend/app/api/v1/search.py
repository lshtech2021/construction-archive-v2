from fastapi import APIRouter, Query

from app.db.qdrant import get_qdrant_client
from app.schemas.search import SearchResult
from app.services.search.engine import SemanticSearchEngine

router = APIRouter()


@router.get("/search", response_model=list[SearchResult])
async def search_sheets(
    q: str = Query(..., description="Natural language or keyword query"),
    project_id: str = Query(...),
    discipline: str | None = Query(None),
    limit: int = Query(5, ge=1, le=20),
):
    engine = SemanticSearchEngine(get_qdrant_client())
    return await engine.search(
        query=q,
        project_id=project_id,
        discipline_filter=discipline,
        top_k=limit,
    )
