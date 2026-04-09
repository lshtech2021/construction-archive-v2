from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.graph.graph_manager import GraphManager

router = APIRouter()


@router.get("/sheets/{sheet_id}/links")
async def get_sheet_links(sheet_id: str, db: AsyncSession = Depends(get_db)):
    """Return callout hyperlinks originating from a sheet (for frontend overlay)."""
    manager = GraphManager()
    return await manager.get_linked_sheets(sheet_id, db)
