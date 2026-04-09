from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat.rag_orchestrator import RAGOrchestrator

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    orchestrator = RAGOrchestrator()
    return await orchestrator.answer(request, db)
