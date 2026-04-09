import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.workers.celery_app import celery_app, ingest_pdf_task

router = APIRouter()


@router.post("/projects/{project_id}/ingest")
async def ingest_pdf(
    project_id: str,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    """Accept a PDF upload, save it, and enqueue background ingestion."""
    upload_dir = os.path.join(settings.local_storage_path, project_id, "uploads")
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(upload_dir, filename)

    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    task = ingest_pdf_task.delay(pdf_path, project_id)
    return {"task_id": task.id, "status": "queued", "project_id": project_id}


@router.get("/projects/{project_id}/ingest/{task_id}")
async def ingest_status(project_id: str, task_id: str):
    """Poll ingestion task status."""
    result = celery_app.AsyncResult(task_id)
    if result.state == "PENDING":
        return {"status": "pending", "progress_pct": 0}
    elif result.state == "SUCCESS":
        data = result.result or {}
        return {"status": "complete", **data}
    elif result.state == "FAILURE":
        return {"status": "failed", "error": str(result.info)}
    return {"status": result.state.lower()}
