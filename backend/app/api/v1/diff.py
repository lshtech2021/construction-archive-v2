import os
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.models.project import Sheet
from app.schemas.diff import RevisionDiffRequest
from app.workers.celery_app import celery_app

router = APIRouter()


def _run_diff(path_v1: str, path_v2: str, output_dir: str) -> dict:
    from app.services.diff.revision_compare import RevisionComparator
    result = RevisionComparator().compare(path_v1, path_v2, output_dir)
    return {
        "diff_dzi_path": result.diff_dzi_path,
        "similarity_score": result.similarity_score,
        "change_count": result.change_count,
    }


@celery_app.task(name="compare_revisions")
def compare_revisions_task(path_v1: str, path_v2: str, output_dir: str) -> dict:
    return _run_diff(path_v1, path_v2, output_dir)


@router.post("/diff")
async def start_diff(request: RevisionDiffRequest, db: AsyncSession = Depends(get_db)):
    """Enqueue a revision comparison task."""
    for sheet_id in [request.sheet_id_v1, request.sheet_id_v2]:
        result = await db.execute(select(Sheet).where(Sheet.id == uuid.UUID(sheet_id)))
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Sheet {sheet_id} not found")

    v1_result = await db.execute(select(Sheet).where(Sheet.id == uuid.UUID(request.sheet_id_v1)))
    v2_result = await db.execute(select(Sheet).where(Sheet.id == uuid.UUID(request.sheet_id_v2)))
    v1 = v1_result.scalar_one()
    v2 = v2_result.scalar_one()

    output_dir = os.path.join(
        settings.local_storage_path,
        request.project_id,
        "diffs",
        f"{request.sheet_id_v1}_vs_{request.sheet_id_v2}",
    )

    task = compare_revisions_task.delay(v1.image_path, v2.image_path, output_dir)
    return {"task_id": task.id, "status": "queued"}


@router.get("/diff/{task_id}")
async def get_diff_status(task_id: str):
    """Poll revision comparison task."""
    result = celery_app.AsyncResult(task_id)
    if result.state == "SUCCESS":
        return {"status": "complete", **result.result}
    elif result.state == "FAILURE":
        return {"status": "failed", "error": str(result.info)}
    return {"status": result.state.lower()}
