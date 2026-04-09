from pydantic import BaseModel


class RevisionDiffRequest(BaseModel):
    project_id: str
    sheet_id_v1: str
    sheet_id_v2: str


class RevisionDiffResult(BaseModel):
    diff_dzi_path: str
    similarity_score: float
    change_count: int
