from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import settings
from app.api.v1 import ingest, search, chat, sheets, diff

app = FastAPI(title="Construction Archive API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/v1", tags=["ingestion"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(sheets.router, prefix="/api/v1", tags=["sheets"])
app.include_router(diff.router, prefix="/api/v1", tags=["diff"])

# Serve DZI tiles and images from archive_output
storage_path = settings.local_storage_path
os.makedirs(storage_path, exist_ok=True)
app.mount("/files", StaticFiles(directory=storage_path), name="files")


@app.get("/")
async def health():
    return {"status": "ok"}
