from functools import lru_cache

from qdrant_client import AsyncQdrantClient

from app.config import settings


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url)
