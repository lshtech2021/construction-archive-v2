from __future__ import annotations

import uuid
from typing import Any

from PIL import Image
from fastembed import SparseTextEmbedding
from colpali_engine.models import ColPali, ColPaliProcessor
import torch
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    MultiVectorComparator,
    MultiVectorConfig,
    Prefetch,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.http.models import FusionQuery, Fusion

from app.config import settings
from app.schemas.search import SearchResult


class SemanticSearchEngine:
    COLLECTION = settings.qdrant_collection_name
    COLPALI_DIM = 128

    def __init__(self, qdrant: AsyncQdrantClient) -> None:
        self._qdrant = qdrant
        device = settings.colpali_device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._processor = ColPaliProcessor.from_pretrained(settings.colpali_model_name)
        self._colpali = ColPali.from_pretrained(
            settings.colpali_model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self._colpali.eval()

        self._sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    async def setup_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        existing = await self._qdrant.collection_exists(self.COLLECTION)
        if existing:
            return

        await self._qdrant.create_collection(
            collection_name=self.COLLECTION,
            vectors_config={
                "colpali_visual": VectorParams(
                    size=self.COLPALI_DIM,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
            sparse_vectors_config={
                "sparse_keyword": SparseVectorParams()
            },
        )
        await self._qdrant.create_payload_index(
            collection_name=self.COLLECTION,
            field_name="discipline",
            field_schema="keyword",
        )
        await self._qdrant.create_payload_index(
            collection_name=self.COLLECTION,
            field_name="project_id",
            field_schema="keyword",
        )

    def _embed_image(self, image_path: str) -> list[list[float]]:
        """Generate ColPali multi-vector embeddings from a blueprint image."""
        image = Image.open(image_path).convert("RGB")
        batch = self._processor.process_images([image])
        with torch.no_grad():
            embeddings = self._colpali(**batch)
        # Shape: [1, num_patches, 128] — return list of patch vectors
        return embeddings[0].float().cpu().tolist()

    def _embed_query_image(self, query: str) -> list[list[float]]:
        """Embed a text query using ColPali's query encoder."""
        batch = self._processor.process_queries([query])
        with torch.no_grad():
            embeddings = self._colpali(**batch)
        return embeddings[0].float().cpu().tolist()

    def _embed_text_sparse(self, text: str) -> SparseVector:
        """Generate SPLADE sparse vector for keyword matching."""
        embeddings = list(self._sparse_model.query_embed(text))
        emb = embeddings[0]
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    async def index_sheet(
        self,
        sheet_id: str,
        image_path: str,
        ocr_text: str,
        discipline: str,
        project_id: str,
    ) -> None:
        """Upsert a sheet's visual + keyword embeddings into Qdrant."""
        multi_vectors = self._embed_image(image_path)
        sparse_vec = self._embed_text_sparse(ocr_text)

        # Deterministic integer ID from UUID
        point_id = uuid.UUID(sheet_id).int % (2**63)

        await self._qdrant.upsert(
            collection_name=self.COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        "colpali_visual": multi_vectors,
                        "sparse_keyword": sparse_vec,
                    },
                    payload={
                        "sheet_id": sheet_id,
                        "discipline": discipline,
                        "project_id": project_id,
                        "image_path": image_path,
                    },
                )
            ],
        )

    async def search(
        self,
        query: str,
        project_id: str,
        discipline_filter: str | None = None,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Hybrid retrieval: ColPali visual + SPLADE keyword with RRF fusion."""
        query_multi_vec = self._embed_query_image(query)
        query_sparse = self._embed_text_sparse(query)

        # Build filter
        must_conditions: list[Any] = [
            FieldCondition(key="project_id", match=MatchValue(value=project_id))
        ]
        if discipline_filter:
            must_conditions.append(
                FieldCondition(key="discipline", match=MatchValue(value=discipline_filter))
            )
        search_filter = Filter(must=must_conditions)

        results = await self._qdrant.query_points(
            collection_name=self.COLLECTION,
            prefetch=[
                Prefetch(
                    query=query_multi_vec,
                    using="colpali_visual",
                    filter=search_filter,
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=query_sparse,
                    using="sparse_keyword",
                    filter=search_filter,
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )

        return [
            SearchResult(
                score=r.score,
                sheet_id=r.payload["sheet_id"],
                sheet_number=r.payload.get("sheet_number", ""),
                sheet_title=r.payload.get("sheet_title", ""),
                discipline=r.payload["discipline"],
                dzi_path=r.payload.get("dzi_path", ""),
                image_path=r.payload["image_path"],
            )
            for r in results.points
        ]
