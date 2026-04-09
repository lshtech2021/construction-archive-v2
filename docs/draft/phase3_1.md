


To implement **Phase 3: Semantic Sheet Search Engine**, we transition from simply storing text to genuinely "understanding" the visual and textual context of the blueprints.

This phase relies heavily on **ColPali** (a revolutionary Vision-Language Model for document retrieval) and **Qdrant** (an advanced Vector Database capable of handling Late-Interaction multi-vectors, sparse vectors, and metadata filtering simultaneously).

### Why ColPali + Hybrid Search?
*   Standard OCR extraction loses spatial context (e.g., it extracts the word "EGRESS", but doesn't know it's pointing to a specific door symbol). 
*   **ColPali** processes the *image directly* in patches. It understands that the visual symbol of a door next to the text "EGRESS" means "Emergency Exit."
*   However, AI embeddings are bad at exact alphanumeric matches. If a user searches for `"AHU-04"`, a pure semantic search might return `"AHU-05"`. This is why we combine ColPali with **BM25 (Sparse Keyword Search)** to guarantee exact matches.

### 1. Prerequisites & Installation

```bash
pip install qdrant-client torch transformers colpali-engine fastembed pillow
```

### 2. Python Implementation (`search_engine.py`)

This module initializes the vector database, indexes the documents (called during Phase 1/2), and performs the hybrid search.

```python
import os
import torch
from typing import List, Optional
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColPali, ColPaliProcessor
from fastembed import SparseTextEmbedding

class SemanticSearchEngine:
    def __init__(self, collection_name: str = "construction_blueprints"):
        self.collection_name = collection_name
        
        # 1. Initialize Qdrant Client (Using local memory for demo; use URL for production)
        self.qdrant = QdrantClient(":memory:")
        
        # 2. Load ColPali (Late-Interaction Visual Embedder)
        print("Loading ColPali Model (Visual Semantic Search)...")
        self.colpali_model_name = "vidore/colpali-v1.2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = ColPaliProcessor.from_pretrained(self.colpali_model_name)
        self.model = ColPali.from_pretrained(self.colpali_model_name, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()

        # 3. Load Sparse Embedder (BM25 Equivalent for exact keyword matches)
        print("Loading Sparse Model (Exact Keyword Search)...")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        
        # Initialize the database schema
        self._setup_vector_db()

    def _setup_vector_db(self):
        """Creates the Qdrant Collection with multi-vector (ColPali) and sparse (BM25) support."""
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                # ColPali uses multi-vector (late interaction) representation, not a single dense vector
                vectors_config={
                    "colpali_visual": models.VectorParams(
                        size=128, # ColPali base dimension
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                },
                sparse_vectors_config={
                    "sparse_keyword": models.SparseVectorParams()
                }
            )
            # Index the discipline field for ultra-fast pre-filtering
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="discipline",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    # ==========================================
    # INGESTION: Indexing Sheet Images + Text
    # ==========================================
    def index_sheet(self, sheet_id: str, image_path: str, ocr_text: str, discipline: str):
        """
        Called during Phase 1/2. Converts the blueprint image into ColPali multi-vectors
        and the OCR text into sparse keyword vectors.
        """
        print(f"Indexing {sheet_id}...")
        
        # 1. Generate ColPali Visual Embeddings
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            batch_images = self.processor.process_images([image]).to(self.device)
            # Returns a list of vectors (one per image patch)
            image_multivector = self.model(**batch_images)[0].cpu().numpy().tolist()

        # 2. Generate Sparse (BM25) Embeddings from OCR text
        # FastEmbed returns a generator, we take the first element
        sparse_embedding = list(self.sparse_model.embed([ocr_text]))[0]

        # 3. Insert into Qdrant Vector DB
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=hash(sheet_id) & ((1<<63)-1), # Generate numeric ID
                    payload={
                        "sheet_id": sheet_id,
                        "discipline": discipline,
                        "image_path": image_path
                    },
                    vector={
                        "colpali_visual": image_multivector,
                        "sparse_keyword": models.SparseVector(
                            indices=sparse_embedding.indices.tolist(),
                            values=sparse_embedding.values.tolist()
                        )
                    }
                )
            ]
        )

    # ==========================================
    # RETRIEVAL: Hybrid Search + Discipline Filter
    # ==========================================
    def search(self, query: str, discipline_filter: Optional[str] = None, top_k: int = 3):
        """
        Takes a natural language query and finds the most relevant blueprints 
        using visual semantics AND exact keyword matching.
        """
        # 1. Embed Query for ColPali (Dense/Visual)
        with torch.no_grad():
            batch_queries = self.processor.process_queries([query]).to(self.device)
            query_multivector = self.model(**batch_queries)[0].cpu().numpy().tolist()

        # 2. Embed Query for Sparse (Keywords)
        sparse_query = list(self.sparse_model.query_embed(query))[0]

        # 3. Setup Discipline Pre-Filtering (Reduces noise significantly)
        search_filters = None
        if discipline_filter:
            search_filters = models.Filter(
                must=[
                    models.FieldCondition(
                        key="discipline",
                        match=models.MatchValue(value=discipline_filter)
                    )
                ]
            )

        # 4. Perform Hybrid Search via Qdrant's Reciprocal Rank Fusion (RRF)
        # Prefetch queries allow us to run dense and sparse searches simultaneously and merge results.
        prefetch_queries =[
            models.Prefetch(
                query=query_multivector,
                using="colpali_visual",
                limit=top_k,
                filter=search_filters
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist()
                ),
                using="sparse_keyword",
                limit=top_k,
                filter=search_filters
            )
        ]

        # Execute fusion search
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=models.FusionQuery(fusion=models.Fusion.RRF), # Reciprocal Rank Fusion
            limit=top_k,
            with_payload=True
        )

        # 5. Format Output
        retrieved_sheets =[]
        for point in results.points:
            retrieved_sheets.append({
                "sheet_id": point.payload["sheet_id"],
                "discipline": point.payload["discipline"],
                "score": point.score
            })
            
        return retrieved_sheets

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    search_engine = SemanticSearchEngine()

    # 1. Simulate Ingestion (Phase 1/2)
    # Assume we have an image "A-101.jpg" and its extracted OCR text
    search_engine.index_sheet(
        sheet_id="A-101",
        image_path="./mock_plans/A-101.jpg", 
        ocr_text="FLOOR PLAN. CORRIDOR. ROOM 101. EXIT DOORS. EGRESS ROUTE.",
        discipline="Architectural"
    )
    
    search_engine.index_sheet(
        sheet_id="M-201",
        image_path="./mock_plans/M-201.jpg",
        ocr_text="MECHANICAL HVAC PLAN. AHU-04. CHILLED WATER LINES.",
        discipline="Mechanical"
    )

    # 2. Semantic Search (Concept matching: EGRESS visual -> Emergency Exit query)
    print("\n--- Searching for 'Emergency Exits' ---")
    results = search_engine.search(query="Where are the emergency exits?", top_k=2)
    for r in results:
        print(f"Found: {r['sheet_id']} (Discipline: {r['discipline']}) - Score: {r['score']}")

    # 3. Exact Keyword Search + Discipline Filtering
    print("\n--- Searching for 'AHU-04' with Mechanical Filter ---")
    results = search_engine.search(query="Details for AHU-04", discipline_filter="Mechanical", top_k=2)
    for r in results:
        print(f"Found: {r['sheet_id']} (Discipline: {r['discipline']}) - Score: {r['score']}")
```

### 3. Deep Dive into the Search Architecture

#### A. Multi-Vector (Late Interaction) via ColPali
Traditional Dense embeddings (like standard OpenAI `text-embedding-3`) compress an entire document into a *single* vector. For a massive, dense construction blueprint, condensing 5,000 architectural elements into one array destroys all spatial nuance.
*   **How ColPali fixes this:** ColPali slices the 8K resolution blueprint into hundreds of visual patches. It feeds these patches through a Vision Transformer and outputs **multiple vectors per document**.
*   **How Qdrant handles it:** We configure Qdrant with `models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)`. During search, the database computes the maximum similarity between the user's query vectors and *every individual patch* of the blueprint, retrieving sheets where even a tiny 256x256 pixel quadrant perfectly matches the query.

#### B. Reciprocal Rank Fusion (RRF)
When a user searches for `"AHU-04 Chilled Water Lines"`:
1.  **Dense Search** finds sheets that look visually like HVAC routing plans (even if the OCR failed to read a blurry text line).
2.  **Sparse Search (Splade/BM25)** ensures that sheets containing the exact string `"AHU-04"` are heavily weighted.
3.  **RRF (Fusion)** mathematically combines the rank lists from both searches, ensuring that documents that score high in *both* semantic meaning and exact keyword match bubble to the absolute top.

#### C. Payload Pre-Filtering
Construction queries are often implicitly discipline-scoped. If a user asks *"Where are the 4-inch cast iron pipes?"*, you want to completely ignore Architectural ceiling grids that happen to look like pipes. 
Because Qdrant indexes the `discipline` metadata as a `PayloadSchemaType.KEYWORD`, passing a `models.Filter` prevents the vector engine from even calculating distances for Architectural or Civil sheets, saving immense compute time and eliminating false positives.